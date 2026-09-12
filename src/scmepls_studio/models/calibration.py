from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

from ..version import __version__


@dataclass(frozen=True)
class CalibrationResult:
    algorithm: str
    target: str
    feature_names: list[str]
    validation_mode: str
    group_column: str | None
    r2: float
    mae: float
    rmse: float
    sample_count: int
    validation_sample_count: int
    feature_importance: dict[str, float]
    leakage_warnings: tuple[str, ...]
    data_fingerprint_sha256: str
    model: object


@dataclass(frozen=True)
class ExternalValidationResult:
    target: str
    sample_count: int
    r2: float
    mae: float
    rmse: float
    dataset_fingerprint_sha256: str
    residual_p05: float
    residual_p95: float


def _make_model(algorithm: str) -> object:
    if algorithm == "Gradient Boosting": return GradientBoostingRegressor(random_state=42)
    if algorithm == "Random Forest": return RandomForestRegressor(n_estimators=300,random_state=42,min_samples_leaf=2)
    raise ValueError(f"Unsupported calibration algorithm: {algorithm}")


def _fingerprint(df: pd.DataFrame) -> str:
    hashed=pd.util.hash_pandas_object(df,index=True).to_numpy(dtype=np.uint64);return hashlib.sha256(hashed.tobytes()).hexdigest()


def _leakage_warnings(clean: pd.DataFrame, features: list[str], target: str) -> tuple[str,...]:
    warnings=[]
    target_values=clean[target].to_numpy(dtype=float)
    for name in features:
        x=clean[name].to_numpy(dtype=float)
        if np.std(x)<=1e-15 or np.std(target_values)<=1e-15:continue
        corr=float(np.corrcoef(x,target_values)[0,1])
        if np.isfinite(corr) and abs(corr)>0.9999:
            warnings.append(f"Feature '{name}' has |correlation|={abs(corr):.6f} with target; verify it is not target-derived/leaked information.")
    return tuple(warnings)


def train_regressor(df:pd.DataFrame,target:str,algorithm:str="Random Forest",feature_names:list[str]|None=None,validation_mode:str="KFold",group_column:str|None=None)->CalibrationResult:
    if target not in df.columns:raise ValueError(f"Target column '{target}' was not found.")
    if group_column and group_column not in df.columns:raise ValueError(f"Group column '{group_column}' was not found.")
    numeric_columns=set(df.select_dtypes(include=[np.number]).columns)
    if target not in numeric_columns:raise ValueError("Target must be numeric.")
    features=[c for c in df.columns if c in numeric_columns and c not in {target,group_column}] if feature_names is None else [str(c) for c in feature_names]
    if not features:raise ValueError("At least one numeric feature column is required.")
    if target in features:raise ValueError("The target column cannot also be used as an input feature.")
    if group_column and group_column in features:raise ValueError("The group column cannot also be used as an input feature.")
    missing=[c for c in features if c not in df.columns]
    if missing:raise ValueError(f"Unknown feature columns: {missing}")
    non_numeric=[c for c in features if c not in numeric_columns]
    if non_numeric:raise ValueError(f"Calibration features must be numeric: {non_numeric}")
    required=[*features,target]+([group_column] if group_column else []);clean=df[required].dropna(axis=0).copy()
    if len(clean)<10:raise ValueError("At least 10 complete rows are required; 20 or more are recommended.")
    X=clean[features];y=clean[target];model=_make_model(algorithm);predicted=np.full(len(clean),np.nan,dtype=float)
    if group_column:
        groups=clean[group_column];unique_groups=int(groups.nunique())
        if unique_groups<2:raise ValueError("Grouped validation requires at least two distinct groups.")
        splits=min(5,unique_groups);splitter=GroupKFold(n_splits=splits);split_iter=splitter.split(X,y,groups);effective_validation_mode="Grouped KFold"
    elif validation_mode=="Chronological":
        splits=min(5,max(2,len(clean)//5));splitter=TimeSeriesSplit(n_splits=splits);split_iter=splitter.split(X,y);effective_validation_mode="Chronological"
    elif validation_mode=="KFold":
        splits=min(5,max(2,len(clean)//4));splitter=KFold(n_splits=splits,shuffle=True,random_state=42);split_iter=splitter.split(X,y);effective_validation_mode="KFold"
    else:raise ValueError("Validation mode must be 'KFold' or 'Chronological'.")
    for train_index,test_index in split_iter:
        fold_model=clone(model);fold_model.fit(X.iloc[train_index],y.iloc[train_index]);predicted[test_index]=fold_model.predict(X.iloc[test_index])
    valid=np.isfinite(predicted)
    if int(np.count_nonzero(valid))<2:raise ValueError("Validation split produced too few out-of-sample predictions.")
    y_valid=y.iloc[np.where(valid)[0]].to_numpy(dtype=float);pred_valid=predicted[valid];model.fit(X,y);importances=getattr(model,"feature_importances_",np.zeros(len(features)))
    return CalibrationResult(algorithm=algorithm,target=target,feature_names=features,validation_mode=effective_validation_mode,group_column=group_column,r2=float(r2_score(y_valid,pred_valid)),mae=float(mean_absolute_error(y_valid,pred_valid)),rmse=float(np.sqrt(np.mean((y_valid-pred_valid)**2))),sample_count=len(clean),validation_sample_count=int(np.count_nonzero(valid)),feature_importance={name:float(value) for name,value in zip(features,importances)},leakage_warnings=_leakage_warnings(clean,features,target),data_fingerprint_sha256=_fingerprint(clean[required]),model=model)


def evaluate_external_validation(result:CalibrationResult,df:pd.DataFrame)->ExternalValidationResult:
    required=[*result.feature_names,result.target];missing=[name for name in required if name not in df.columns]
    if missing:raise ValueError(f"External validation dataset is missing columns: {missing}")
    clean=df[required].dropna(axis=0).copy()
    if len(clean)<2:raise ValueError("External validation requires at least two complete rows.")
    for name in required:
        if not pd.api.types.is_numeric_dtype(clean[name]):raise ValueError(f"External validation column '{name}' must be numeric.")
    y=clean[result.target].to_numpy(dtype=float);pred=np.asarray(result.model.predict(clean[result.feature_names]),dtype=float);residual=y-pred
    return ExternalValidationResult(target=result.target,sample_count=len(clean),r2=float(r2_score(y,pred)),mae=float(mean_absolute_error(y,pred)),rmse=float(np.sqrt(np.mean(residual**2))),dataset_fingerprint_sha256=_fingerprint(clean),residual_p05=float(np.quantile(residual,0.05)),residual_p95=float(np.quantile(residual,0.95)))


def save_calibration(result:CalibrationResult,path:str|Path)->None:
    payload={"format_version":3,"saved_utc":datetime.now(timezone.utc).isoformat(),"scmepls_version":__version__,"numpy_version":np.__version__,"scikit_learn_version":sklearn.__version__,"algorithm":result.algorithm,"target":result.target,"feature_names":result.feature_names,"validation_mode":result.validation_mode,"group_column":result.group_column,"r2":result.r2,"mae":result.mae,"rmse":result.rmse,"sample_count":result.sample_count,"validation_sample_count":result.validation_sample_count,"feature_importance":result.feature_importance,"leakage_warnings":result.leakage_warnings,"data_fingerprint_sha256":result.data_fingerprint_sha256,"model":result.model};joblib.dump(payload,Path(path))

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal

ModelType = Literal["first_order", "second_order", "mass_damper_pid"]


@dataclass(frozen=True)
class ResponseParameters:
    model_type: ModelType = "first_order"
    gain: float = 1.0
    time_constant_s: float = 1.0
    natural_frequency_rad_s: float = 2.0
    damping_ratio: float = 0.9
    mass_kg: float = 75.0
    damping_ns_m: float = 150.0
    actuator_gain: float = 1.0
    kp: float = 90.0
    ki: float = 15.0
    kd: float = 60.0
    derivative_filter_rad_s: float = 100.0


@dataclass(frozen=True)
class ResponseMetrics:
    rise_time_s: float
    settling_time_s: float
    overshoot_percent: float
    steady_state_value: float
    steady_state_error: float
    iae: float
    itae: float


def _trapezoidal_integral(values: np.ndarray, time_s: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None: return float(trapezoid(values, time_s))
    return float(np.trapz(values, time_s))


def transfer_function(params: ResponseParameters) -> signal.TransferFunction:
    if params.model_type == "first_order":
        if params.time_constant_s <= 0: raise ValueError("Time constant must be positive.")
        return signal.TransferFunction([params.gain], [params.time_constant_s, 1.0])
    if params.model_type == "second_order":
        if params.natural_frequency_rad_s <= 0 or params.damping_ratio <= 0: raise ValueError("Natural frequency and damping ratio must be positive.")
        wn=params.natural_frequency_rad_s
        return signal.TransferFunction([params.gain*wn**2],[1.0,2.0*params.damping_ratio*wn,wn**2])
    if params.model_type == "mass_damper_pid":
        if params.mass_kg<=0 or params.actuator_gain<=0: raise ValueError("Mass and actuator gain must be positive.")
        if params.derivative_filter_rad_s<=0 or not np.isfinite(params.derivative_filter_rad_s): raise ValueError("Derivative-filter corner frequency must be finite and positive.")
        if min(params.kp,params.ki,params.kd)<0: raise ValueError("PID gains must be non-negative.")
        # C(s)=Kp + Ki/s + Kd*N*s/(s+N). The derivative roll-off avoids the
        # non-physical infinite high-frequency gain of an ideal derivative.
        ka=float(params.actuator_gain); n=float(params.derivative_filter_rad_s); m=float(params.mass_kg); b=float(params.damping_ns_m)
        controller_num=np.array([params.kp+params.kd*n, params.kp*n+params.ki, params.ki*n],dtype=float)
        numerator=ka*controller_num
        plant_controller_den=np.array([m,m*n+b,b*n,0.0,0.0],dtype=float)
        closed_den=plant_controller_den.copy(); closed_den[2:]+=numerator
        return signal.TransferFunction(numerator,closed_den)
    raise ValueError(f"Unsupported model type: {params.model_type}")


def simulate_step(params: ResponseParameters,duration_s:float=10.0,points:int=2000)->tuple[np.ndarray,np.ndarray]:
    if duration_s<=0 or points<100: raise ValueError("Duration must be positive and points must be at least 100.")
    system=transfer_function(params);t=np.linspace(0.0,duration_s,points);tout,y=signal.step(system,T=t);return np.asarray(tout),np.asarray(y)


def calculate_response_metrics(t:np.ndarray,y:np.ndarray,reference:float=1.0)->ResponseMetrics:
    t=np.asarray(t,dtype=float);y=np.asarray(y,dtype=float)
    if t.size!=y.size or t.size<10:raise ValueError("Step response arrays must have equal length and at least 10 points.")
    if not np.isfinite(t).all() or not np.isfinite(y).all():raise ValueError("Step response arrays must contain only finite values.")
    if np.any(np.diff(t)<=0):raise ValueError("Step response time values must be strictly increasing.")
    final=float(np.mean(y[-max(10,y.size//50):]));target_for_rise=final if abs(final)>1e-12 else reference;low,high=0.1*target_for_rise,0.9*target_for_rise
    try:t10=float(t[np.where(y>=low)[0][0]]);t90=float(t[np.where(y>=high)[0][0]]);rise=max(0.0,t90-t10)
    except IndexError:rise=float("nan")
    band=0.02*max(abs(target_for_rise),1e-9);outside=np.where(np.abs(y-target_for_rise)>band)[0];settling=float(t[outside[-1]+1]) if outside.size and outside[-1]+1<t.size else 0.0;peak=float(np.max(y));overshoot=max(0.0,100.0*(peak-target_for_rise)/max(abs(target_for_rise),1e-9));error=reference-y;iae=_trapezoidal_integral(np.abs(error),t);itae=_trapezoidal_integral(t*np.abs(error),t)
    return ResponseMetrics(rise_time_s=rise,settling_time_s=settling,overshoot_percent=overshoot,steady_state_value=final,steady_state_error=float(reference-final),iae=iae,itae=itae)

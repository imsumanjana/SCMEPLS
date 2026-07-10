# Data formats

## Vibration CSV

Required structure:

```csv
time_s,Crawler Transporter,Rail Platform,Maglev Platform
0.000,0.01,0.02,0.00
0.005,0.03,0.01,0.01
```

- `time_s` must be strictly increasing.
- Every other numeric column is interpreted as acceleration in m/s².
- Sampling should be sufficiently high for the frequency range of interest.

## AI calibration CSV

Use one row per independent experiment or validated simulation run. Include numeric input features and one or more numeric target columns, for example:

```csv
mass_kg,speed_mps,gap_mm,wind_n,controller_kp,rms_accel_mps2,settling_time_s,net_energy_kwh
75,0.05,10,0,90,0.11,2.4,0.008
```

Select one target at a time in the GUI. Avoid mixing repeated samples from the same run across training and validation folds without grouping; doing so can overestimate model quality.

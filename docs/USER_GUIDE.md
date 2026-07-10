# User guide

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Run `pip install -r requirements.txt`.
4. Start the application using `python main.py`.

## Recommended workflow

1. **Vibration:** import accelerometer data or generate illustrative data. Verify units, RMS, crest factor, and dominant frequency. Export the normalized RMS plot at 600 or 1200 dpi.
2. **Control Response:** choose an equivalent or mass–damper PID model. Use identified parameters where available. Review rise time, settling time, overshoot, IAE, and ITAE.
3. **Energy:** enter a complete energy balance for each transport method. Prefer the specific-energy plot for scientific reporting.
4. **Radar / MCDA:** replace default scores with a documented evidence-based rubric. Keep every axis oriented so higher is better.
5. **SC-MEPLS Simulation:** adjust reduced-order model parameters and fault scenarios. Export the dashboard and time-history CSV.
6. **AI Calibration:** import at least 20 traceable runs, train a model, and reject weak cross-validation results.
7. **Batch export:** use File > Export All Current Article Figures. The program creates 600 dpi PNGs, CSV result tables, and metadata JSON sidecars.

## Article wording

For illustrative outputs, use: “The figures present normalized technology-level comparisons for feasibility assessment and do not represent direct experimental measurements.”

For simulation outputs, report all model equations, parameters, solver settings, validation metrics, and limitations.

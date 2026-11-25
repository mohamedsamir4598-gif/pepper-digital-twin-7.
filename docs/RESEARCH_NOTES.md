# Research notes (for Mohamed)

## 1. Is it a scientific contribution?
A digital twin for irrigation is not novel by itself. The novelty has to come from a specific, testable claim. Candidates this platform lets you test:

1. **Joint irrigation + fertigation control from drainage feedback** (EC, N, P, K) rather than irrigation alone, in a stage-based model.
2. **Weight-based state assimilation** (daily scale readings correct the model and adapt Kc) versus an open-loop schedule.
3. **A calibrated, validated process model for pepper under a full water x nutrient factorial**, with leave-one-replicate-out skill versus a quadratic response surface.
4. **A fair comparison** of an automatic twin against (a) the grower's practice and (b) the best fixed combination found in your own factorial.

Pick one or two as the research questions. Without a clear question the work stays a software project.

## 2. The comparison that will be asked about
The virtual trial showed that the twin's advantage depends on the baseline: against a fixed 100 % ETc / full-recipe schedule it
raised profit a lot; against a tuned fixed schedule (120 % ETc, 0.7 recipe) profit was similar, with the twin using less water
and nutrients. In the real study the best cell of the 4 x 4 factorial is the fair "traditional" benchmark.

## 3. Statistics
* n = 3 replicates per cell. The scale is the unit; the 3 pots are subsamples.
* Split calibration/validation by replicate or by whole treatments, never by random rows (daily data are autocorrelated).
* Keep the final twin-vs-traditional trial independent of the calibration data (another season or another site).

## 4. What to measure well
* Tare: weigh each scale once after saturation and free drainage (full container capacity); set Tare in Site setup.
* Weigh plants (fresh mass) every 2-3 weeks to replace the assumed plant-mass curve.
* Drainage sample (N, P, K, EC, pH) at least every 2-3 days with the drained volume.
* Record exactly what was applied (volume, concentrations), not what was planned.
* Final yield per scale.

## 5. Models
Used in the platform: process water/nutrient balance, stage-weighted yield function, scalar Kalman update, receding-horizon search, differential evolution.
Use outside, then import: **DSSAT (CROPGRO-Pepper)** and **HYDRUS-1D** for scenarios, **AquaCrop** as an independent crop model for the same plans (model comparison is itself a result).
Surrogates only emulate the simulator they were trained on; the twin itself already runs in milliseconds.

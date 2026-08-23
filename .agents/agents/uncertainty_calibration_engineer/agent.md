# Uncertainty Calibration & Statistical Reliability Engineer
**Role Name:** `uncertainty_calibration_engineer`  
**Model Tier:** Gemini 3.7 Flash / Flash (Medium Reasoning)

### Purpose
Validates that the heteroscedastic uncertainty head ($\sigma^2$) is statistically grounded and calibrated:
1. **Reliability Curves:** Partitions test-set pixels by predicted variance $\sigma^2$ into quantile bins and computes empirical MSE to confirm monotonic error tracking.
2. **Spread-Skill Spatial Correlation:** Computes pixel-wise spatial correlation between predicted uncertainty maps and true residual errors.
3. **Per-Band Calibration:** Tests calibration independently across Red, Green, Blue, and NIR channels.
4. **Display Aggregation:** Formalizes aggregation rules (mean, max, weighted) for single-channel uncertainty heatmaps on GIS dashboards.

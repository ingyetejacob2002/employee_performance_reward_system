# Employee Performance Reward System

This application implements the statistical algorithms specified in `Task.docx`. It is **not a machine-learning system**.

## Algorithms

1. Distribution selection using skewness and Shapiro-Wilk:
   - `|skewness| < 0.5` and `p > alpha` → Normal → confidence-adjusted Z-score.
   - Otherwise → Non-normal/skewed → confidence-adjusted IQR.
2. Confidence-adjusted Z-score using the t critical value and `Z_CI = t_critical * sqrt(1 + 1/n)`.
3. Confidence-adjusted IQR using Q1, Q3, IQR, z critical value, `delta = z_critical/sqrt(n)`, and the bounds in the task.

The task document permits `k = 1.5` or `k = 3`; the interface lets the user choose either.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

CSV must contain:

```text
StaffID,Score
1,67
2,88
```

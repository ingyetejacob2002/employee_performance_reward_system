# Employee Performance Reward System

A statistical employee appraisal analysis application based on the algorithms specified in the project document.

## Run

```powershell
python -m streamlit run app.py
```

## CSV format

```csv
StaffID,Score
001,67
002,88
003,99
004,87
```

StaffID values are displayed as three-digit identifiers (001, 002, 003, ...).

## Decision process

1. Calculate skewness and perform the Shapiro-Wilk test.
2. If `|skewness| < 0.5` and `p-value > alpha`, use the confidence-adjusted Z-score method.
3. Otherwise use the confidence-adjusted IQR method.
4. Separate positive and negative outliers.
5. Positive outliers are marked reward eligible.
6. Negative outliers are retained but marked not eligible for reward because they represent very poor performance.
7. Results are displayed in descending order of Score.

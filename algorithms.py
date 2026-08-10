"""Statistical algorithms for the Employee Performance Reward System.

Implements the algorithms specified in Task.docx without machine-learning models.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DistributionResult:
    skewness: float
    shapiro_statistic: float
    shapiro_p_value: float
    is_normal: bool
    decision: str
    method: str


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the required StaffID and Score columns and clean missing rows."""
    required = {"StaffID", "Score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    out = df[["StaffID", "Score"]].copy()
    out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
    if out["Score"].isna().any():
        bad = int(out["Score"].isna().sum())
        raise ValueError(f"Score contains {bad} non-numeric or missing value(s).")
    if len(out) < 3:
        raise ValueError("At least 3 appraisal scores are required for the Shapiro-Wilk test.")
    if out["Score"].std(ddof=1) == 0:
        raise ValueError("All appraisal scores are identical; standard deviation is zero.")
    return out.reset_index(drop=True)


def check_distribution(scores, alpha=0.05) -> DistributionResult:
    """Listing 3.1: choose Z-score for normal data, IQR otherwise."""
    x = np.asarray(scores, dtype=float)
    skewness = float(stats.skew(x, bias=False))
    shapiro_stat, shapiro_p = stats.shapiro(x)
    is_normal = abs(skewness) < 0.5 and shapiro_p > alpha
    if is_normal:
        decision = "Normal dataset"
        method = "Confidence-adjusted Z-score"
    else:
        decision = "Non-normal/skewed dataset"
        method = "Confidence-adjusted IQR"
    return DistributionResult(
        skewness=skewness,
        shapiro_statistic=float(shapiro_stat),
        shapiro_p_value=float(shapiro_p),
        is_normal=is_normal,
        decision=decision,
        method=method,
    )


def confidence_adjusted_zscore(scores, alpha=0.05):
    """Listing 3.2: confidence-adjusted Z-score method."""
    x = np.asarray(scores, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    s = float(np.std(x, ddof=1))

    # Positive upper critical t value corresponding to alpha/2 in each tail.
    t_critical = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    z_ci = t_critical * np.sqrt(1 + 1 / n)
    z_scores = (x - mean) / s
    outlier_mask = np.abs(z_scores) > z_ci

    return {
        "mean": mean,
        "std": s,
        "n": n,
        "t_critical": t_critical,
        "threshold": float(z_ci),
        "z_scores": z_scores,
        "outlier_mask": outlier_mask,
    }


def confidence_adjusted_iqr(scores, alpha=0.05, k=1.5):
    """Listing 3.3: confidence-adjusted IQR method.

    The document gives k as 1.5 or 3; the GUI lets the user choose either.
    The decision rule in the document explicitly marks x > UB as an outlier.
    """
    x = np.asarray(scores, dtype=float)
    n = len(x)
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    iqr = q3 - q1
    z_critical = float(stats.norm.ppf(1 - alpha / 2))
    delta = z_critical / np.sqrt(n)
    lb = q1 - (k + delta) * iqr
    ub = q3 + (k + delta) * iqr
    outlier_mask = x > ub

    return {
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "n": n,
        "z_critical": z_critical,
        "delta": float(delta),
        "k": float(k),
        "lower_bound": float(lb),
        "upper_bound": float(ub),
        "outlier_mask": outlier_mask,
    }


def analyze_dataset(df, alpha=0.05, k=1.5):
    """Run the complete distribution-selection and outlier-detection pipeline."""
    data = validate_data(df)
    dist = check_distribution(data["Score"].to_numpy(), alpha=alpha)

    if dist.is_normal:
        details = confidence_adjusted_zscore(data["Score"], alpha=alpha)
        data["Z_score"] = details["z_scores"]
        data["Outlier"] = details["outlier_mask"]
    else:
        details = confidence_adjusted_iqr(data["Score"], alpha=alpha, k=k)
        data["Outlier"] = details["outlier_mask"]

    # The source document identifies exceptional staff using appraisal scores,
    # but does not define a separate reward-ranking formula. Therefore the
    # system reports detected outliers rather than inventing a reward formula.
    outliers = data[data["Outlier"]].copy()
    return data, dist, details, outliers

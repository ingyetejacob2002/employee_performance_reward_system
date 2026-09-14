"""Statistical algorithms for the Employee Performance Reward System.

Implements the statistical procedures specified in the project document.
No machine-learning model is used.
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
    """Validate StaffID and Score and prepare StaffID as 001, 002, 003, ..."""
    required = {"StaffID", "Score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(sorted(missing))}"
        )

    out = df[["StaffID", "Score"]].copy()

    if out["StaffID"].isna().any():
        raise ValueError("StaffID contains missing value(s).")

    # Staff IDs are displayed as three-digit values: 001, 002, 003, ...
    numeric_staff_ids = pd.to_numeric(out["StaffID"], errors="coerce")
    if numeric_staff_ids.isna().any():
        raise ValueError("StaffID must contain numeric staff identifiers.")
    if (numeric_staff_ids % 1 != 0).any():
        raise ValueError("StaffID values must be whole numbers.")
    if (numeric_staff_ids <= 0).any():
        raise ValueError("StaffID values must be positive.")

    out["StaffID"] = numeric_staff_ids.astype(int).map(lambda x: f"{x:03d}")

    out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
    if out["Score"].isna().any():
        bad = int(out["Score"].isna().sum())
        raise ValueError(f"Score contains {bad} non-numeric or missing value(s).")

    if len(out) < 3:
        raise ValueError(
            "At least 3 appraisal scores are required for the Shapiro-Wilk test."
        )

    if out["Score"].std(ddof=1) == 0:
        raise ValueError(
            "All appraisal scores are identical; standard deviation is zero."
        )

    return out.reset_index(drop=True)


def check_distribution(scores, alpha=0.05) -> DistributionResult:
    """Listing 3.1: select Z-score for normal data and IQR otherwise."""
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
    """Listing 3.2: confidence-adjusted Z-score method.

    Negative outlier: Zi < -ZCI
    Positive outlier: Zi > ZCI
    """
    x = np.asarray(scores, dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    s = float(np.std(x, ddof=1))

    t_critical = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    z_ci = t_critical * np.sqrt(1 + 1 / n)
    z_scores = (x - mean) / s

    negative_mask = z_scores < -z_ci
    positive_mask = z_scores > z_ci
    outlier_mask = negative_mask | positive_mask

    return {
        "mean": mean,
        "std": s,
        "n": n,
        "t_critical": t_critical,
        "threshold": float(z_ci),
        "z_scores": z_scores,
        "negative_mask": negative_mask,
        "positive_mask": positive_mask,
        "outlier_mask": outlier_mask,
    }


def confidence_adjusted_iqr(scores, alpha=0.05, k=1.5):
    """Listing 3.3: confidence-adjusted IQR method.

    Negative outlier: Xi < LB
    Positive outlier: Xi > UB
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

    negative_mask = x < lb
    positive_mask = x > ub
    outlier_mask = negative_mask | positive_mask

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
        "negative_mask": negative_mask,
        "positive_mask": positive_mask,
        "outlier_mask": outlier_mask,
    }


def analyze_dataset(df, alpha=0.05, k=1.5):
    """Run distribution selection and confidence-adjusted outlier detection."""
    data = validate_data(df)
    dist = check_distribution(data["Score"].to_numpy(), alpha=alpha)

    if dist.is_normal:
        details = confidence_adjusted_zscore(data["Score"], alpha=alpha)
        data["Z_score"] = details["z_scores"]
        data["Negative_Outlier"] = details["negative_mask"]
        data["Positive_Outlier"] = details["positive_mask"]
    else:
        details = confidence_adjusted_iqr(
            data["Score"], alpha=alpha, k=k
        )
        data["Negative_Outlier"] = details["negative_mask"]
        data["Positive_Outlier"] = details["positive_mask"]

    data["Outlier"] = data["Negative_Outlier"] | data["Positive_Outlier"]

    # Positive outliers are exceptional high performers and can be rewarded.
    # Negative outliers are retained and separately labelled because they
    # represent extremely poor performance and cannot be rewarded.
    data["Reward_Status"] = np.select(
        [data["Positive_Outlier"], data["Negative_Outlier"]],
        ["Reward Eligible", "Not Eligible - Very Poor Performance"],
        default="Not an Outlier",
    )

    # Present staff in descending order of appraisal score as requested.
    data = data.sort_values("Score", ascending=False).reset_index(drop=True)

    positive_outliers = data[data["Positive_Outlier"]].copy()
    negative_outliers = data[data["Negative_Outlier"]].copy()

    positive_outliers = positive_outliers.sort_values(
        "Score", ascending=False
    ).reset_index(drop=True)
    negative_outliers = negative_outliers.sort_values(
        "Score", ascending=False
    ).reset_index(drop=True)

    # Keep the complete detected-outlier table available to the application.
    outliers = data[data["Outlier"]].copy()

    return (
        data,
        dist,
        details,
        outliers,
        positive_outliers,
        negative_outliers,
    )

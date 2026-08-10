
import io

import pandas as pd
import streamlit as st
import plotly.express as px

from algorithms import analyze_dataset


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Employee Performance Reward System",
    page_icon="🏆",
    layout="wide"
)


# ---------------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------------
st.title("🏆 Employee Performance Reward System")

st.caption(
    "Statistical appraisal-score analysis using the methods "
    "specified in the project document."
)


# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------
with st.sidebar:
    st.header("Analysis Settings")

    alpha = st.selectbox(
        "Significance level (α)",
        [0.05, 0.01],
        index=0,
        format_func=lambda x: (
            f"{x} ({int((1 - x) * 100)}% confidence)"
        )
    )

    k = st.selectbox(
        "IQR multiplier (k)",
        [1.5, 3.0],
        index=0
    )

    st.divider()

    st.markdown("**Automatic method selection**")

    st.write(
        "Normal → confidence-adjusted Z-score"
    )

    st.write(
        "Non-normal/skewed → confidence-adjusted IQR"
    )


# ---------------------------------------------------------
# CSV UPLOAD
# ---------------------------------------------------------
uploaded = st.file_uploader(
    "Upload staff appraisal CSV",
    type=["csv"]
)


# ---------------------------------------------------------
# IF NO FILE HAS BEEN UPLOADED
# ---------------------------------------------------------
if uploaded is None:

    st.info(
        "Upload a CSV containing the columns **StaffID** and **Score**."
    )

    st.markdown("### Expected format")

    sample_data = pd.DataFrame(
        {
            "StaffID": [1, 2, 3, 4],
            "Score": [67, 88, 99, 87]
        }
    )

    st.dataframe(
        sample_data,
        width="stretch",
        hide_index=True
    )

    st.stop()


# ---------------------------------------------------------
# READ CSV
# ---------------------------------------------------------
try:

    raw = pd.read_csv(uploaded)

except Exception as exc:

    st.error(
        f"Unable to read the CSV file: {exc}"
    )

    st.stop()


# ---------------------------------------------------------
# DISPLAY UPLOADED DATASET
# ---------------------------------------------------------
st.subheader("Uploaded Dataset")

st.dataframe(
    raw,
    width="stretch",
    hide_index=True
)


# ---------------------------------------------------------
# RUN STATISTICAL ANALYSIS
# ---------------------------------------------------------
try:

    result_df, dist, details, outliers = analyze_dataset(
        raw,
        alpha=alpha,
        k=k
    )

except Exception as exc:

    st.error(str(exc))

    st.stop()


# ---------------------------------------------------------
# 1. DISTRIBUTION ANALYSIS
# ---------------------------------------------------------
st.subheader("1. Distribution Analysis")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Skewness",
    f"{dist.skewness:.4f}"
)

c2.metric(
    "Shapiro-Wilk p-value",
    f"{dist.shapiro_p_value:.4f}"
)

c3.metric(
    "Decision",
    dist.decision
)

c4.metric(
    "Selected Method",
    dist.method
)


# ---------------------------------------------------------
# DISTRIBUTION DECISION MESSAGE
# ---------------------------------------------------------
if dist.is_normal:

    st.success(
        "Normal dataset detected. "
        "The confidence-adjusted Z-score method was selected."
    )

else:

    st.warning(
        "Non-normal/skewed dataset detected. "
        "The confidence-adjusted IQR method was selected."
    )


# ---------------------------------------------------------
# DISTRIBUTION HISTOGRAM
# ---------------------------------------------------------
fig = px.histogram(
    result_df,
    x="Score",
    nbins=max(
        5,
        min(30, max(1, len(result_df) // 2))
    ),
    title="Distribution of Appraisal Scores"
)

st.plotly_chart(
    fig,
    width="stretch"
)


# ---------------------------------------------------------
# 2. OUTLIER DETECTION
# ---------------------------------------------------------
st.subheader("2. Outlier Detection")


# ---------------------------------------------------------
# Z-SCORE RESULTS
# ---------------------------------------------------------
if dist.is_normal:

    a, b, c = st.columns(3)

    a.metric(
        "Mean",
        f"{details['mean']:.4f}"
    )

    b.metric(
        "Standard deviation",
        f"{details['std']:.4f}"
    )

    c.metric(
        "Confidence-adjusted Z threshold",
        f"{details['threshold']:.4f}"
    )

    st.write(
        f"t-critical = **{details['t_critical']:.4f}**; "
        f"n = **{details['n']}**"
    )


# ---------------------------------------------------------
# IQR RESULTS
# ---------------------------------------------------------
else:

    a, b, c, d = st.columns(4)

    a.metric(
        "Q1",
        f"{details['q1']:.4f}"
    )

    b.metric(
        "Q3",
        f"{details['q3']:.4f}"
    )

    c.metric(
        "IQR",
        f"{details['iqr']:.4f}"
    )

    d.metric(
        "Upper bound",
        f"{details['upper_bound']:.4f}"
    )

    st.write(
        f"z-critical = **{details['z_critical']:.4f}**; "
        f"δ = **{details['delta']:.4f}**; "
        f"k = **{details['k']}**; "
        f"lower bound = **{details['lower_bound']:.4f}**"
    )


# ---------------------------------------------------------
# ANALYSIS RESULTS TABLE
# ---------------------------------------------------------
st.dataframe(
    result_df,
    width="stretch",
    hide_index=True
)


# ---------------------------------------------------------
# 3. DETECTED EXCEPTIONAL STAFF / OUTLIERS
# ---------------------------------------------------------
st.subheader("3. Detected Exceptional Staff / Outliers")

if len(outliers) == 0:

    st.info(
        "No observations were identified as outliers "
        "by the selected method."
    )

else:

    st.success(
        f"{len(outliers)} staff member(s) were identified as outliers."
    )

    st.dataframe(
        outliers,
        width="stretch",
        hide_index=True
    )


# ---------------------------------------------------------
# STAFF SCORE CHART
# ---------------------------------------------------------
chart_df = result_df.copy()

chart_df["Status"] = chart_df["Outlier"].map(
    {
        True: "Outlier",
        False: "Not outlier"
    }
)


fig2 = px.scatter(
    chart_df,
    x="StaffID",
    y="Score",
    color="Status",
    hover_data=["Score"],
    title="Staff Appraisal Scores and Detected Outliers"
)

st.plotly_chart(
    fig2,
    width="stretch"
)


# ---------------------------------------------------------
# 4. DOWNLOAD RESULTS
# ---------------------------------------------------------
st.subheader("4. Download Results")

buf = io.StringIO()

result_df.to_csv(
    buf,
    index=False
)

st.download_button(
    "Download analyzed results (CSV)",
    buf.getvalue(),
    file_name="employee_performance_analysis.csv",
    mime="text/csv",
    width="stretch"
)
import io

import pandas as pd
import plotly.express as px
import streamlit as st

from algorithms import analyze_dataset


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Employee Performance Reward System",
    page_icon="🏆",
    layout="wide",
)

# ---------------------------------------------------------
# COMPACT UI STYLING
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        h1 {
            font-size: 2rem !important;
        }

        h2, h3 {
            font-size: 1.25rem !important;
        }

        div[data-testid="stMetric"] {
            padding: 0.35rem 0.5rem;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1rem !important;
        }

        div[data-testid="stMetricDelta"] {
            font-size: 0.7rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("🏆 Employee Performance Reward System")


# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------
with st.sidebar:
    st.header("Analysis Settings")

    alpha = st.selectbox(
        "Significance level (α)",
        [0.05, 0.01],
        index=0,
        format_func=lambda x: f"{x} ({int((1 - x) * 100)}% confidence)",
    )

    k = st.selectbox(
        "IQR multiplier (k)",
        [1.5, 3.0],
        index=0,
    )

    st.divider()
    st.markdown("**Automatic method selection**")
    st.write("Normal → confidence-adjusted Z-score")
    st.write("Non-normal/skewed → confidence-adjusted IQR")


# ---------------------------------------------------------
# CSV UPLOAD
# ---------------------------------------------------------
uploaded = st.file_uploader(
    "Upload staff appraisal CSV",
    type=["csv"],
)


if uploaded is None:
    st.info("Upload a CSV containing the columns **StaffID** and **Score**.")
    st.markdown("### Expected format")

    sample_data = pd.DataFrame(
        {
            "StaffID": ["001", "002", "003", "004"],
            "Score": [67, 88, 99, 87],
        }
    )

    st.dataframe(
        sample_data,
        width="stretch",
        hide_index=True,
        column_config={
            "StaffID": st.column_config.TextColumn("StaffID", width="small"),
            "Score": st.column_config.NumberColumn("Score", width="small"),
        },
    )
    st.stop()


# ---------------------------------------------------------
# READ CSV
# ---------------------------------------------------------
try:
    raw = pd.read_csv(uploaded, dtype={"StaffID": str})
except Exception as exc:
    st.error(f"Unable to read the CSV file: {exc}")
    st.stop()


# ---------------------------------------------------------
# ANALYZE DATASET
# ---------------------------------------------------------
try:
    (
        result_df,
        dist,
        details,
        outliers,
        positive_outliers,
        negative_outliers,
    ) = analyze_dataset(raw, alpha=alpha, k=k)
except Exception as exc:
    st.error(str(exc))
    st.stop()


# ---------------------------------------------------------
# UPLOADED DATASET
# ---------------------------------------------------------
st.subheader("Uploaded Dataset")

st.dataframe(
    result_df[["StaffID", "Score"]],
    width="stretch",
    hide_index=True,
    column_config={
        "StaffID": st.column_config.TextColumn("StaffID", width="small"),
        "Score": st.column_config.NumberColumn("Score", width="small", format="%.2f"),
    },
)


# ---------------------------------------------------------
# 1. DISTRIBUTION ANALYSIS
# ---------------------------------------------------------
st.subheader("1. Distribution Analysis")

c1, c2, c3, c4 = st.columns(4, gap="small")

c1.metric("Skewness", f"{dist.skewness:.4f}")
c2.metric("Shapiro-Wilk p-value", f"{dist.shapiro_p_value:.4f}")
c3.metric("Decision", dist.decision)
c4.metric("Selected Method", dist.method)

if dist.is_normal:
    st.success(
        "Normal dataset detected. The confidence-adjusted Z-score method was selected."
    )
else:
    st.warning(
        "Non-normal/skewed dataset detected. The confidence-adjusted IQR method was selected."
    )


# Grey/black distribution graph
fig = px.histogram(
    result_df,
    x="Score",
    nbins=max(5, min(30, max(1, len(result_df) // 2))),
    title="Distribution of Appraisal Scores",
    color_discrete_sequence=["#555555"],
)
fig.update_layout(
    template="plotly_white",
    font=dict(size=11),
    title_font=dict(size=14),
)
fig.update_traces(marker_line_color="#222222", marker_line_width=1)

st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------
# 2. OUTLIER DETECTION
# ---------------------------------------------------------
st.subheader("2. Outlier Detection")

if dist.is_normal:
    a, b, c = st.columns(3, gap="small")

    a.metric("Mean", f"{details['mean']:.4f}")
    b.metric("Standard deviation", f"{details['std']:.4f}")
    c.metric(
        "Confidence-adjusted Z threshold",
        f"±{details['threshold']:.4f}",
    )

    st.caption(
        f"t-critical = {details['t_critical']:.4f}  |  n = {details['n']}"
    )

else:
    a, b, c, d = st.columns(4, gap="small")

    a.metric("Q1", f"{details['q1']:.4f}")
    b.metric("Q3", f"{details['q3']:.4f}")
    c.metric("IQR", f"{details['iqr']:.4f}")
    d.metric("Upper bound", f"{details['upper_bound']:.4f}")

    st.caption(
        f"z-critical = {details['z_critical']:.4f}  |  "
        f"δ = {details['delta']:.4f}  |  "
        f"k = {details['k']}  |  "
        f"lower bound = {details['lower_bound']:.4f}"
    )


# Full analysis result
# Display only the useful result columns; the internal outlier flags are
# retained for the calculations but are not shown to the user.
display_columns = ["StaffID", "Score", "Z_score", "Reward_Status"]
st.dataframe(
    result_df[display_columns],
    width="stretch",
    hide_index=True,
    column_config={
        "StaffID": st.column_config.TextColumn("StaffID", width="small"),
        "Score": st.column_config.NumberColumn("Score", width="small", format="%.2f"),
        "Z_score": st.column_config.NumberColumn("Z-score", width="small", format="%.4f"),
        "Reward_Status": st.column_config.TextColumn("Reward Status", width="medium"),
    },
)


# ---------------------------------------------------------
# 3. DETECTED EXCEPTIONAL STAFF / OUTLIERS
# ---------------------------------------------------------
st.subheader("3. Detected Exceptional Staff / Outliers")

if len(positive_outliers) > 0:
    st.markdown("#### Positive Outliers")
    st.success(
        f"{len(positive_outliers)} staff member(s) identified as exceptionally high performers."
    )
    st.dataframe(
        positive_outliers[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "StaffID": st.column_config.TextColumn("StaffID", width="small"),
            "Score": st.column_config.NumberColumn("Score", width="small", format="%.2f"),
            "Z_score": st.column_config.NumberColumn("Z-score", width="small", format="%.4f"),
            "Reward_Status": st.column_config.TextColumn("Reward Status", width="medium"),
        },
    )
else:
    st.info("No positive outliers were identified for reward.")


if len(negative_outliers) > 0:
    st.markdown("#### Negative Outliers")
    st.warning(
        f"{len(negative_outliers)} staff member(s) identified as very poor performers. "
        "They are retained in the results but are not eligible for reward."
    )
    st.dataframe(
        negative_outliers[display_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "StaffID": st.column_config.TextColumn("StaffID", width="small"),
            "Score": st.column_config.NumberColumn("Score", width="small", format="%.2f"),
            "Z_score": st.column_config.NumberColumn("Z-score", width="small", format="%.4f"),
            "Reward_Status": st.column_config.TextColumn("Reward Status", width="medium"),
        },
    )
else:
    st.info("No negative outliers were identified.")


if len(outliers) == 0:
    st.info("No observations were identified as outliers by the selected method.")


# ---------------------------------------------------------
# STAFF SCORE CHART
# ---------------------------------------------------------
chart_df = result_df.copy()
chart_df["Status"] = "Not an Outlier"
chart_df.loc[chart_df["Positive_Outlier"], "Status"] = "Positive Outlier"
chart_df.loc[chart_df["Negative_Outlier"], "Status"] = "Negative Outlier"

fig2 = px.scatter(
    chart_df,
    x="StaffID",
    y="Score",
    color="Status",
    hover_data=["Score", "Reward_Status"],
    title="Staff Appraisal Scores and Detected Outliers",
    color_discrete_map={
        "Not an Outlier": "#888888",
        "Positive Outlier": "#111111",
        "Negative Outlier": "#444444",
    },
)
fig2.update_layout(
    template="plotly_white",
    font=dict(size=11),
    title_font=dict(size=14),
)

st.plotly_chart(fig2, width="stretch")


# ---------------------------------------------------------
# 4. DOWNLOAD RESULTS
# ---------------------------------------------------------
st.subheader("4. Download Results")

buf = io.StringIO()
result_df.to_csv(buf, index=False)

st.download_button(
    "Download analyzed results (CSV)",
    buf.getvalue(),
    file_name="employee_performance_analysis.csv",
    mime="text/csv",
    width="stretch",
)

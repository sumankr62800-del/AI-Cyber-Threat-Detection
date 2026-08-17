from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Cyber Threat Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"

PREPROCESSOR_PATH = (
    PROJECT_ROOT / "models" / "preprocessor.pkl"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "threat_analysis_results.csv"
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found:\n{MODEL_PATH}"
    )
    st.stop()


if not PREPROCESSOR_PATH.exists():
    st.error(
        f"Preprocessor file not found:\n{PREPROCESSOR_PATH}"
    )
    st.stop()


if not RESULTS_PATH.exists():
    st.error(
        f"Threat analysis results not found:\n{RESULTS_PATH}"
    )
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    return model, preprocessor


# ============================================================
# LOAD THREAT RESULTS
# ============================================================

@st.cache_data
def load_results():

    return pd.read_csv(
        RESULTS_PATH
    )


# ============================================================
# LOAD DATA
# ============================================================

try:

    model, preprocessor = load_model()

    results = load_results()

except Exception as e:

    st.error(
        f"Error loading project data: {e}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ AI Cyber Threat Detection System"
)

st.markdown(
    """
    ### Machine Learning Based Network Intrusion Detection

    This system analyzes network traffic using a
    **Random Forest machine learning model** trained
    on the **UNSW-NB15 dataset**.
    """
)

st.divider()


# ============================================================
# BASIC STATISTICS
# ============================================================

total_records = len(results)

attack_count = int(
    (results["predicted_label"] == 1).sum()
)

normal_count = int(
    (results["predicted_label"] == 0).sum()
)

attack_rate = (
    attack_count / total_records * 100
)


critical_count = int(
    (results["risk_level"] == "CRITICAL").sum()
)

high_count = int(
    (results["risk_level"] == "HIGH").sum()
)

medium_count = int(
    (results["risk_level"] == "MEDIUM").sum()
)

low_count = int(
    (results["risk_level"] == "LOW").sum()
)


# ============================================================
# DASHBOARD METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Traffic",
        f"{total_records:,}"
    )


with col2:

    st.metric(
        "Detected Attacks",
        f"{attack_count:,}"
    )


with col3:

    st.metric(
        "Normal Traffic",
        f"{normal_count:,}"
    )


with col4:

    st.metric(
        "Threat Rate",
        f"{attack_rate:.2f}%"
    )


st.divider()


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader(
    "🚨 Risk Level Distribution"
)


risk_data = pd.DataFrame(
    {
        "Risk Level": [
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW"
        ],
        "Count": [
            critical_count,
            high_count,
            medium_count,
            low_count
        ]
    }
)


col1, col2 = st.columns(2)


with col1:

    st.bar_chart(
        risk_data.set_index(
            "Risk Level"
        )
    )


with col2:

    st.dataframe(
        risk_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ATTACK PROBABILITY
# ============================================================

st.subheader(
    "📊 Attack Probability"
)


probability_data = results[
    ["attack_probability"]
].head(500).copy()


probability_data[
    "attack_probability"
] = (
    probability_data[
        "attack_probability"
    ] * 100
)


probability_data = probability_data.rename(
    columns={
        "attack_probability":
        "Attack Probability (%)"
    }
)


st.line_chart(
    probability_data
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader(
    "🎯 Detection Performance"
)


actual = results["label"]

predicted = results[
    "predicted_label"
]


tn, fp, fn, tp = confusion_matrix(
    actual,
    predicted
).ravel()


performance = pd.DataFrame(
    {
        "Metric": [
            "True Normal",
            "False Alarm",
            "Missed Attack",
            "True Attack"
        ],
        "Count": [
            tn,
            fp,
            fn,
            tp
        ]
    }
)


st.dataframe(
    performance,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# THREAT LOGS
# ============================================================

st.subheader(
    "🔍 Network Threat Logs"
)


risk_filter = st.multiselect(
    "Filter by Risk Level",
    [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW"
    ],
    default=[
        "CRITICAL",
        "HIGH"
    ]
)


filtered_results = results[
    results["risk_level"].isin(
        risk_filter
    )
]


display_columns = [
    "proto",
    "service",
    "state",
    "attack_cat",
    "label",
    "predicted_label",
    "attack_probability",
    "risk_level"
]


available_columns = [
    column
    for column in display_columns
    if column in filtered_results.columns
]


display_df = filtered_results[
    available_columns
].head(100).copy()


if "attack_probability" in display_df.columns:

    display_df[
        "attack_probability"
    ] = (
        display_df[
            "attack_probability"
        ] * 100
    ).round(2)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# INSPECT INDIVIDUAL TRAFFIC
# ============================================================

st.divider()

st.subheader(
    "🔎 Inspect Network Traffic"
)


record_number = st.number_input(
    "Select network record",
    min_value=0,
    max_value=len(results) - 1,
    value=0,
    step=1
)


selected_record = results.iloc[
    int(record_number)
]


prediction = int(
    selected_record[
        "predicted_label"
    ]
)


probability = float(
    selected_record[
        "attack_probability"
    ]
)


risk = selected_record[
    "risk_level"
]


if prediction == 1:

    st.error(
        f"🚨 ATTACK DETECTED\n\n"
        f"Attack Probability: "
        f"{probability * 100:.2f}%"
    )

else:

    st.success(
        f"✅ NORMAL TRAFFIC\n\n"
        f"Attack Probability: "
        f"{probability * 100:.2f}%"
    )


st.write(
    f"**Risk Level:** {risk}"
)


# ============================================================
# NETWORK DETAILS
# ============================================================

st.write(
    "**Network Traffic Details**"
)


detail_columns = [
    "proto",
    "service",
    "state",
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "attack_cat"
]


available_details = [
    column
    for column in detail_columns
    if column in selected_record.index
]


details = selected_record[
    available_details
]


st.dataframe(
    details.to_frame(
        name="Value"
    ),
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Cyber Threat Detection | "
    "UNSW-NB15 | "
    "Random Forest"
)
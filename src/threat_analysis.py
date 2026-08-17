from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "dataset"


MODEL_PATH = MODEL_DIR / "random_forest.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

TEST_DATA_PATH = DATA_DIR / "UNSW_NB15_testing-set.csv"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("AI CYBER THREAT DETECTION - THREAT ANALYSIS")
print("=" * 65)

print("\nLoading Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

print("\nLoading preprocessing pipeline...")

preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Preprocessor loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading UNSW-NB15 testing dataset...")

df = pd.read_csv(TEST_DATA_PATH)

print(
    f"Testing records loaded: {len(df)}"
)


# ============================================================
# PREPARE FEATURES
# ============================================================

columns_to_remove = [
    "id",
    "attack_cat",
    "label"
]

X = df.drop(
    columns=columns_to_remove,
    errors="ignore"
)

y = df["label"]


# ============================================================
# PREPROCESS
# ============================================================

print("\nPreprocessing network traffic...")

X_processed_array = preprocessor.transform(X)

X_processed = pd.DataFrame(
    X_processed_array,
    columns=preprocessor.get_feature_names_out()
)

print(
    f"Processed features: {X_processed.shape[1]}"
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nRunning threat detection...")

predictions = model.predict(X_processed)

probabilities = model.predict_proba(
    X_processed
)[:, 1]


# ============================================================
# MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions,
    zero_division=0
)

recall = recall_score(
    y,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y,
    probabilities
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y,
    predictions
).ravel()


# ============================================================
# FALSE POSITIVE RATE
# ============================================================

false_positive_rate = fp / (fp + tn)

false_negative_rate = fn / (fn + tp)


# ============================================================
# DISPLAY PERFORMANCE
# ============================================================

print("\n" + "=" * 65)
print("THREAT DETECTION PERFORMANCE")
print("=" * 65)

print(
    f"\nAccuracy           : {accuracy * 100:.2f}%"
)

print(
    f"Precision          : {precision * 100:.2f}%"
)

print(
    f"Attack Recall      : {recall * 100:.2f}%"
)

print(
    f"F1 Score           : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC            : {roc_auc * 100:.2f}%"
)

print(
    f"False Positive Rate: {false_positive_rate * 100:.2f}%"
)

print(
    f"Missed Attack Rate : {false_negative_rate * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

print()

print(f"True Normal       : {tn}")
print(f"False Alarm       : {fp}")
print(f"Missed Attack     : {fn}")
print(f"True Attack       : {tp}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)

print(
    classification_report(
        y,
        predictions,
        target_names=[
            "Normal",
            "Attack"
        ],
        zero_division=0
    )
)


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def get_risk_level(probability):

    if probability >= 0.90:
        return "CRITICAL"

    elif probability >= 0.70:
        return "HIGH"

    elif probability >= 0.50:
        return "MEDIUM"

    else:
        return "LOW"


risk_levels = [
    get_risk_level(probability)
    for probability in probabilities
]


# ============================================================
# THREAT SUMMARY
# ============================================================

risk_counts = pd.Series(
    risk_levels
).value_counts()


print("\n" + "=" * 65)
print("RISK LEVEL DISTRIBUTION")
print("=" * 65)

for level in [
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW"
]:

    count = risk_counts.get(
        level,
        0
    )

    print(
        f"{level:<10}: {count}"
    )


# ============================================================
# ATTACK / NORMAL SUMMARY
# ============================================================

normal_predictions = (
    predictions == 0
).sum()

attack_predictions = (
    predictions == 1
).sum()


print("\n" + "=" * 65)
print("TRAFFIC SUMMARY")
print("=" * 65)

print(
    f"\nTotal Traffic Records : {len(df)}"
)

print(
    f"Predicted Normal      : {normal_predictions}"
)

print(
    f"Predicted Attacks     : {attack_predictions}"
)


# ============================================================
# SAVE ANALYSIS RESULTS
# ============================================================

results = df.copy()

results["predicted_label"] = predictions

results["attack_probability"] = probabilities

results["risk_level"] = risk_levels


results_path = (
    PROJECT_ROOT
    / "processed_data"
    / "threat_analysis_results.csv"
)


results.to_csv(
    results_path,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 65)
print("THREAT ANALYSIS COMPLETED")
print("=" * 65)

print(
    "\nResults saved to:"
)

print(results_path)

print(
    "\nAI Cyber Threat Detection analysis completed successfully."
)
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "dataset"


MODEL_PATH = MODEL_DIR / "random_forest.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

TEST_DATA_PATH = DATASET_DIR / "UNSW_NB15_testing-set.csv"


# ============================================================
# LOAD MODEL AND PREPROCESSOR
# ============================================================

print("=" * 60)
print("AI CYBER THREAT DETECTION")
print("=" * 60)

print("\nLoading model...")

model = joblib.load(MODEL_PATH)

print("Random Forest loaded successfully.")

print("\nLoading preprocessor...")

preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Preprocessor loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading network traffic data...")

df = pd.read_csv(TEST_DATA_PATH)

print("Dataset loaded successfully.")
print(f"Total records: {len(df)}")


# ============================================================
# SELECT ONE NETWORK TRAFFIC RECORD
# ============================================================

# Change this number to test another network record.
ROW_NUMBER = 0

sample = df.iloc[[ROW_NUMBER]].copy()


# ============================================================
# SAVE ORIGINAL INFORMATION
# ============================================================

original_label = int(sample["label"].iloc[0])

if "attack_cat" in sample.columns:
    actual_attack = sample["attack_cat"].iloc[0]
else:
    actual_attack = "Unknown"


# ============================================================
# PREPARE INPUT
# ============================================================

# These columns were removed during preprocessing.

columns_to_remove = [
    "id",
    "attack_cat",
    "label"
]

X_sample = sample.drop(
    columns=columns_to_remove,
    errors="ignore"
)


# ============================================================
# PREPROCESS INPUT
# ============================================================

X_processed = preprocessor.transform(X_sample)


# ============================================================
# PREDICTION
# ============================================================
# ============================================================
# PREPROCESS INPUT
# ============================================================

X_processed_array = preprocessor.transform(X_sample)

X_processed = pd.DataFrame(
    X_processed_array,
    columns=preprocessor.get_feature_names_out()
)
prediction = model.predict(
    X_processed
)[0]

probabilities = model.predict_proba(
    X_processed
)[0]


attack_probability = probabilities[1]
normal_probability = probabilities[0]


# ============================================================
# DETERMINE RESULT
# ============================================================

if prediction == 1:

    prediction_text = "ATTACK"

    if attack_probability >= 0.90:
        risk_level = "CRITICAL"

    elif attack_probability >= 0.70:
        risk_level = "HIGH"

    elif attack_probability >= 0.50:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

else:

    prediction_text = "NORMAL"

    risk_level = "LOW"


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("THREAT DETECTION RESULT")
print("=" * 60)

print(f"\nRecord Number      : {ROW_NUMBER}")

print(f"Prediction         : {prediction_text}")

print(
    f"Attack Probability : {attack_probability * 100:.2f}%"
)

print(
    f"Normal Probability : {normal_probability * 100:.2f}%"
)

print(
    f"Risk Level         : {risk_level}"
)

print(
    f"Actual Dataset     : {actual_attack}"
)

print(
    f"Actual Label       : "
    f"{'ATTACK' if original_label == 1 else 'NORMAL'}"
)


# ============================================================
# SECURITY MESSAGE
# ============================================================

print("\n" + "-" * 60)

if prediction == 1:

    print("⚠ WARNING: Suspicious network activity detected.")

    print(
        "Recommended Action: Investigate the network traffic."
    )

else:

    print("✓ Network traffic appears normal.")

    print(
        "Recommended Action: No immediate threat detected."
    )


print("-" * 60)

print("\nPrediction completed successfully.")
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "dataset" / "UNSW_NB15_training-set.csv"
TEST_PATH = PROJECT_ROOT / "dataset" / "UNSW_NB15_testing-set.csv"

OUTPUT_DIR = PROJECT_ROOT / "processed_data"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 60)
print("AI CYBER THREAT DETECTION - PREPROCESSING")
print("=" * 60)

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training dataset loaded successfully.")
print("Testing dataset loaded successfully.")

print("\nTraining shape:", train_df.shape)
print("Testing shape :", test_df.shape)


# ============================================================
# REMOVE ID AND TARGET-LEAKAGE COLUMN
# ============================================================

print("\nRemoving unnecessary columns...")

# id = dataset record identifier
# attack_cat = removed to prevent target leakage

columns_to_remove = ["id", "attack_cat"]

X_train = train_df.drop(
    columns=columns_to_remove + ["label"]
)

y_train = train_df["label"]

X_test = test_df.drop(
    columns=columns_to_remove + ["label"]
)

y_test = test_df["label"]


# ============================================================
# IDENTIFY FEATURES
# ============================================================

categorical_features = [
    "proto",
    "service",
    "state"
]

numerical_features = [
    column
    for column in X_train.columns
    if column not in categorical_features
]

print("\nCategorical features:")
print(categorical_features)

print("\nNumber of numerical features:")
print(len(numerical_features))


# ============================================================
# CREATE PREPROCESSING PIPELINE
# ============================================================

print("\nCreating preprocessing pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# FIT ON TRAINING DATA
# ============================================================

print("\nFitting preprocessing pipeline...")

X_train_processed = preprocessor.fit_transform(X_train)


# ============================================================
# TRANSFORM TEST DATA
# ============================================================

print("Transforming testing data...")

X_test_processed = preprocessor.transform(X_test)


# ============================================================
# GET FEATURE NAMES
# ============================================================

feature_names = preprocessor.get_feature_names_out()


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

print("\nSaving processed datasets...")

X_train_processed.to_csv(
    OUTPUT_DIR / "X_train.csv",
    index=False
)

X_test_processed.to_csv(
    OUTPUT_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    OUTPUT_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    OUTPUT_DIR / "y_test.csv",
    index=False
)


# ============================================================
# SAVE PREPROCESSOR
# ============================================================

preprocessor_path = MODEL_DIR / "preprocessor.pkl"

joblib.dump(
    preprocessor,
    preprocessor_path
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print("\nOriginal training features:", X_train.shape[1])
print("Processed training features:", X_train_processed.shape[1])

print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nProcessed testing shape:")
print(X_test_processed.shape)

print("\nPreprocessor saved to:")
print(preprocessor_path)

print("\nProcessed datasets saved to:")
print(OUTPUT_DIR)

print("\nTarget distribution:")
print(y_train.value_counts())

print("\nDone.")
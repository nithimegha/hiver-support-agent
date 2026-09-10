import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# =========================================================
# PATHS
# =========================================================

TRAIN_PATH = "data/processed/train.csv"
GOLDEN_PATH = "data/golden/golden_review.csv"


# =========================================================
# START
# =========================================================

print("=" * 70)
print("FINAL EVALUATION SUMMARY")
print("=" * 70)


# =========================================================
# LOAD TRAINING DATA
# =========================================================

print("\nLoading training data...")

train_df = pd.read_csv(TRAIN_PATH)

print(f"Training examples: {len(train_df)}")

print("Training columns:")
print(train_df.columns.tolist())


# =========================================================
# LOAD GOLDEN DATA
# =========================================================

print("\nLoading golden evaluation set...")

golden_df = pd.read_csv(GOLDEN_PATH)

# Keep only manually labelled examples
golden_df = golden_df.dropna(
    subset=["golden_intent"]
).copy()

print(
    f"Manually labelled examples: "
    f"{len(golden_df)}"
)


if len(golden_df) == 0:

    print(
        "No manually labelled examples found."
    )

    raise SystemExit


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_train_columns = [
    "customer_text",
    "intent"
]

required_golden_columns = [
    "customer_text",
    "golden_intent"
]


for column in required_train_columns:

    if column not in train_df.columns:

        raise ValueError(
            f"Missing column in train.csv: {column}"
        )


for column in required_golden_columns:

    if column not in golden_df.columns:

        raise ValueError(
            f"Missing column in golden_review.csv: {column}"
        )


# =========================================================
# PREPARE TRAINING DATA
# =========================================================

train_df = train_df.dropna(
    subset=[
        "customer_text",
        "intent"
    ]
)

X_train = (
    train_df["customer_text"]
    .astype(str)
)

y_train = (
    train_df["intent"]
    .astype(str)
)


# =========================================================
# PREPARE GOLDEN DATA
# =========================================================

X_golden = (
    golden_df["customer_text"]
    .astype(str)
)

y_true = (
    golden_df["golden_intent"]
    .astype(str)
)


# =========================================================
# TF-IDF VECTORIZATION
# =========================================================

print("\n" + "-" * 70)
print("TRAINING TF-IDF + LOGISTIC REGRESSION")
print("-" * 70)

print("\nCreating TF-IDF features...")


vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)


print(
    f"TF-IDF training matrix: "
    f"{X_train_tfidf.shape}"
)


# =========================================================
# TRANSFORM GOLDEN SET
# =========================================================

X_golden_tfidf = vectorizer.transform(
    X_golden
)


print(
    f"Golden evaluation matrix: "
    f"{X_golden_tfidf.shape}"
)


# =========================================================
# TRAIN LOGISTIC REGRESSION
# =========================================================

print("\nTraining Logistic Regression...")


model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)


model.fit(
    X_train_tfidf,
    y_train
)


print("Model training complete.")


# =========================================================
# PREDICT GOLDEN SET
# =========================================================

print("\nGenerating predictions...")


y_pred = model.predict(
    X_golden_tfidf
)


# =========================================================
# CALCULATE METRICS
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 70)
print("TF-IDF + LOGISTIC REGRESSION RESULTS")
print("=" * 70)

print(
    f"\nAccuracy:    {accuracy:.4f}"
)

print(
    f"Macro F1:    {macro_f1:.4f}"
)

print(
    f"Weighted F1: {weighted_f1:.4f}"
)


# =========================================================
# HUMAN / MODEL AGREEMENT
# =========================================================

agreement = (
    y_true.to_numpy()
    == y_pred
).mean()


print("\nHuman / model agreement:")

print(
    f"{agreement:.2%}"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n" + "-" * 70)
print("CLASSIFICATION REPORT")
print("-" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)


# =========================================================
# SAVE PREDICTIONS
# =========================================================

golden_results = golden_df.copy()

golden_results[
    "final_model_prediction"
] = y_pred


golden_results[
    "final_model_correct"
] = (
    golden_results["golden_intent"].astype(str)
    ==
    golden_results["final_model_prediction"].astype(str)
)


OUTPUT_PATH = (
    "data/golden/final_evaluation_results.csv"
)


golden_results.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nDetailed results saved to:"
)

print(OUTPUT_PATH)


# =========================================================
# SHOW MISTAKES
# =========================================================

mistakes = golden_results[
    golden_results["final_model_correct"] == False
]


print("\n" + "-" * 70)
print("MISTAKES")
print("-" * 70)

print(
    f"Incorrect examples: "
    f"{len(mistakes)}"
)


if len(mistakes) > 0:

    for _, row in mistakes.head(5).iterrows():

        print("\nCustomer:")

        print(
            row["customer_text"]
        )

        print(
            f"Model: "
            f"{row['final_model_prediction']}"
        )

        print(
            f"Human: "
            f"{row['golden_intent']}"
        )


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
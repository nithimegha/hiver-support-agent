import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"


print("Loading training data...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


# Find the most common intent in the training data.
majority_intent = train_df["intent"].value_counts().idxmax()

print(f"\nMajority intent: {majority_intent}")


# Predict the same intent for every test example.
predictions = [
    majority_intent
] * len(test_df)


true_labels = test_df["intent"]


# Calculate metrics.
accuracy = accuracy_score(
    true_labels,
    predictions
)

macro_f1 = f1_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    true_labels,
    predictions,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("MAJORITY CLASS BASELINE")
print("=" * 60)

print(f"Accuracy:    {accuracy:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")


print("\nClassification report:")
print(
    classification_report(
        true_labels,
        predictions,
        zero_division=0
    )
)
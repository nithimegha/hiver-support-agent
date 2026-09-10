import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

INPUT_PATH = "data/golden/golden_review.csv"

df = pd.read_csv(INPUT_PATH)

# Keep only manually labelled examples
df = df.dropna(subset=["golden_intent"])

y_true = df["golden_intent"]
y_pred = df["model_prediction"]

print("=" * 70)
print("GOLDEN SET EVALUATION")
print("=" * 70)

print(f"Labelled examples: {len(df)}")

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro")
weighted_f1 = f1_score(y_true, y_pred, average="weighted")

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Macro F1:  {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))
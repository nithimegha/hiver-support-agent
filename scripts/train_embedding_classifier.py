import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm


# ---------------------------------------------------------
# 1. Load training and test data
# ---------------------------------------------------------

print("Loading training and test data...")

train_path = "data/processed/train.csv"
test_path = "data/processed/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Training examples: {len(train_df):,}")
print(f"Test examples: {len(test_df):,}")


# ---------------------------------------------------------
# 2. Get text and labels
# ---------------------------------------------------------

X_train = train_df["customer_text"].fillna("").astype(str)
y_train = train_df["intent"]

X_test = test_df["customer_text"].fillna("").astype(str)
y_test = test_df["intent"]

print("\nIntent classes:")
print(sorted(y_train.unique()))


# ---------------------------------------------------------
# 3. Load sentence embedding model
# ---------------------------------------------------------

print("\nLoading sentence embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# ---------------------------------------------------------
# 4. Convert customer messages into embeddings
# ---------------------------------------------------------

print("\nCreating training embeddings...")

X_train_embeddings = model.encode(
    X_train.tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nCreating test embeddings...")

X_test_embeddings = model.encode(
    X_test.tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)


# ---------------------------------------------------------
# 5. Train Logistic Regression
# ---------------------------------------------------------

print("\nTraining embedding + Logistic Regression model...")

classifier = LogisticRegression(
    max_iter=1000,
    random_state=42
)

classifier.fit(X_train_embeddings, y_train)

print("Training complete!")


# ---------------------------------------------------------
# 6. Make predictions
# ---------------------------------------------------------

print("\nMaking predictions...")

y_pred = classifier.predict(X_test_embeddings)


# ---------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\n" + "=" * 65)
print("SENTENCE EMBEDDINGS + LOGISTIC REGRESSION")
print("=" * 65)

print(f"Accuracy:   {accuracy:.4f}")
print(f"Macro F1:   {macro_f1:.4f}")
print(f"Weighted F1:{weighted_f1:.4f}")


# ---------------------------------------------------------
# 8. Detailed classification report
# ---------------------------------------------------------

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=2
    )
)


# ---------------------------------------------------------
# 9. Save predictions
# ---------------------------------------------------------

results = test_df.copy()

results["predicted_intent"] = y_pred

results["correct"] = (
    results["intent"] == results["predicted_intent"]
)

output_path = "data/processed/embedding_predictions.csv"

results.to_csv(
    output_path,
    index=False
)

print(f"\nPredictions saved to:")
print(output_path)
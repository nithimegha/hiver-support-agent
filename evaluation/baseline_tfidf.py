import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"


print("Loading training and test data...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


X_train = train_df["customer_text"].fillna("")
y_train = train_df["intent"]

X_test = test_df["customer_text"].fillna("")
y_test = test_df["intent"]


print(f"Training examples: {len(train_df):,}")
print(f"Test examples:     {len(test_df):,}")


print("\nBuilding TF-IDF + Logistic Regression model...")


model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=100000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


print("Training model...")

model.fit(X_train, y_train)


print("Training complete!")


print("\nMaking predictions...")

predictions = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("TF-IDF + LOGISTIC REGRESSION BASELINE")
print("=" * 60)

print(f"Accuracy:    {accuracy:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")


print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)
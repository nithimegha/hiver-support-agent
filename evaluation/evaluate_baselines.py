import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"


def find_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


print("=" * 70)
print("BASELINE EVALUATION")
print("=" * 70)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nTrain columns:")
print(train_df.columns.tolist())

print("\nTest columns:")
print(test_df.columns.tolist())


text_candidates = [
    "customer_text",
    "text",
    "message",
    "tweet_text"
]

label_candidates = [
    "intent",
    "label",
    "golden_intent",
    "category"
]

text_column = find_column(train_df, text_candidates)
label_column = find_column(train_df, label_candidates)

if text_column is None:
    raise ValueError("Could not find text column.")

if label_column is None:
    raise ValueError("Could not find label column.")


print(f"\nText column: {text_column}")
print(f"Label column: {label_column}")


# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------

train_df = train_df.dropna(
    subset=[text_column, label_column]
)

test_df = test_df.dropna(
    subset=[text_column, label_column]
)

X_train = train_df[text_column].astype(str)
y_train = train_df[label_column].astype(str)

X_test = test_df[text_column].astype(str)
y_test = test_df[label_column].astype(str)


print(f"\nTraining examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")


# ---------------------------------------------------------
# BASELINE 1: MAJORITY CLASS
# ---------------------------------------------------------

majority_class = y_train.value_counts().idxmax()

majority_predictions = [
    majority_class
] * len(y_test)

majority_accuracy = accuracy_score(
    y_test,
    majority_predictions
)

majority_f1 = f1_score(
    y_test,
    majority_predictions,
    average="macro"
)


print("\n" + "-" * 70)
print("BASELINE 1: MAJORITY CLASS")
print("-" * 70)

print(f"Majority class: {majority_class}")
print(f"Accuracy: {majority_accuracy:.4f}")
print(f"Macro F1: {majority_f1:.4f}")


# ---------------------------------------------------------
# BASELINE 2: TF-IDF + LOGISTIC REGRESSION
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("BASELINE 2: TF-IDF + LOGISTIC REGRESSION")
print("-" * 70)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

classifier.fit(
    X_train_tfidf,
    y_train
)

tfidf_predictions = classifier.predict(
    X_test_tfidf
)


tfidf_accuracy = accuracy_score(
    y_test,
    tfidf_predictions
)

tfidf_f1 = f1_score(
    y_test,
    tfidf_predictions,
    average="macro"
)


print(f"Accuracy: {tfidf_accuracy:.4f}")
print(f"Macro F1: {tfidf_f1:.4f}")


# ---------------------------------------------------------
# FINAL COMPARISON
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

print(
    f"Majority Class     | "
    f"Accuracy: {majority_accuracy:.4f} | "
    f"Macro F1: {majority_f1:.4f}"
)

print(
    f"TF-IDF + Logistic  | "
    f"Accuracy: {tfidf_accuracy:.4f} | "
    f"Macro F1: {tfidf_f1:.4f}"
)

print("\nBaseline evaluation complete.")
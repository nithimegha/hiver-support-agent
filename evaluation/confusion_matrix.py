import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"


print("Loading data...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


X_train = train_df["customer_text"].fillna("")
y_train = train_df["intent"]

X_test = test_df["customer_text"].fillna("")
y_test = test_df["intent"]


print("Training TF-IDF + Logistic Regression...")


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


model.fit(X_train, y_train)


print("Making predictions...")

predictions = model.predict(X_test)


labels = sorted(y_test.unique())


cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)


print("\nConfusion matrix:")
print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)


display.plot(
    xticks_rotation=45
)

plt.title("AmazonHelp Intent Classification")
plt.tight_layout()

plt.savefig(
    "data/processed/confusion_matrix.png",
    dpi=150
)

print("\nSaved confusion matrix to:")
print("data/processed/confusion_matrix.png")

plt.show()
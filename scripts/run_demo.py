import os
import pandas as pd
import joblib
import faiss

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =========================================================
# PATHS
# =========================================================

TRAIN_PATH = "data/processed/train.csv"
RETRIEVAL_CASES_PATH = "data/processed/amazonhelp_retrieval_cases.csv"
FAISS_INDEX_PATH = "data/processed/amazonhelp_faiss.index"

TFIDF_MODEL_PATH = "data/processed/tfidf_classifier.joblib"
TFIDF_VECTORIZER_PATH = "data/processed/tfidf_vectorizer.joblib"


# =========================================================
# START
# =========================================================

print("=" * 70)
print("AMAZONHELP AI SUPPORT AGENT")
print("=" * 70)


# =========================================================
# LOAD TF-IDF CLASSIFIER
# =========================================================

print("\nLoading TF-IDF classifier...")

if os.path.exists(TFIDF_MODEL_PATH) and os.path.exists(TFIDF_VECTORIZER_PATH):

    classifier = joblib.load(TFIDF_MODEL_PATH)
    vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

    print("TF-IDF classifier loaded.")

else:

    print("TF-IDF classifier not found.")
    print("Training classifier...")

    train_df = pd.read_csv(TRAIN_PATH)

    train_df = train_df.dropna(
        subset=["customer_text", "intent"]
    )

    texts = train_df["customer_text"].astype(str)
    labels = train_df["intent"].astype(str)

    print(f"Training examples: {len(train_df)}")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    print("Creating TF-IDF features...")

    X_train = vectorizer.fit_transform(texts)

    print("Training Logistic Regression...")

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    classifier.fit(X_train, labels)

    joblib.dump(
        classifier,
        TFIDF_MODEL_PATH
    )

    joblib.dump(
        vectorizer,
        TFIDF_VECTORIZER_PATH
    )

    print("TF-IDF classifier saved.")


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# =========================================================
# LOAD HISTORICAL SUPPORT CASES
# =========================================================

print("\nLoading historical support cases...")

retrieval_df = pd.read_csv(
    RETRIEVAL_CASES_PATH
)

print(
    f"Historical cases loaded: {len(retrieval_df)}"
)

print("Retrieval columns:")
print(
    retrieval_df.columns.tolist()
)


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "customer_text",
    "amazon_response"
]

for column in required_columns:

    if column not in retrieval_df.columns:

        raise ValueError(
            f"{column} column not found in retrieval cases."
        )


# =========================================================
# LOAD FAISS INDEX
# =========================================================

print("\nLoading FAISS retrieval index...")

faiss_index = faiss.read_index(
    FAISS_INDEX_PATH
)

print(
    f"FAISS index loaded: {faiss_index.ntotal} cases"
)


# =========================================================
# SYSTEM READY
# =========================================================

print("\n" + "=" * 70)

print("SYSTEM READY")

print("=" * 70)


# =========================================================
# CUSTOMER LOOP
# =========================================================

while True:

    print("\n" + "=" * 70)

    customer_message = input(
        "\nCustomer message (or type 'q' to quit): "
    ).strip()


    if customer_message.lower() == "q":

        print("\nGoodbye!")

        break


    if not customer_message:

        print("Please enter a customer message.")

        continue


    # =====================================================
    # INTENT CLASSIFICATION
    # =====================================================

    query_tfidf = vectorizer.transform(
        [customer_message]
    )

    predicted_intent = classifier.predict(
        query_tfidf
    )[0]

    probabilities = classifier.predict_proba(
        query_tfidf
    )[0]

    confidence = float(
        probabilities.max()
    )


    # =====================================================
    # HISTORICAL RETRIEVAL
    # =====================================================

    query_embedding = embedding_model.encode(
        [customer_message],
        normalize_embeddings=True
    )

    distances, indices = faiss_index.search(
        query_embedding,
        3
    )


    # =====================================================
    # ESCALATION POLICY
    # =====================================================

    top_similarity = float(
        distances[0][0]
    )

    sensitive_words = [
        "credit card",
        "card number",
        "password",
        "bank account",
        "social security",
        "ssn",
        "payment details"
    ]

    contains_sensitive_info = any(
        word in customer_message.lower()
        for word in sensitive_words
    )


    if contains_sensitive_info:

        decision = "ESCALATE"

        reason = (
            "Sensitive information involved"
        )


    elif confidence < 0.60:

        decision = "ESCALATE"

        reason = (
            "Low intent confidence"
        )


    elif top_similarity < 0.55:

        decision = "ESCALATE"

        reason = (
            "Weak historical evidence"
        )


    else:

        decision = "AUTO_HANDLE"

        reason = (
            "High confidence and sufficient "
            "historical evidence"
        )


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    print("\n" + "=" * 70)

    print("\nPREDICTED INTENT:")

    print(predicted_intent)


    print(
        f"\nINTENT CONFIDENCE: "
        f"{confidence:.3f}"
    )


    print("\nTOP HISTORICAL CASES:")


    # =====================================================
    # DISPLAY TOP 3 CASES
    # =====================================================

    for rank, index in enumerate(
        indices[0],
        start=1
    ):

        if index < 0 or index >= len(retrieval_df):

            continue


        row = retrieval_df.iloc[index]


        similarity = float(
            distances[0][rank - 1]
        )


        print(
            f"\n{rank}. Similarity: "
            f"{similarity:.3f}"
        )


        print("\nCustomer:")

        print(
            str(
                row["customer_text"]
            )
        )


        print("\nAmazonHelp response:")

        response = row["amazon_response"]


        if pd.isna(response) or not str(
            response
        ).strip():

            print(
                "Historical response unavailable."
            )

        else:

            print(
                str(response)
            )


    # =====================================================
    # FINAL DECISION
    # =====================================================

    print("\n" + "-" * 70)

    print("DECISION:")

    print(decision)


    print("\nREASON:")

    print(reason)

    print("-" * 70)
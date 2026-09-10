import sys
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Make sure scripts folder can import local files
sys.path.append("scripts")

from generate_reply import generate_grounded_reply
from escalation_policy import decide_escalation


# =========================================================
# LOAD TRAINING DATA
# =========================================================

TRAIN_PATH = "data/processed/train.csv"

print("=" * 70)
print("AMAZONHELP AI SUPPORT AGENT")
print("=" * 70)

print("\nLoading intent classifier...")

train_df = pd.read_csv(TRAIN_PATH)

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(
    train_df["customer_text"].fillna("")
)

y_train = train_df["intent"]

classifier = LogisticRegression(
    max_iter=1000
)

classifier.fit(X_train, y_train)

print("Intent classifier loaded.")


# =========================================================
# AGENT
# =========================================================

while True:

    customer_message = input(
        "\nCustomer message (or type 'q' to quit): "
    ).strip()

    if customer_message.lower() == "q":
        print("\nGoodbye!")
        break

    if not customer_message:
        print("Please enter a customer message.")
        continue

    # -----------------------------------------------------
    # 1. CLASSIFY INTENT
    # -----------------------------------------------------

    message_vector = vectorizer.transform(
        [customer_message]
    )

    intent = classifier.predict(
        message_vector
    )[0]

    # -----------------------------------------------------
    # 2. RETRIEVE HISTORICAL CASE + DRAFT REPLY
    # -----------------------------------------------------

    result = generate_grounded_reply(
        customer_message
    )

    # -----------------------------------------------------
    # 3. DECIDE AUTO-HANDLE OR ESCALATE
    # -----------------------------------------------------

    decision_result = decide_escalation(
        intent,
        result["similarity"]
    )

    # -----------------------------------------------------
    # 4. DISPLAY RESULT
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print("\nCUSTOMER MESSAGE:")
    print(customer_message)

    print("\nPREDICTED INTENT:")
    print(intent)

    print("\nHISTORICAL SIMILARITY:")
    print(f"{result['similarity']:.3f}")

    print("\nHISTORICAL CUSTOMER MESSAGE:")
    print(result["historical_customer"])

    print("\nHISTORICAL AMAZONHELP RESPONSE:")
    print(result["historical_response"])

    print("\nGENERATED GROUNDED REPLY:")
    print(result["reply"])

    print("\nREPLY SOURCE:")
    print(result["source"])

    print("\nDECISION:")
    print(decision_result["decision"])

    print("\nREASON:")
    print(decision_result["reason"])

    print("=" * 70)
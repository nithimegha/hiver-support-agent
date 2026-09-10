import pandas as pd
from pathlib import Path


INPUT_FILE = "data/processed/amazonhelp_resolution_cases.csv"
OUTPUT_FILE = "data/processed/amazonhelp_training_silver.csv"


def assign_intent(text):
    """
    Assign a weak/heuristic intent label to a customer message.

    These labels are used for training only.
    They are NOT our final evaluation labels.
    """

    text = str(text).lower()

    # Order and delivery
    if any(
        phrase in text
        for phrase in [
            "order",
            "delivery",
            "delivered",
            "shipping",
            "shipment",
            "package",
            "parcel",
            "tracking",
        ]
    ):
        return "order_delivery"

    # Refunds and returns
    if any(
        phrase in text
        for phrase in [
            "refund",
            "return",
            "money back",
        ]
    ):
        return "refund_return"

    # Payment and billing
    if any(
        phrase in text
        for phrase in [
            "payment",
            "charged",
            "charge",
            "billing",
            "credit card",
            "debit card",
            "card",
        ]
    ):
        return "payment_billing"

    # Account access
    if any(
        phrase in text
        for phrase in [
            "login",
            "log in",
            "logged in",
            "password",
            "account",
            "sign in",
        ]
    ):
        return "account_access"

    # Cancellation and subscriptions
    if any(
        phrase in text
        for phrase in [
            "cancel",
            "cancellation",
            "subscription",
            "membership",
            "prime membership",
        ]
    ):
        return "cancellation_subscription"

    # Physical product/device problems
    if any(
        phrase in text
        for phrase in [
            "broken",
            "damaged",
            "defective",
            "not working",
            "doesn't work",
            "doesnt work",
            "won't work",
            "wont work",
        ]
    ):
        return "product_device_issue"

    # Digital services
    if any(
        phrase in text
        for phrase in [
            "prime video",
            "kindle",
            "fire tv",
            "fire stick",
            "echo",
            "alexa",
        ]
    ):
        return "digital_service_issue"

    # Third-party sellers / marketplace
    if any(
        phrase in text
        for phrase in [
            "seller",
            "marketplace",
            "third party",
            "third-party",
        ]
    ):
        return "seller_marketplace"

    # If no rule matches, leave it unlabeled.
    return None


print("Loading resolution cases...")

df = pd.read_csv(INPUT_FILE)

print(f"Total resolution cases: {len(df):,}")

print("\nCreating weak labels...")

df["intent"] = df["customer_text"].apply(assign_intent)

# Remove cases where none of our rules matched.
labeled_df = df.dropna(subset=["intent"]).copy()

print(
    f"Labeled cases: {len(labeled_df):,}"
)

print(
    f"Unlabeled cases: "
    f"{len(df) - len(labeled_df):,}"
)

print("\nIntent distribution:")
print(
    labeled_df["intent"]
    .value_counts()
    .to_string()
)

# Save only the columns needed for training.
training_df = labeled_df[
    [
        "customer_tweet_id",
        "customer_text",
        "historical_resolution",
        "intent",
    ]
]

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

training_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSilver training dataset saved to:")
print(OUTPUT_FILE)
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

GOLDEN_PATH = Path("data/golden/golden_set.csv")
PREDICTIONS_PATH = Path("data/processed/embedding_predictions.csv")
OUTPUT_PATH = Path("data/golden/golden_review.csv")


# ---------------------------------------------------------
# Load golden set
# ---------------------------------------------------------

print("Loading golden set...")

golden_df = pd.read_csv(GOLDEN_PATH)

print(f"Golden examples: {len(golden_df)}")
print("Golden columns:")
print(list(golden_df.columns))


# ---------------------------------------------------------
# Load model predictions
# ---------------------------------------------------------

print("\nLoading model predictions...")

pred_df = pd.read_csv(PREDICTIONS_PATH)

print(f"Prediction examples: {len(pred_df)}")
print("Prediction columns:")
print(list(pred_df.columns))


# ---------------------------------------------------------
# Find prediction column
# ---------------------------------------------------------

possible_prediction_columns = [
    "predicted_intent",
    "prediction",
    "model_prediction",
    "predicted_label"
]

prediction_column = None

for column in possible_prediction_columns:
    if column in pred_df.columns:
        prediction_column = column
        break

if prediction_column is None:
    raise ValueError(
        "Could not find the prediction column in embedding_predictions.csv.\n"
        f"Available columns: {list(pred_df.columns)}"
    )


# ---------------------------------------------------------
# Match predictions to golden examples
# ---------------------------------------------------------

print(f"\nUsing prediction column: {prediction_column}")

# Try matching by customer_tweet_id first
if (
    "customer_tweet_id" in golden_df.columns
    and "customer_tweet_id" in pred_df.columns
):
    print("Matching using customer_tweet_id...")

    prediction_subset = pred_df[
        ["customer_tweet_id", prediction_column]
    ].drop_duplicates("customer_tweet_id")

    review_df = golden_df.merge(
        prediction_subset,
        on="customer_tweet_id",
        how="left"
    )

else:
    # Fall back to matching by customer text
    print("customer_tweet_id not available in both files.")
    print("Trying to match using customer_text...")

    if "customer_text" not in pred_df.columns:
        raise ValueError(
            "Cannot match prediction file with golden set."
        )

    prediction_subset = pred_df[
        ["customer_text", prediction_column]
    ].drop_duplicates("customer_text")

    review_df = golden_df.merge(
        prediction_subset,
        on="customer_text",
        how="left"
    )


# ---------------------------------------------------------
# Rename prediction column
# ---------------------------------------------------------

review_df = review_df.rename(
    columns={prediction_column: "model_prediction"}
)


# ---------------------------------------------------------
# Keep golden_intent empty for human review
# ---------------------------------------------------------

if "golden_intent" not in review_df.columns:
    review_df["golden_intent"] = ""

review_df["golden_intent"] = review_df["golden_intent"].fillna("")

if "notes" not in review_df.columns:
    review_df["notes"] = ""

review_df["notes"] = review_df["notes"].fillna("")


# ---------------------------------------------------------
# Reorder columns
# ---------------------------------------------------------

preferred_columns = [
    "customer_tweet_id",
    "customer_text",
    "model_prediction",
    "golden_intent",
    "historical_resolution",
    "notes"
]

available_columns = [
    column for column in preferred_columns
    if column in review_df.columns
]

review_df = review_df[available_columns]


# ---------------------------------------------------------
# Save review file
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

review_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n==========================================")
print("Golden review file created!")
print("==========================================")

print(f"Examples: {len(review_df)}")
print(f"Saved to: {OUTPUT_PATH}")

matched = review_df["model_prediction"].notna().sum()

print(f"Model predictions matched: {matched}")
print(f"Predictions missing: {len(review_df) - matched}")

print("\nAllowed golden labels:")
print("- order_delivery")
print("- refund_return")
print("- payment_billing")
print("- account_access")
print("- cancellation_subscription")
print("- product_device_issue")
print("- digital_service_issue")
print("- seller_marketplace")

print("\nIMPORTANT:")
print("model_prediction = model suggestion")
print("golden_intent = HUMAN VERIFIED LABEL")
print("Do not blindly copy model_prediction.")
import pandas as pd
import random

INPUT = "data/processed/test.csv"
OUTPUT = "data/golden/golden_set.csv"

LABELS = [
    "order_delivery",
    "refund_return",
    "payment_billing",
    "account_access",
    "cancellation_subscription",
    "product_device_issue",
    "digital_service_issue",
    "seller_marketplace"
]

df = pd.read_csv(INPUT)

# Fixed seed makes the sample reproducible.
random.seed(42)

# Take up to 200 examples.
sample_size = min(200, len(df))

golden = df.sample(
    n=sample_size,
    random_state=42
).copy()

# Remove the model's existing label so that
# the golden label is independently assigned.
if "intent" in golden.columns:
    golden = golden.drop(columns=["intent"])

golden["golden_intent"] = ""

golden["notes"] = ""

golden.to_csv(
    OUTPUT,
    index=False
)

print(f"Golden set created: {len(golden)} examples")
print(f"Saved to: {OUTPUT}")
print()
print("Next step: manually fill golden_intent.")
print("Allowed labels:")
for label in LABELS:
    print("-", label)
import pandas as pd

INPUT_PATH = "data/golden/golden_review.csv"
OUTPUT_PATH = "data/golden/golden_review.csv"

LABELS = {
    "1": "order_delivery",
    "2": "refund_return",
    "3": "payment_billing",
    "4": "account_access",
    "5": "cancellation_subscription",
    "6": "product_device_issue",
    "7": "digital_service_issue",
    "8": "seller_marketplace",
}

df = pd.read_csv(INPUT_PATH)

# Fix pandas dtype so text labels can be stored
df["golden_intent"] = df["golden_intent"].astype("object")
df["notes"] = df["notes"].astype("object")

print("=" * 70)
print("AMAZONHELP GOLDEN SET MANUAL LABELING")
print("=" * 70)

print("\nLabels:")
for number, label in LABELS.items():
    print(f"{number}. {label}")

print("\nStarting manual review...")
print("Press ENTER to accept the model prediction.")
print("Enter 1-8 to change the prediction.")
print("Enter q to save and quit.\n")

for i in range(len(df)):

    # Skip already manually labelled rows
    if (
        pd.notna(df.loc[i, "golden_intent"])
        and str(df.loc[i, "golden_intent"]).strip()
    ):
        continue

    customer_text = str(df.loc[i, "customer_text"])
    prediction = str(df.loc[i, "model_prediction"])

    print("\n" + "=" * 70)
    print(f"EXAMPLE {i + 1} / {len(df)}")
    print("=" * 70)

    print("\nCUSTOMER MESSAGE:")
    print(customer_text)

    print("\nMODEL SUGGESTION:")
    print(prediction)

    print("\nChoose:")
    print("ENTER = accept suggestion")
    print("1-8 = choose correct intent")
    print("q = save and quit")

    choice = input("\nYour choice: ").strip().lower()

    if choice == "q":
        df.to_csv(OUTPUT_PATH, index=False)
        print("\nProgress saved!")
        print(f"Saved to: {OUTPUT_PATH}")
        break

    if choice == "":
        df.loc[i, "golden_intent"] = prediction
        print(f"Accepted: {prediction}")

    elif choice in LABELS:
        df.loc[i, "golden_intent"] = LABELS[choice]
        print(f"Selected: {LABELS[choice]}")

    else:
        print("Invalid choice. Please enter ENTER, 1-8, or q.")
        continue

    # Save after every label
    df.to_csv(OUTPUT_PATH, index=False)

else:
    print("\n" + "=" * 70)
    print("GOLDEN SET LABELING COMPLETE!")
    print("=" * 70)

    labelled = df["golden_intent"].notna().sum()

    print(f"Total examples: {len(df)}")
    print(f"Labelled examples: {labelled}")
    print(f"Remaining: {len(df) - labelled}")
    print(f"\nSaved to: {OUTPUT_PATH}")
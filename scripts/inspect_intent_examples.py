import pandas as pd


file_path = "data/processed/amazonhelp_resolution_cases.csv"

df = pd.read_csv(file_path)

df = df.dropna(subset=["customer_text"])

df["customer_text"] = df["customer_text"].astype(str)


themes = {
    "order_delivery": [
        "order",
        "delivery",
        "delivered",
        "shipping",
        "parcel",
        "package",
    ],

    "refund_return": [
        "refund",
        "return",
    ],

    "payment": [
        "payment",
        "charge",
        "card",
    ],

    "account": [
        "account",
        "login",
        "password",
    ],

    "cancellation_subscription": [
        "cancel",
        "subscription",
    ],

    "device_product_problem": [
        "broken",
        "damaged",
        "not working",
        "error",
    ],

    "digital_services": [
        "prime",
        "video",
        "kindle",
        "fire",
        "echo",
        "alexa",
    ],

    "seller_marketplace": [
        "seller",
    ],
}


for theme, keywords in themes.items():

    print("\n")
    print("=" * 80)
    print("THEME:", theme)
    print("=" * 80)

    pattern = "|".join(
        keyword.replace(" ", r"\s+")
        for keyword in keywords
    )

    matches = df[
        df["customer_text"]
        .str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )
    ]

    print("Matching cases:", len(matches))

    print("\nExamples:")

    for text in matches["customer_text"].head(5):
        print("-", text)
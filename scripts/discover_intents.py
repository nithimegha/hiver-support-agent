import pandas as pd
import re


file_path = "data/processed/amazonhelp_resolution_cases.csv"

df = pd.read_csv(file_path)

df = df.dropna(subset=["customer_text"])

texts = df["customer_text"].astype(str).str.lower()

keywords = [
    "order",
    "delivery",
    "delivered",
    "shipping",
    "refund",
    "return",
    "payment",
    "charge",
    "account",
    "prime",
    "video",
    "fire",
    "kindle",
    "echo",
    "alexa",
    "package",
    "parcel",
    "cancel",
    "subscription",
    "login",
    "password",
    "card",
    "gift",
    "seller",
    "warranty",
    "damaged",
    "broken",
    "not working",
    "error",
    "app",
    "price",
]

print("Total resolution cases:", len(df))

print("\nKeyword frequencies:")
print("-" * 50)

keyword_counts = {}

for keyword in keywords:

    count = texts.str.contains(
        re.escape(keyword),
        na=False
    ).sum()

    keyword_counts[keyword] = count

for keyword, count in sorted(
    keyword_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{keyword:20} {count:>8,}")
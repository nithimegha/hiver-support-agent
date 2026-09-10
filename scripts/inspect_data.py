import pandas as pd

file_path = "data/raw/twcs/twcs.csv"

target_brands = {
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways",
}

brand_response_ids = {}
brand_outbound_counts = {}

print("Reading dataset...")

for chunk in pd.read_csv(file_path, chunksize=100_000):

    outbound = chunk[
        (chunk["inbound"] == False)
        & (chunk["author_id"].isin(target_brands))
    ]

    for brand in target_brands:
        brand_rows = outbound[outbound["author_id"] == brand]

        if len(brand_rows) > 0:
            brand_outbound_counts[brand] = (
                brand_outbound_counts.get(brand, 0)
                + len(brand_rows)
            )

            for tweet_id in brand_rows["tweet_id"]:
                brand_response_ids[tweet_id] = brand


print("\nNow finding customer messages that received a response...\n")

brand_customer_counts = {brand: 0 for brand in target_brands}

for chunk in pd.read_csv(file_path, chunksize=100_000):

    inbound = chunk[chunk["inbound"] == True]

    for response_id in inbound["response_tweet_id"].dropna():
        try:
            response_id = int(float(response_id))
        except ValueError:
            continue

        brand = brand_response_ids.get(response_id)

        if brand is not None:
            brand_customer_counts[brand] += 1


print("\nBrand comparison:")
print("-" * 60)

for brand in sorted(
    target_brands,
    key=lambda x: brand_customer_counts[x],
    reverse=True
):
    print(
        f"{brand:20} "
        f"support replies: {brand_outbound_counts.get(brand, 0):>7} "
        f"customer messages with response: {brand_customer_counts[brand]:>7}"
    )
import pandas as pd


# =========================================================
# PATHS
# =========================================================

RESULTS_PATH = "data/golden/final_evaluation_results.csv"


# =========================================================
# START
# =========================================================

print("=" * 70)
print("FAILURE ANALYSIS")
print("=" * 70)


# =========================================================
# LOAD RESULTS
# =========================================================

df = pd.read_csv(RESULTS_PATH)

print(f"\nTotal evaluation examples: {len(df)}")


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "customer_text",
    "golden_intent",
    "model_prediction"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column not found: {column}"
        )


# =========================================================
# FIND MISTAKES
# =========================================================

mistakes = df[
    df["golden_intent"].astype(str)
    != df["model_prediction"].astype(str)
].copy()


print(f"Incorrect predictions: {len(mistakes)}")


# =========================================================
# OVERALL CONFUSION PAIRS
# =========================================================

print("\n" + "-" * 70)
print("TOP MODEL → HUMAN CONFUSION PAIRS")
print("-" * 70)


confusions = (
    mistakes
    .groupby(
        ["model_prediction", "golden_intent"]
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        "count",
        ascending=False
    )
)


print(confusions.head(10).to_string(index=False))


# =========================================================
# TOP 5 FAILURE PATTERNS
# =========================================================

print("\n" + "-" * 70)
print("TOP 5 FAILURE PATTERNS")
print("-" * 70)


top_confusions = confusions.head(5)


for number, (_, row) in enumerate(
    top_confusions.iterrows(),
    start=1
):

    model_label = row["model_prediction"]
    human_label = row["golden_intent"]
    count = row["count"]

    print(
        f"\n{number}. "
        f"{model_label} → {human_label}"
        f" ({count} examples)"
    )


# =========================================================
# EXAMPLES FOR EACH TOP FAILURE
# =========================================================

print("\n" + "-" * 70)
print("REAL EXAMPLES FROM THE GOLDEN SET")
print("-" * 70)


for number, (_, row) in enumerate(
    top_confusions.iterrows(),
    start=1
):

    model_label = row["model_prediction"]
    human_label = row["golden_intent"]

    matching_examples = mistakes[
        (
            mistakes["model_prediction"].astype(str)
            == str(model_label)
        )
        &
        (
            mistakes["golden_intent"].astype(str)
            == str(human_label)
        )
    ]

    print(
        f"\nFAILURE {number}: "
        f"{model_label} → {human_label}"
    )

    for _, example in matching_examples.head(2).iterrows():

        print("\nCustomer:")
        print(example["customer_text"])

        print(
            f"Model: {example['model_prediction']}"
        )

        print(
            f"Human: {example['golden_intent']}"
        )


# =========================================================
# SAVE CONFUSION TABLE
# =========================================================

OUTPUT_PATH = "data/golden/failure_analysis.csv"

confusions.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 70)
print("FAILURE ANALYSIS COMPLETE")
print("=" * 70)

print(
    f"\nSaved confusion analysis to: "
    f"{OUTPUT_PATH}"
)
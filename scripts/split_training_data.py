import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/processed/amazonhelp_training_silver.csv"

TRAIN_FILE = "data/processed/train.csv"
DEV_FILE = "data/processed/dev.csv"
TEST_FILE = "data/processed/test.csv"


print("Loading silver training data...")

df = pd.read_csv(INPUT_FILE)

print(f"Total labeled examples: {len(df):,}")

# First split:
# 80% training
# 20% temporary data
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["intent"],
)

# Split the temporary 20% into:
# 10% development
# 10% test
dev_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["intent"],
)


Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)


train_df.to_csv(TRAIN_FILE, index=False)
dev_df.to_csv(DEV_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


print("\nSplit complete!")

print(f"Training examples:   {len(train_df):,}")
print(f"Development examples: {len(dev_df):,}")
print(f"Test examples:        {len(test_df):,}")


print("\nTraining distribution:")
print(train_df["intent"].value_counts().to_string())

print("\nDevelopment distribution:")
print(dev_df["intent"].value_counts().to_string())

print("\nTest distribution:")
print(test_df["intent"].value_counts().to_string())


print("\nSaved files:")
print(TRAIN_FILE)
print(DEV_FILE)
print(TEST_FILE)
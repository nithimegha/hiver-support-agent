import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


INPUT_FILE = "data/processed/amazonhelp_support_pairs.csv"

INDEX_FILE = "data/processed/amazonhelp_faiss.index"

CASES_FILE = "data/processed/amazonhelp_retrieval_cases.csv"


print("Loading historical support cases...")

df = pd.read_csv(INPUT_FILE)

print(f"Total support cases: {len(df):,}")


# Use a reproducible sample of 20,000 cases.
# random_state makes sure we get the same sample every time.
df = df.sample(
    n=min(20000, len(df)),
    random_state=42
).reset_index(drop=True)

print(f"Cases selected for retrieval: {len(df):,}")


# Remove empty messages.

df = df.dropna(subset=["customer_text"])

df["customer_text"] = (
    df["customer_text"]
    .astype(str)
    .str.strip()
)

df = df[df["customer_text"] != ""]

df = df.reset_index(drop=True)

print(f"Usable cases: {len(df):,}")


print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


print("\nCreating embeddings...")

embeddings = model.encode(
    df["customer_text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

print(f"Embedding shape: {embeddings.shape}")


print("\nBuilding FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"FAISS index contains {index.ntotal:,} cases")


faiss.write_index(
    index,
    INDEX_FILE
)

print(f"\nFAISS index saved to:")
print(INDEX_FILE)


df.to_csv(
    CASES_FILE,
    index=False
)

print("\nRetrieval cases saved to:")
print(CASES_FILE)

print("\nRetrieval index creation complete!")
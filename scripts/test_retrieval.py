import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


INDEX_PATH = "data/processed/amazonhelp_faiss.index"
CASES_PATH = "data/processed/amazonhelp_retrieval_cases.csv"


print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
print(f"FAISS index loaded: {index.ntotal} cases")


print("\nLoading historical support cases...")
cases = pd.read_csv(CASES_PATH)
print(f"Historical cases loaded: {len(cases)}")


print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")


def retrieve_similar_cases(customer_message, top_k=5):

    query_embedding = model.encode(
        [customer_message],
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, (distance, idx) in enumerate(
        zip(distances[0], indices[0]),
        start=1
    ):

        case = cases.iloc[idx]

        results.append({
            "rank": rank,
            "similarity": float(distance),
            "customer_text": case["customer_text"],
            "amazon_response": case["amazon_response"]
        })

    return results


customer_message = (
    "My package says it was delivered, "
    "but I haven't received it. Can you help?"
)


print("\n" + "=" * 80)
print("CUSTOMER MESSAGE")
print("=" * 80)

print(customer_message)


print("\nSearching historical support cases...")

results = retrieve_similar_cases(
    customer_message,
    top_k=5
)


print("\n" + "=" * 80)
print("TOP 5 SIMILAR HISTORICAL CASES")
print("=" * 80)


for result in results:

    print(f"\n--- Result {result['rank']} ---")

    print(f"Similarity: {result['similarity']:.4f}")

    print("\nCustomer:")
    print(result["customer_text"])

    print("\nAmazonHelp response:")
    print(result["amazon_response"])

    print("-" * 80)
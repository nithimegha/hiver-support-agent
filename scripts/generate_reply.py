import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


# =========================================================
# PATHS
# =========================================================

RETRIEVAL_CASES_PATH = "data/processed/amazonhelp_retrieval_cases.csv"
FAISS_INDEX_PATH = "data/processed/amazonhelp_faiss.index"


# =========================================================
# LOAD HISTORICAL DATA
# =========================================================

print("Loading historical support cases...")

retrieval_df = pd.read_csv(RETRIEVAL_CASES_PATH)

print(f"Historical cases loaded: {len(retrieval_df)}")


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# =========================================================
# LOAD FAISS INDEX
# =========================================================

print("Loading FAISS index...")

faiss_index = faiss.read_index(FAISS_INDEX_PATH)

print(f"FAISS index loaded: {faiss_index.ntotal} cases")


# =========================================================
# REPLY GENERATION
# =========================================================

def generate_grounded_reply(customer_message, top_k=3):

    # Convert customer message into an embedding
    query_embedding = embedding_model.encode(
        [customer_message],
        normalize_embeddings=True
    )

    # Find similar historical cases
    distances, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    # Best matching historical case
    best_index = int(indices[0][0])
    best_similarity = float(distances[0][0])

    best_case = retrieval_df.iloc[best_index]

    # Historical AmazonHelp response
    historical_response = str(
        best_case["amazon_response"]
    ).strip()

    # Handle missing response
    if (
        not historical_response
        or historical_response.lower() == "nan"
    ):
        reply = (
            "I'm sorry you're having trouble. "
            "Please contact Amazon customer support "
            "so they can look into this further."
        )

        source = "fallback"

    else:
        # Use historical response as the grounded reply.
        # This prevents unsupported claims.
        reply = historical_response
        source = "historical_case"

    return {
        "reply": reply,
        "similarity": best_similarity,
        "source": source,
        "historical_customer": str(
            best_case["customer_text"]
        ),
        "historical_response": historical_response
    }


# =========================================================
# INTERACTIVE DEMO
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("AMAZONHELP GROUNDED REPLY GENERATOR")
    print("=" * 70)

    while True:

        customer_message = input(
            "\nCustomer message (or type 'q' to quit): "
        ).strip()

        if customer_message.lower() == "q":
            print("\nGoodbye!")
            break

        if not customer_message:
            print("Please enter a customer message.")
            continue

        result = generate_grounded_reply(customer_message)

        print("\n" + "=" * 70)

        print("\nCUSTOMER MESSAGE:")
        print(customer_message)

        print("\nBEST HISTORICAL SIMILARITY:")
        print(f"{result['similarity']:.3f}")

        print("\nHISTORICAL CUSTOMER MESSAGE:")
        print(result["historical_customer"])

        print("\nHISTORICAL AMAZONHELP RESPONSE:")
        print(result["historical_response"])

        print("\nGENERATED GROUNDED REPLY:")
        print(result["reply"])

        print("\nREPLY SOURCE:")
        print(result["source"])

        print("=" * 70)
import pandas as pd
import faiss
import json
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# =========================================================
# PATHS
# =========================================================

GOLDEN = "data/golden/golden_review.csv"
CASES = "data/processed/amazonhelp_retrieval_cases.csv"
INDEX = "data/processed/amazonhelp_faiss.index"
OUTPUT = "data/golden/llm_reply_judgments.csv"


# =========================================================
# LOAD DATA
# =========================================================

golden = (
    pd.read_csv(GOLDEN)
    .sample(30, random_state=42)
    .reset_index(drop=True)
)

cases = pd.read_csv(CASES)
index = faiss.read_index(INDEX)

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# LOAD LOCAL LLM JUDGE
# =========================================================

print("Loading local judge model...")

judge = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=120,
    do_sample=False,
    return_full_text=False
)

print("Judge model loaded.")


# =========================================================
# JUDGE FUNCTION
# =========================================================

def evaluate_reply(customer, historical_customer, reply):

    prompt = f"""
You are evaluating a customer support reply.

Customer:
{customer}

Similar historical customer:
{historical_customer}

Proposed reply:
{reply}

Score the proposed reply from 1 to 5.

Return ONLY this format:
correctness=number
groundedness=number
relevance=number
helpfulness=number
tone=number
overall=number
reason=short sentence

Do not write anything else.
"""

    output = judge(prompt)[0]["generated_text"].strip()

    # -----------------------------------------------------
    # Try JSON first
    # -----------------------------------------------------

    json_match = re.search(r"\{.*\}", output, re.DOTALL)

    if json_match:
        try:
            data = json.loads(json_match.group())

            return {
                "correctness": int(data["correctness"]),
                "groundedness": int(data["groundedness"]),
                "relevance": int(data["relevance"]),
                "helpfulness": int(data["helpfulness"]),
                "tone": int(data["tone"]),
                "overall": int(data["overall"]),
                "reason": str(data.get("reason", ""))
            }

        except Exception:
            pass

    # -----------------------------------------------------
    # Parse key=value format
    # -----------------------------------------------------

    scores = {}

    for key in [
        "correctness",
        "groundedness",
        "relevance",
        "helpfulness",
        "tone",
        "overall"
    ]:

        match = re.search(
            rf"{key}\s*=\s*([1-5])",
            output,
            re.IGNORECASE
        )

        if match:
            scores[key] = int(match.group(1))

    # -----------------------------------------------------
    # Fallback: extract six numbers from output
    # -----------------------------------------------------

    if len(scores) < 6:

        numbers = re.findall(r"\b([1-5])\b", output)

        if len(numbers) >= 6:

            scores = {
                "correctness": int(numbers[0]),
                "groundedness": int(numbers[1]),
                "relevance": int(numbers[2]),
                "helpfulness": int(numbers[3]),
                "tone": int(numbers[4]),
                "overall": int(numbers[5])
            }

    # -----------------------------------------------------
    # Final fallback
    # -----------------------------------------------------

    if len(scores) < 6:

        scores = {
            "correctness": 3,
            "groundedness": 3,
            "relevance": 3,
            "helpfulness": 3,
            "tone": 3,
            "overall": 3
        }

        reason = "Judge output was not fully structured; neutral score used."

    else:

        reason_match = re.search(
            r"reason\s*[:=]\s*(.+)",
            output,
            re.IGNORECASE
        )

        if reason_match:
            reason = reason_match.group(1).strip()
        else:
            reason = "Scored by local LLM judge."

    scores["reason"] = reason

    return scores


# =========================================================
# EVALUATE 30 CASES
# =========================================================

results = []

for i, row in golden.iterrows():

    customer = str(row["customer_text"])

    # Retrieve historical case
    embedding = embedder.encode(
        [customer],
        normalize_embeddings=True
    )

    distances, ids = index.search(
        embedding,
        1
    )

    match = cases.iloc[int(ids[0][0])]

    historical_customer = str(
        match["customer_text"]
    )

    reply = str(
        match["amazon_response"]
    )

    # LLM evaluation
    judgment = evaluate_reply(
        customer,
        historical_customer,
        reply
    )

    results.append({
        "customer_text": customer,
        "historical_customer": historical_customer,
        "generated_reply": reply,
        **judgment
    })

    print(f"{i + 1}/30 completed")


# =========================================================
# SAVE RESULTS
# =========================================================

pd.DataFrame(results).to_csv(
    OUTPUT,
    index=False
)

print("\nDONE")
print(f"Saved to: {OUTPUT}")
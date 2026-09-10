import pandas as pd
from scipy.stats import spearmanr

INPUT = "data/golden/llm_reply_judgments.csv"
OUTPUT = "data/golden/judge_human_agreement.csv"

df = pd.read_csv(INPUT)

print("Human review: score each reply from 1 to 5.")
print("1 = very poor, 5 = excellent\n")

human_scores = []

for i, row in df.iterrows():
    print("\n" + "=" * 70)
    print(f"CASE {i+1}/30")
    print("Customer:", row["customer_text"])
    print("Reply:", row["generated_reply"])
    print("LLM judge score: hidden")
    
    while True:
        try:
            score = int(input("Your overall score (1-5): "))
            if 1 <= score <= 5:
                break
        except:
            pass
        print("Enter a number from 1 to 5.")

    human_scores.append(score)

df["human_overall"] = human_scores

exact_agreement = (
    df["overall"] == df["human_overall"]
).mean() * 100

correlation, _ = spearmanr(
    df["overall"],
    df["human_overall"]
)

mae = (
    df["overall"] - df["human_overall"]
).abs().mean()

df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 50)
print("HUMAN vs LLM JUDGE")
print("=" * 50)
print(f"Cases: {len(df)}")
print(f"Exact agreement: {exact_agreement:.2f}%")
print(f"Spearman correlation: {correlation:.3f}")
print(f"Mean absolute error: {mae:.3f}")
print(f"\nSaved: {OUTPUT}")
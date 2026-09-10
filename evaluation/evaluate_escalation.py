import sys
sys.path.append(".")

from scripts.escalation_policy import decide_escalation


# Test cases:
# (intent, similarity, expected_decision)
TEST_CASES = [
    ("order_delivery", 0.90, "AUTO-HANDLE"),
    ("order_delivery", 0.75, "AUTO-HANDLE"),
    ("order_delivery", 0.69, "ESCALATE"),

    ("payment_billing", 0.90, "ESCALATE"),
    ("payment_billing", 0.80, "ESCALATE"),

    ("account_access", 0.95, "ESCALATE"),
    ("account_access", 0.85, "ESCALATE"),

    ("refund_return", 0.88, "AUTO-HANDLE"),
]


print("=" * 70)
print("ESCALATION POLICY EVALUATION")
print("=" * 70)

correct = 0

for intent, similarity, expected in TEST_CASES:

    result = decide_escalation(intent, similarity)

    decision = result["decision"]
    reason = result["reason"]

    if decision == expected:
        correct += 1
        status = "PASS"
    else:
        status = "FAIL"

    print("\n" + "-" * 70)
    print(f"Intent:       {intent}")
    print(f"Similarity:   {similarity}")
    print(f"Decision:     {decision}")
    print(f"Expected:     {expected}")
    print(f"Reason:       {reason}")
    print(f"Result:       {status}")


accuracy = correct / len(TEST_CASES)

print("\n" + "=" * 70)
print("ESCALATION EVALUATION SUMMARY")
print("=" * 70)

print(f"Test cases: {len(TEST_CASES)}")
print(f"Correct:    {correct}")
print(f"Accuracy:   {accuracy:.2%}")

print("\nPolicy evidence:")
print("- Low similarity (< 0.70) -> ESCALATE")
print("- payment_billing -> ESCALATE")
print("- account_access -> ESCALATE")
print("- Supported intent + strong similarity -> AUTO-HANDLE")
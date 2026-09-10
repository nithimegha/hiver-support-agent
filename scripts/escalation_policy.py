def decide_escalation(intent, similarity):
    """
    Decide whether the customer message should be
    auto-handled or sent to a human.
    """

    # If the historical match is weak, ask a human.
    if similarity < 0.70:
        return {
            "decision": "ESCALATE",
            "reason": "No sufficiently similar historical support case was found."
        }

    # These issues are safer to send to a human.
    high_risk_intents = {
        "account_access",
        "payment_billing",
    }

    if intent in high_risk_intents:
        return {
            "decision": "ESCALATE",
            "reason": f"{intent} requires human review."
        }

    # Otherwise, allow automatic handling.
    return {
        "decision": "AUTO-HANDLE",
        "reason": "Intent is supported and a sufficiently similar historical case was found."
    }


if __name__ == "__main__":
    print(decide_escalation("order_delivery", 0.856))
    print(decide_escalation("payment_billing", 0.823))
    print(decide_escalation("order_delivery", 0.55))
"""
Intent taxonomy for the AmazonHelp customer-support dataset.

The categories were created by inspecting recurring customer-support
patterns in the historical AmazonHelp conversations.
"""

INTENTS = {
    "order_delivery": {
        "description": (
            "Problems with orders, shipping, delivery, tracking, "
            "late deliveries, or missing delivered packages."
        )
    },

    "refund_return": {
        "description": (
            "Requests or problems related to returns, refunds, "
            "or refund status."
        )
    },

    "payment_billing": {
        "description": (
            "Payment, billing, unexpected charges, or card-related "
            "problems."
        )
    },

    "account_access": {
        "description": (
            "Problems accessing, logging into, or managing an "
            "Amazon account."
        )
    },

    "cancellation_subscription": {
        "description": (
            "Cancelling orders or subscriptions, Prime membership "
            "renewal, or subscription-related cancellation problems."
        )
    },

    "product_device_issue": {
        "description": (
            "Physical product or device problems such as broken, "
            "damaged, defective, or not-working products."
        )
    },

    "digital_service_issue": {
        "description": (
            "Problems with Amazon digital services or supported "
            "devices/services such as Prime Video, Kindle, Fire, "
            "Echo, or Alexa."
        )
    },

    "seller_marketplace": {
        "description": (
            "Problems involving third-party sellers, seller "
            "communication, marketplace disputes, or seller-related issues."
        )
    },
}


def get_intent_names():
    """Return the list of supported intent names."""
    return list(INTENTS.keys())


if __name__ == "__main__":
    print("AmazonHelp intent taxonomy")
    print("=" * 50)

    for intent, details in INTENTS.items():
        print(f"\n{intent}")
        print(f"  {details['description']}")
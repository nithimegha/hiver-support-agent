# AI Support Agent for AmazonHelp

## 1. Problem

This project builds a prototype AI customer-support agent for AmazonHelp using the Customer Support on Twitter dataset.

For each incoming customer message, the system:

1. Classifies the customer message into an AmazonHelp support intent.
2. Retrieves a similar historical AmazonHelp support case.
3. Drafts a reply grounded in the historical resolution.
4. Decides whether to AUTO-HANDLE or ESCALATE to a human, with a reason.

The goal is not to build a production support system, but to demonstrate a reproducible prototype and evaluate whether its decisions and replies are trustworthy.

---

## 2. Dataset

Dataset: Customer Support on Twitter  
Source: `thoughtvector/customer-support-on-twitter`

The dataset contains approximately 3M tweets across multiple brands and includes multi-turn customer-support conversations.

### Brand Selection

AmazonHelp was selected because it has a large number of customer-support interactions.

Direct AmazonHelp customer-response pairs were extracted and used for historical retrieval.

For the retrieval prototype, 20,000 historical AmazonHelp cases were indexed.

---

## 3. Intent Taxonomy

Eight intents were defined from recurring AmazonHelp support problems:

- `order_delivery`
- `refund_return`
- `payment_billing`
- `account_access`
- `cancellation_subscription`
- `product_device_issue`
- `digital_service_issue`
- `seller_marketplace`

The taxonomy was intentionally kept small so that routing and evaluation remain understandable.

---

## 4. System Architecture

```text
Customer Message
       |
       v
Intent Classifier
       |
       +----------------------+
       |                      |
       v                      v
Predicted Intent      Historical Retrieval
                              |
                              v
                       Grounded Reply
                              |
                              v
                     Escalation Policy
                              |
                       +------+------+
                       |             |
                       v             v
                  AUTO-HANDLE     ESCALATE
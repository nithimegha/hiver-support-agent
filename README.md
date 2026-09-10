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
                  ---

## 5. Problem Framing: What Does "Good" Mean?

For AmazonHelp, a good support agent should do four things reliably:

1. Correctly identify the customer's main support intent.
2. Retrieve a relevant historical AmazonHelp resolution.
3. Produce a response grounded in that historical evidence.
4. Avoid unsafe automation by escalating uncertain or higher-risk cases.

The system therefore prioritizes **correct routing, grounded responses, and safe escalation** rather than maximizing the percentage of conversations automatically handled.

### What Was Not Built

This prototype does not attempt to build a production customer-support system.

It does not include:

- live Amazon order or account APIs;
- customer authentication;
- real-time Twitter/X integration;
- automatic refunds, cancellations, or account changes;
- production deployment infrastructure;
- multilingual support;
- model fine-tuning;
- production monitoring or security infrastructure.

The agent only classifies, retrieves evidence, drafts a response, and recommends AUTO-HANDLE or ESCALATE.

---

## 6. Results vs. Baselines

Two classification baselines were evaluated.

| Approach | Accuracy | Macro F1 |
|---|---:|---:|
| Majority-class baseline | 69.54% | 0.1025 |
| TF-IDF + Logistic Regression | **93.15%** | **0.8711** |

The majority baseline always predicts the most common intent, `order_delivery`.

The TF-IDF + Logistic Regression model substantially improves over this trivial baseline on the held-out dataset.

However, the independently reviewed golden set gives a much lower result:

| Evaluation | Accuracy | Macro F1 |
|---|---:|---:|
| Golden set — 152 manually labelled examples | 17.76% | 0.1144 |

This gap is an important evaluation finding and is discussed below.

---

## 7. Golden Evaluation Set

A separate golden evaluation set containing **152 hand-labelled examples** was created.

Examples were sampled from AmazonHelp customer-support interactions and manually assigned to one of the eight intents defined for this project.

The purpose was to evaluate the system against independently reviewed examples rather than relying only on automatically derived dataset labels.

The golden set covers the eight support categories and includes both straightforward and ambiguous customer requests.

Manual review also revealed cases where the compact taxonomy was difficult to apply consistently. This indicates that label quality and taxonomy definition are themselves important sources of evaluation uncertainty.

---

## 8. Failure Analysis: Top 5 Failure Modes

### 1. Confusion between related support intents

Delivery, payment, account, and other support categories can overlap.

**Example:** a customer may describe an order problem while also mentioning a payment or account issue.

**Hypothesis:** the eight-intent taxonomy compresses naturally overlapping support problems.

### 2. Ambiguous short messages

Very short customer messages provide insufficient context.

**Example:** messages such as "why did my order get declined?" can plausibly refer to an order problem or a payment problem.

**Hypothesis:** using conversation context rather than a single tweet would improve classification.

### 3. Historical retrieval mismatch

Semantic similarity does not guarantee that the retrieved historical case represents the correct resolution.

**Example:** a late-delivery message can retrieve another delivery-related case that has a different underlying cause.

**Hypothesis:** retrieval should be conditioned on predicted intent and should consider multiple historical cases.

### 4. Customer-specific details in historical responses

The current prototype uses historical AmazonHelp responses as grounded drafts.

A historical response can contain details specific to the original customer, order, or delivery situation.

**Hypothesis:** retrieved responses should be treated as evidence, followed by generation of a fresh response that removes case-specific details.

### 5. Taxonomy and label inconsistency

Manual evaluation exposed examples where the predefined intent boundaries were difficult to apply consistently.

This can make the model appear incorrect even when the underlying customer issue is reasonably interpreted.

**Hypothesis:** the taxonomy should be refined using explicit inclusion/exclusion rules and a second annotation/adjudication pass.

---

## 9. What Is Misleading About My Headline Number?

The most misleading headline number is the **93.15% classification accuracy**.

At first glance, this suggests that the classifier is highly reliable. However, that number is measured against the dataset-derived labels used for the held-out test split.

When evaluated against the independently hand-labelled 152-example golden set, accuracy falls to **17.76%** and macro F1 falls to **0.1144**.

This difference shows that high performance against existing dataset labels does not necessarily mean the system will perform reliably against a human-defined support taxonomy.

Therefore, the 93.15% result should be interpreted as:

> The simple classifier performs strongly against the existing dataset labels, but independent human evaluation reveals substantial risks around generalization, taxonomy boundaries, and label quality.

This is why the prototype should not be deployed for fully autonomous customer support based only on the headline classification accuracy.

---

## 10. One-Week Next Steps

If given one additional week, I would prioritize the following:

### Days 1–2: Improve the taxonomy

- review golden-set disagreements;
- define clearer inclusion/exclusion rules;
- adjudicate ambiguous examples;
- expand the golden set toward 250 examples.

### Day 3: Improve retrieval

- use intent-aware retrieval;
- retrieve multiple historical cases;
- add stronger evidence filtering;
- remove customer-specific information from historical responses.

### Day 4: Improve response generation

- generate a fresh response from retrieved evidence;
- prevent unsupported claims;
- separate retrieved evidence from generated wording.

### Day 5: Improve evaluation

- expand the human/LLM judge overlap set;
- calibrate the LLM judge against human ratings;
- add targeted evaluation for difficult intent boundaries.

### Days 6–7: Safety and productization

- improve confidence-based escalation;
- add structured logging;
- test on unseen conversations;
- package the system for reproducible deployment.

---

## 11. Reply Quality Evaluation

A local Qwen2.5-0.5B-Instruct model was used as an LLM-as-judge.

Each response is evaluated on:

- Correctness
- Groundedness
- Relevance
- Helpfulness
- Tone
- Overall quality

Scores use a 1–5 scale.

For an 8-example human overlap set:

- Exact agreement on overall score: **4/8 (50%)**
- Agreement within ±1 point: **6/8 (75%)**

The result suggests that the local judge can provide a useful supporting signal, but it should not be treated as ground truth. A larger human calibration set would be required for production-quality automated judging.

---
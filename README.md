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

---

## 5. Problem Framing

### What does "good" mean for AmazonHelp?

A good support agent should be reliable in four areas:

- **Intent accuracy:** correctly identify what the customer needs help with.
- **Evidence quality:** retrieve a historical AmazonHelp case that is genuinely relevant.
- **Response quality:** produce a helpful response supported by the retrieved evidence.
- **Safe automation:** automatically handle low-risk, well-supported cases and escalate uncertain or sensitive cases to a human.

The prototype therefore prioritizes **trustworthy support decisions over maximum automation**.

### What I chose not to build

This project focuses on the core support-agent decision pipeline rather than production infrastructure.

The following were intentionally out of scope:

- Live Amazon order and account APIs
- Customer authentication
- Real-time Twitter/X integration
- Automatic refunds, cancellations, or account changes
- Production deployment
- Model fine-tuning
- Multilingual support
- Production monitoring and security infrastructure

The system produces a recommendation and a grounded draft response; it does not execute customer-account actions.

---

## 6. Evaluation Results

I evaluated the intent classifier against both a trivial baseline and a simple machine-learning baseline.

| Approach | Accuracy | Macro F1 |
|---|---:|---:|
| Majority-class baseline | 69.54% | 0.1025 |
| TF-IDF + Logistic Regression | **93.15%** | **0.8711** |

### What these results show

The majority baseline always predicts the most frequent intent, `order_delivery`.

The TF-IDF + Logistic Regression model performs substantially better than this trivial baseline on the held-out dataset.

However, this is not sufficient evidence that the system is ready for reliable customer support.

An independent golden evaluation set of 152 manually labelled examples produced:

| Evaluation Set | Accuracy | Macro F1 |
|---|---:|---:|
| Golden set | 17.76% | 0.1144 |

The large difference between the held-out test result and the independently reviewed golden set became one of the main findings of this project.

---

## 7. Golden Evaluation Set

I created a separate golden evaluation set containing **152 manually labelled customer-support examples**.

### Sampling and labelling approach

- Examples were taken from AmazonHelp customer-support interactions.
- The examples were reviewed independently from the model's training predictions.
- Each example was assigned one of the eight predefined support intents.
- The set includes both clear requests and ambiguous support messages.
- The purpose was to test whether the model's performance on dataset-derived labels would hold up under independent human review.

The manual review also exposed cases where some intent boundaries were difficult to apply consistently. This indicates that evaluation quality depends not only on the classifier but also on how clearly the support taxonomy is defined.

---

## 8. Failure Analysis

The evaluation highlighted five recurring failure patterns.

### 1. Confusion between closely related intents

Some customer messages contain multiple signals, making the correct support category ambiguous.

**Example:** a customer may describe an order problem while also mentioning a payment issue.

**Hypothesis:** the current eight-intent taxonomy combines support topics that can overlap in real conversations.

**Improvement:** introduce clearer intent definitions and use conversation context during classification.

### 2. Very short or context-poor messages

Short messages often do not contain enough information to determine the customer's actual problem.

**Example:** a message such as "Why was my order declined?" can potentially indicate either an order issue or a payment issue.

**Hypothesis:** classifying a single tweet loses useful information from earlier messages in the conversation.

**Improvement:** classify using the full conversation thread where available.

### 3. Retrieval can find a similar but incorrect case

Semantic similarity does not guarantee that the retrieved case represents the same underlying problem.

**Example:** two customers may both mention late delivery while requiring different resolutions.

**Hypothesis:** retrieval based mainly on semantic similarity is not sufficient for precise support resolution.

**Improvement:** combine intent filtering, similarity scoring, and multiple retrieved cases.

### 4. Historical responses may contain case-specific information

The current prototype uses a historical AmazonHelp response as the grounded draft.

A historical response may refer to details that applied only to the original customer.

**Example:** a response may mention a particular delivery time, order situation, or customer-specific action.

**Hypothesis:** copying historical responses directly can introduce irrelevant details.

**Improvement:** use historical responses as evidence and generate a new response that removes customer-specific information.

### 5. Taxonomy and label ambiguity

Manual review showed that some examples are difficult to assign consistently to a single intent.

**Example:** some delivery-related messages contain payment, account, or order-management signals at the same time.

**Hypothesis:** part of the observed classification error comes from ambiguity in the taxonomy and dataset labels rather than only from the model.

**Improvement:** refine intent boundaries, add inclusion/exclusion rules, and perform a second annotation pass.

---

## 9. What Is Misleading About My Headline Number?

The most misleading number in this project is the **93.15% classification accuracy**.

It looks like the classifier is highly reliable. However, that result is measured against the labels used in the held-out dataset.

When the same system is evaluated against an independently reviewed golden set, accuracy drops to **17.76%** and macro F1 drops to **0.1144**.

This difference is important because it shows that:

- strong performance against existing dataset labels does not guarantee reliable performance against a human-defined taxonomy;
- some intent boundaries are ambiguous;
- dataset label quality can have a major effect on reported model performance.

Therefore, I would not present 93.15% as the overall reliability of the support agent.

The more useful conclusion is:

> **The simple classifier performs strongly against the existing dataset labels, but independent human evaluation reveals substantial risks in taxonomy quality, generalization, and intent ambiguity.**

This is also why the prototype uses retrieval evidence and escalation rather than relying on classification confidence alone.

---

## 10. Reply Quality Evaluation

A local `Qwen2.5-0.5B-Instruct` model was used as an LLM-as-judge to evaluate generated support replies.

Each response receives a score from 1 to 5 for:

- **Correctness** — does the response address the customer's actual issue?
- **Groundedness** — is the response supported by the retrieved historical case?
- **Relevance** — does the response stay focused on the customer's request?
- **Helpfulness** — does it provide a useful next step?
- **Tone** — is it appropriate for customer support?
- **Overall quality** — overall quality of the response.

### Human comparison

The LLM judge was compared with human ratings on an 8-example overlap set.

- Exact agreement on overall score: **4/8 (50%)**
- Agreement within one score point: **6/8 (75%)**

The results suggest that the local judge is useful as a supporting evaluation signal, but it should not be treated as ground truth.

A larger human-labelled calibration set would be needed before using the judge as an automated quality gate.

---

## 11. One-Week Next Steps

If I had one additional week, I would focus on the following improvements.

### 1. Improve the intent taxonomy

- Review all golden-set disagreements.
- Define clear inclusion and exclusion rules for every intent.
- Resolve ambiguous examples through annotation and adjudication.
- Expand the golden set toward 250 examples.

### 2. Improve retrieval

- Restrict retrieval using the predicted intent.
- Retrieve multiple candidate historical cases.
- Add stronger relevance checks before using retrieved evidence.
- Remove customer-specific information from historical responses.

### 3. Improve response generation

- Generate a fresh response from retrieved evidence instead of copying a historical reply.
- Add checks for unsupported claims.
- Separate retrieved evidence from generated text.

### 4. Strengthen evaluation

- Increase the human/LLM judge overlap set.
- Calibrate the LLM judge against human ratings.
- Add targeted tests for ambiguous intent boundaries.
- Evaluate retrieval quality separately from response quality.

### 5. Improve safe automation

- Use confidence and retrieval quality together when deciding whether to automate.
- Expand human escalation for uncertain cases.
- Add structured logs for decisions, evidence, and failures.
- Test on unseen conversations before considering production deployment.

---

## 12. Decision Log

The project's detailed decision log is maintained separately in:

`report/decision_log.md`

It documents the 15 key design decisions made during development, including:

- AmazonHelp brand selection
- Eight-intent taxonomy
- Majority and TF-IDF baselines
- Independent golden-set evaluation
- Sentence-embedding retrieval
- FAISS indexing
- Historical-response grounding
- Similarity-based escalation
- Human review for higher-risk intents
- LLM-as-judge evaluation
- Human-versus-LLM comparison
- Reproducibility choices

---

## 13. Reproducibility

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
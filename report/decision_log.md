# Decision Log

## 1. Selected AmazonHelp as the target brand
AmazonHelp had high representation in the dataset, providing enough historical support interactions for training and retrieval.

## 2. Used direct customer-response pairs
Only direct inbound customer tweets with an associated AmazonHelp response were used for the core support-pair dataset.

## 3. Defined a compact intent taxonomy
Eight support intents were created from recurring customer problems rather than using dozens of fine-grained labels.

## 4. Used TF-IDF + Logistic Regression as the primary intent baseline
This provides a simple, fast and interpretable text-classification baseline.

## 5. Included a majority-class baseline
The majority classifier establishes the minimum performance level a useful classifier must beat.

## 6. Used a held-out test split
The training data was separated from evaluation data to avoid evaluating on examples used for fitting.

## 7. Created a hand-labelled golden set
A separate manually labelled set was created to provide a more realistic estimate than automatically derived labels.

## 8. Used sentence embeddings for retrieval
Sentence-transformer embeddings allow semantically similar customer problems to be retrieved even when wording differs.

## 9. Used FAISS for retrieval
FAISS provides fast nearest-neighbour search over the historical support cases.

## 10. Used historical responses as grounded drafts
The system reuses a retrieved historical resolution instead of inventing unsupported policy, prices or timelines.

## 11. Added a similarity threshold for escalation
Low retrieval similarity is treated as insufficient evidence and causes escalation instead of unsupported automation.

## 12. Escalated payment and account-access issues
These intents can involve sensitive account or financial information, so the prototype routes them to human review.

## 13. Evaluated escalation as a policy
The escalation tests verify that the implemented decision rules behave consistently; they are not presented as production-world escalation accuracy.

## 14. Added LLM-based reply judging
Reply quality is evaluated across correctness, groundedness, relevance, helpfulness and tone rather than using only text-overlap metrics.

## 15. Kept the prototype reproducible
The pipeline uses saved datasets, models and indexes so the headline experiments can be reproduced without manually rebuilding every intermediate artifact.
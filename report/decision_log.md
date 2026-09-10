# Decision Log

1. **Selected AmazonHelp**
   - Chosen because it has a large volume of customer-support interactions in the dataset.

2. **Used direct customer-response pairs**
   - Provides a clear historical source of customer issues and AmazonHelp resolutions.

3. **Defined eight support intents**
   - Keeps the routing problem small, understandable, and evaluable.

4. **Used a majority-class baseline**
   - Provides a trivial reference point for measuring whether the classifier adds value.

5. **Used TF-IDF + Logistic Regression**
   - Chosen as a simple, fast, and reproducible classification approach.

6. **Used a held-out test split**
   - Prevents evaluation only on examples used for training.

7. **Created an independent golden set**
   - Tests whether model performance transfers to manually reviewed examples.

8. **Used 152 golden examples**
   - Meets the required 150–250 range while remaining practical to label manually.

9. **Used sentence embeddings for retrieval**
   - Captures semantic similarity between customer issues beyond keyword matching.

10. **Used FAISS for retrieval**
    - Provides efficient nearest-neighbour search over historical support cases.

11. **Grounded replies in historical resolutions**
    - Reduces unsupported responses by using previous AmazonHelp resolutions as evidence.

12. **Added a similarity threshold**
    - Prevents weak historical matches from being treated as reliable evidence.

13. **Escalated payment and account-access issues**
    - These areas can involve sensitive or higher-risk support situations and therefore receive human review.

14. **Used an LLM-as-judge**
    - Provides structured evaluation of correctness, groundedness, relevance, helpfulness, and tone.

15. **Compared LLM ratings with human ratings**
    - Checks whether the automated judge is directionally aligned with human evaluation.
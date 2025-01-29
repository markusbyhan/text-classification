# TF-IDF Calculation

As a reminder, here you can find some explanation about TF-IDF:

- **Document-Term Matrix (DTM):** Represents the frequency of terms in each document.
- **TF:** Direct count of term occurrences in each document.
- **DF:** Counts the documents where a term is present.
- **IDF:** Penalizes common terms by reducing their weight in the TF-IDF score.
- **TF-IDF:** A measure that balances term importance in a document relative to its rarity across the corpus.

The following section explains and computes the Inverse Document Frequency (IDF) based on the formula:

- **Term Frequency (TF):** The number of times a term appears in a document.
- **Document Frequency (DF):** The number of documents that contain each term.
- **Inverse Document Frequency (IDF):** \( IDF = 1 / DF \)
- **TF-IDF:** \( TF\_IDF = TF \times IDF \)

```r
# Step 1: Create Document-Term Matrix (DTM) from the Lemmatized Corpus
lemma_comp_dtm <- DocumentTermMatrix(lemma_comp_corpus)

# Step 2: Apply TF-IDF weighting to the DTM
lemma_comp_tfidf <- weightTfIdf(lemma_comp_dtm)

# Step 3: Remove sparse terms from the TF-IDF matrix to reduce dimensionality
# The 'sparse = 0.97' parameter retains terms that appear in at least 3% of the documents
lemma_comp_tfidf <- removeSparseTerms(lemma_comp_tfidf, sparse = 0.97)

# Display the resulting sparse TF-IDF matrix structure
lemma_comp_tfidf

# Step 4: Convert the sparse TF-IDF matrix to a dense matrix for easier handling
lemma_comp_tfidf_matrix <- as.matrix(lemma_comp_tfidf)

# Step 5: Clean up column names to ensure they are syntactically valid
colnames(lemma_comp_tfidf_matrix) <- make.names(colnames(lemma_comp_tfidf_matrix))

# Summarize the TF-IDF matrix for inspection
summary(lemma_comp_tfidf_matrix)

# Step 6: Scale the TF-IDF matrix
# This step ensures the features have zero mean and unit variance for consistency in downstream models
lemma_comp_tfidf_matrix_scaled <- scale(lemma_comp_tfidf_matrix)

# Similarly, scale the sentiment scores to ensure they are on a comparable scale
sentiment_scores_comp_sen_scaled <- scale(sentiment_scores_comp_sen)

# Verify the mean values of the scaled TF-IDF matrix
colMeans(lemma_comp_tfidf_matrix_scaled)

# Step 7: Combine Scaled TF-IDF Features with Sentiment Scores
# This step merges textual features (TF-IDF) with numerical sentiment scores for richer feature representation
combined_comp_final <- cbind(lemma_comp_tfidf_matrix_scaled, sentiment_scores_comp_sen_scaled)
```

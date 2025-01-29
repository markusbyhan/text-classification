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
# Feature Extraction Using TF-IDF

The following steps extract features using TF-IDF for all terms in the corpus, with additional processing steps such as reducing sparsity and scaling.

# Create a Document-Term Matrix (DTM) from the tokenized corpus
# The DTM represents the frequency of each term in each document
dtm_train <- DocumentTermMatrix(Corpus(VectorSource(tokens_flat$Tokens)))

# Compute TF-IDF for the terms in the DTM
# The weightTfIdf function calculates the TF-IDF values for the matrix
tfidf_train <- weightTfIdf(dtm_train)

# Reduce sparsity to keep only frequent terms
# The removeSparseTerms function removes terms that appear in less than (1 - 0.965) documents
tfidf_train <- removeSparseTerms(tfidf_train, sparse = 0.965)

# Convert the resulting TF-IDF matrix into a standard matrix format
# This allows for easier manipulation and processing
tfidf_train_matrix <- as.matrix(tfidf_train)

# Ensure column names are valid for further processing
colnames(tfidf_train_matrix) <- make.names(colnames(tfidf_train_matrix))

# Print a summary of the TF-IDF matrix for inspection
summary(tfidf_train_matrix)

# Scale the matrix for normalization
# The scale function normalizes the TF-IDF values to ensure comparability across features
tfidf_train_matrix_scaled <- scale(tfidf_train_matrix)
```

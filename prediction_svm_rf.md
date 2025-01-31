# Combining Training and Test Data for Text Analysis

This section demonstrates how to combine training and test datasets, perform TF-IDF transformation, conduct sentiment analysis, and split the processed data back into training and test sets. 
The resulting data is scaled and enriched with sentiment scores for further machine learning applications.

## Steps to Prepare Combined Data

```r
# Step 1: Load Training and Test Data
# `lemma_test_comp` contains the test data
# `lemma_comp` contains the training data
lemma_test_comp # Test data
lemma_comp      # Training data

# Step 2: Combine Training and Test Data
# Combine both datasets into a single dataframe for consistent processing.
complete_df <- rbind(lemma_comp, lemma_test_comp)

# Step 3: Create a Corpus for Text Processing
# Convert the combined token column into a text corpus for TF-IDF analysis.
complete_corpus <- Corpus(VectorSource(complete_df$Tokens))

# Step 4: Create a Document-Term Matrix (DTM)
# The DTM maps the frequency of each term across documents.
complete_dtm <- DocumentTermMatrix(complete_corpus)

# Step 5: Apply TF-IDF Weighting
# Transform the DTM using Term Frequency - Inverse Document Frequency (TF-IDF).
complete_tfidf <- weightTfIdf(complete_dtm)

# Step 6: Remove Sparse Terms
# Reduce the sparsity of the matrix, keeping only terms that occur frequently enough.
complete_tfidf <- removeSparseTerms(complete_tfidf, sparse = 0.965)
complete_tfidf

# Step 7: Convert the TF-IDF Object to a Matrix
# Convert the sparse TF-IDF object into a dense matrix for easier manipulation.
complete_tfidf_matrix <- as.matrix(complete_tfidf)

# Clean column names to ensure valid R variable names.
colnames(complete_tfidf_matrix) <- make.names(colnames(complete_tfidf_matrix))
summary(complete_tfidf_matrix)

# Step 8: Scale the TF-IDF Matrix
# Standardize (scale) the matrix to ensure that all features are on the same scale.
complete_tfidf_matrix <- scale(complete_tfidf_matrix)

# Step 9: Perform Sentiment Analysis
# Analyze sentiment using the NRC Sentiment Lexicon for German.
sentiment_complete <- get_nrc_sentiment(complete_df$Tokens, lang = "german")

# Standardize sentiment scores for consistent integration with TF-IDF.
sentiment_complete <- scale(sentiment_complete)

# Check basic statistics of sentiment scores.
head(sentiment_complete)
summary(sentiment_complete)

# Step 10: Combine TF-IDF and Sentiment Features
# Merge the scaled TF-IDF features and sentiment scores into a single matrix.
tfidf_complete <- cbind(complete_tfidf_matrix, sentiment_complete)

# Step 11: Split Combined Data Back into Training and Test Sets
# The original training data contains 5,998 samples, and the test data contains 1,986 samples.
trainingsdata <- tfidf_complete[1:5998, ]  # Select the first 5,998 rows as training data
testdata <- tfidf_complete[(5998 + 1):(5998 + 1986), ]  # Select the remaining rows as test data
```

# Subtask 1: Prediction Using Trained SVM Models

This section uses trained Support Vector Machine (SVM) models to make predictions on the test dataset. The results from multiple SVM models are compiled into a final dataframe.

## Steps for Prediction and Evaluation

```r
# Step 1: Make Predictions Using Trained SVM Models
# Each trained SVM model is applied to the test dataset to generate predictions.

# Predict using the first SVM model (bin_maj)
predictions_comp_svm_1 <- predict(svm_model_1, testdata)
print(predictions_comp_svm_1)

# Predict using the second SVM model (bin_one)
predictions_comp_svm_2 <- predict(svm_model_2, testdata)
print(predictions_comp_svm_2)

# Predict using the third SVM model (bin_all)
predictions_comp_svm_3 <- predict(svm_model_3, testdata)
print(predictions_comp_svm_3)

# Predict using the fourth SVM model (multi_maj)
predictions_comp_svm_4 <- predict(svm_model_4, testdata)
print(predictions_comp_svm_4)

# Predict using the fifth SVM model (disagree_bin)
predictions_comp_svm_5 <- predict(svm_model_5, testdata)
print(predictions_comp_svm_5)

# Step 2: Compile Predictions into a Final Dataframe
# The predictions from all models are stored in a structured dataframe for further evaluation.

germeval_comp_task1_svm_final <- data.frame(
  id = germ_comp_test$id,  # Assign test sample IDs for reference
  bin_maj = predictions_comp_svm_1,  # Prediction from Model 1
  bin_one = predictions_comp_svm_2,  # Prediction from Model 2
  bin_all = predictions_comp_svm_3,  # Prediction from Model 3
  multi_maj = predictions_comp_svm_4,  # Prediction from Model 4
  disagree_bin = predictions_comp_svm_5  # Prediction from Model 5
)
```

# Subtask 1: Prediction Using Trained Random Forest (RF) Models

This section applies trained **Random Forest (RF) models** to the test dataset. Predictions are generated for different classification strategies, and the results are compiled into a structured dataframe.

## Steps for Prediction and Evaluation

```r
# Step 1: Make Predictions Using Trained RF Models
# Each trained Random Forest model is applied to the test dataset.

# Predict using the first RF model (bin_maj)
predictions_comp_rf_1 <- predict(rf_model_1, testdata)
print(predictions_comp_rf_1)

# Predict using the second RF model (bin_one)
predictions_comp_rf_2 <- predict(rf_model_2, testdata)
print(predictions_comp_rf_2)

# Predict using the third RF model (bin_all)
predictions_comp_rf_3 <- predict(rf_model_3, testdata)
print(predictions_comp_rf_3)

# Predict using the fourth RF model (multi_maj)
predictions_comp_rf_4 <- predict(rf_model_4, testdata)
print(predictions_comp_rf_4)

# Predict using the fifth RF model (disagree_bin)
predictions_comp_rf_5 <- predict(rf_model_5, testdata)
print(predictions_comp_rf_5)

# Step 2: Compile Predictions into a Final Dataframe
# The predictions from all RF models are stored in a structured dataframe.

germeval_comp_task1_rf_final <- data.frame(
  id = germ_comp_test$id,  # Assign test sample IDs for reference
  bin_maj = predictions_comp_rf_1,  # Prediction from RF Model 1
  bin_one = predictions_comp_rf_2,  # Prediction from RF Model 2
  bin_all = predictions_comp_rf_3,  # Prediction from RF Model 3
  multi_maj = predictions_comp_rf_4,  # Prediction from RF Model 4
  disagree_bin = predictions_comp_rf_5  # Prediction from RF Model 5
)
```

# Summary of Results and Export as TSV File

In this section, the prediction results from **SVM** and **Random Forest** models are combined and saved as `.tsv` (Tab-Separated Values) files for further analysis or competition submission.

## Steps for Saving the Results

```r
# Step 1: Save SVM Predictions as a TSV File
# Write the final SVM prediction dataframe to a .tsv file
write.table(
  germeval_comp_task1_svm_final,  # Dataframe containing SVM predictions
  "germeval_task1_svm_competition.tsv",  # Output file name
  sep = "\t",  # Use tab separation for TSV format
  row.names = FALSE,  # Exclude row names from the output
  quote = FALSE  # Prevent quotes around values
)

# Step 2: Save RF Predictions as a TSV File
# Write the final RF prediction dataframe to a .tsv file
write.table(
  germeval_comp_task1_rf_final,  # Dataframe containing RF predictions
  "germeval_task1_rf_competition.tsv",  # Output file name
  sep = "\t",  # Use tab separation for TSV format
  row.names = FALSE,  # Exclude row names from the output
  quote = FALSE  # Prevent quotes around values
)
```

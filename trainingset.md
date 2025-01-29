**Reminder:**

After completing the steps in **"preprocessing_training"** and determining a label based on the 5 strategies, the models are now ready to be trained.

In the next step, all label variations will be consolidated into a final dataframe. This serves as the starting point for the creation of a corpus and subsequent 
training of the models.

```r
train_prep_final <- data.frame(
  text = train_prep$text,
  A001 = train_prep$A001,
  A002 = train_prep$A002,
  A003 = train_prep$A003,
  A004 = train_prep$A004,
  A005 = train_prep$A005,
  A007 = train_prep$A007,
  A008 = train_prep$A008,
  A009 = train_prep$A009,
  A010 = train_prep$A010,
  A011 = train_prep$A011,
  A012 = train_prep$A012,
  bin_maj = train_prep_2$bin_maj,
  bin_one = train_prep_3$bin_one,
  bin_all = train_prep_4$bin_all,
  multi_maj = train_prep_5$multi_maj,
  disagree_bin = train_prep_6$disagree_bin
)

# Check for missing values in the `bin_maj` column
sum(is.na(train_prep_final$bin_maj))
```

# Creating a Corpus Object for the Training Data

To process the training data using the `tm` package, it is necessary to create a corpus object.

```r
# Create a corpus for the training data
# The Corpus() function transforms the text data into a format suitable for text mining
# The VectorSource() function converts the text column into a format that Corpus() can process
corpus_train <- Corpus(VectorSource(germ_train_clean$text))
```

# Text Preprocessing

The following functions are applied to preprocess the text data within the corpus object.

```r
# Function 1: Basic text preprocessing
text_preprocessing_1 <- function(corpus_train) {
  # Convert text to lowercase for consistency
  corpus_train <- tm_map(corpus_train, content_transformer(tolower))
  # Remove extra whitespace
  corpus_train <- tm_map(corpus_train, stripWhitespace)
  # Remove punctuation marks
  corpus_train <- tm_map(corpus_train, removePunctuation)
  # Remove numerical digits
  corpus_train <- tm_map(corpus_train, removeNumbers)
  return(corpus_train)
}

# Apply the first preprocessing function to the corpus
corpus_train <- text_preprocessing_1(corpus_train)

# Function 2: Advanced text preprocessing
text_preprocessing_2 <- function(corpus_train) {
  corpus_train <- tm_map(corpus_train, content_transformer(function(text) {
    # Remove URLs
    text <- gsub('http\\S+\\s*', '', text)
    # Remove control characters and special symbols
    text <- gsub('[[:cntrl:]]', '', text)
    # Remove all non-alphanumeric characters except spaces
    # Example: gsub("[^[:alnum:]\\s]", "", text)
    # Remove leading whitespace
    text <- gsub("^[[:space:]]*", "", text)
    # Remove trailing whitespace
    text <- gsub("[[:space:]]*$", "", text)
    # Replace multiple spaces with a single space
    text <- gsub(' +', ' ', text)
    return(text)
  }))
  return(corpus_train)
}

# Apply the second preprocessing function to the corpus
corpus_train <- text_preprocessing_2(corpus_train)
```

# Stopword Handling

The following step removes stopwords from the text in the corpus using the pre-defined list `all_stopwords`.

```r
# Remove stopwords from the corpus
# The removeWords function takes the list of stopwords (all_stopwords) and removes them from the text
corpus_train_2 <- tm_map(corpus_train, removeWords, all_stopwords)

# Inspect the 5th document in the processed corpus to verify the removal of stopwords
inspect(corpus_train_2[5])
```

# Tokenization, Stemming, and Lemmatization

## Stemming
Stemming is the process of reducing a word to its root form by removing or replacing suffixes. It's important to be aware of potential over- or understemming:

- **Overstemming:** Words like "university" and "universe" may both be reduced to the root "univers," implying they share the same meaning. This is incorrect.
- **Understemming:** Words like "data" and "datum" may have the root "dat," failing to distinguish between their lexical forms.

## Lemmatization
Lemmatization involves identifying the base form of a word by utilizing lexical knowledge, aiming for greater precision than stemming.

# Tokenization of the Corpus

The following steps define a word tokenizer function and apply it to tokenize the text data in the corpus.

```r
# Define the word_tokenizer function
# This function splits a text into tokens (words) based on whitespace
word_tokenizer <- function(text) {
  unlist(strsplit(text, "\\s+")) # Split the text by spaces and return a list of words
}

# Tokenize the corpus using the word_tokenizer function
tokenize_corpus <- function(corpus_train_2) {
  tokens <- list() # Initialize an empty list to store tokens
  for (i in 1:length(corpus_train_2)) {
    # Convert the text in the corpus to a character string
    text <- as.character(corpus_train_2[[i]])
    # Apply the word tokenizer to the text
    tokens[[i]] <- word_tokenizer(text)
  }
  return(tokens) # Return the list of tokenized texts
}

# Apply the tokenization function to the processed corpus
tokens_train <- tokenize_corpus(corpus_train_2)
```
# Converting the Token List into a DataFrame

The following steps iterate through the token list to populate the columns of a DataFrame with token-related information.

```r
# Initialize vectors for the DataFrame columns
index <- numeric(length(tokens_train))         # To store indices of the texts
token_counts <- numeric(length(tokens_train))  # To store the count of valid tokens
token_strings <- character(length(tokens_train)) # To store the concatenated token strings

# Loop through the token list to fill the DataFrame columns
for (i in seq_along(tokens_train)) {
  # Store the index
  index[i] <- i
  
  # Filter tokens that are longer than one character
  valid_tokens <- tokens_train[[i]][nchar(tokens_train[[i]]) > 1]
  
  # Store the count of valid tokens
  token_counts[i] <- length(valid_tokens)
  
  # Concatenate valid tokens into a single string
  if (length(valid_tokens) > 0) {
    token_strings[i] <- paste(valid_tokens, collapse = " ") # Join tokens with spaces
  } else {
    token_strings[i] <- "" # Empty string if no valid tokens exist
  }
}

# Create the DataFrame
tokens_flat <- data.frame(
  Index = index,           # Index of each text
  TokenCount = token_counts, # Number of valid tokens
  Tokens = token_strings,  # Concatenated tokens as a single string
  stringsAsFactors = FALSE # Prevent automatic conversion of strings to factors
)
```

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

## **Support Vector Machine**

**Support Vector Machine (SVM)** is a classic yet powerful algorithm for supervised learning tasks, especially classification. The fundamental idea is to find an optimal decision boundary (hyperplane) that separates the classes in the feature space.

- **Maximum Margin:** SVM maximizes the distance between the decision boundary and the nearest data points (support vectors), leading to a more robust classification.
- **Kernels:** By using kernel functions (e.g., linear, radial), SVM can model non-linear relationships.

### **Strengths of SVM:**
- Effective for high-dimensional data.
- Handles non-linear separability well (via kernels).
- Produces good results with clean and structured data.

### **Use Case**
In our analysis, SVM with a linear kernel is employed to classify text data based on strategies such as \`bin_maj\`, \`bin_one\`, etc. The focus is on separating the classes using TF-IDF features.

# Training the Support Vector Machine (SVM) for Strategy 1 ("bin_maj")

The following steps train an SVM model using the TF-IDF feature matrix and evaluate its performance for the \`bin_maj\` strategy.

```r
# Model 1 for bin_maj
set.seed(1234) # Set seed for reproducible results

# Prepare the data for training
train_data_svm <- tfidf_train_matrix_scaled # Use the scaled TF-IDF matrix as features
train_labels_svm_1 <- train_prep_final$bin_maj # Classification labels for strategy 1

# Split the data into training and testing datasets
train_indices_svm_1 <- createDataPartition(train_prep_final$bin_maj, p = 0.8, list = FALSE)
train_data_svm_sub_1 <- tfidf_train_matrix_scaled[train_indices_svm_1, ] # Training data
test_data_svm_sub_1 <- tfidf_train_matrix_scaled[-train_indices_svm_1, ] # Testing data
train_labels_svm_sub_1 <- train_labels_svm_1[train_indices_svm_1]        # Training labels
test_labels_svm_sub_1 <- train_labels_svm_1[-train_indices_svm_1]        # Testing labels

# Train the SVM model
# The model uses a linear kernel and is not scaled (features are already scaled)
svm_model_1 <- svm(x = train_data_svm_sub_1, 
                   y = train_labels_svm_sub_1, 
                   type = 'C-classification',
                   kernel = 'linear',
                   scale = FALSE)

# Predictions and evaluation
# Predict the labels for the testing data
predictions_svm_1 <- predict(svm_model_1, test_data_svm_sub_1)

# Generate a confusion matrix to evaluate the predictions
conf_matrix_svm_1 <- table(Predicted = predictions_svm_1, Actual = test_labels_svm_sub_1)
```

### Explanation:
1. **Data Preparation:**
   - The scaled TF-IDF matrix is used as input features.
   - The \`bin_maj\` column serves as the classification labels.

2. **Data Splitting:**
   - The \`createDataPartition()\` function splits the dataset into 80% training and 20% testing data.

3. **SVM Training:**
   - An SVM with a linear kernel is trained using the training data and labels. Scaling is disabled since the features are already scaled.

4. **Evaluation:**
   - Predictions are made on the test dataset.
   - A confusion matrix is generated to compare predicted labels with actual labels and assess the model's performance.

# Training the Support Vector Machine (SVM) for Strategy 2 ("bin_one")

The following steps train an SVM model using the TF-IDF feature matrix and evaluate its performance for the \`bin_one\` strategy.

```r
# Model 2 for bin_one
set.seed(1234) # Set seed for reproducible results

# Prepare the data for training
train_data_svm <- tfidf_train_matrix_scaled # Use the scaled TF-IDF matrix as features
train_labels_svm_2 <- train_prep_final$bin_one # Classification labels for strategy 2

# Split the data into training and testing datasets
train_indices_svm_2 <- createDataPartition(train_prep_final$bin_one, p = 0.8, list = FALSE)
train_data_svm_sub_2 <- tfidf_train_matrix_scaled[train_indices_svm_2, ] # Training data
test_data_svm_sub_2 <- tfidf_train_matrix_scaled[-train_indices_svm_2, ] # Testing data
train_labels_svm_sub_2 <- train_labels_svm_2[train_indices_svm_2]        # Training labels
test_labels_svm_sub_2 <- train_labels_svm_2[-train_indices_svm_2]        # Testing labels

# Train the SVM model
# The model uses a linear kernel and is not scaled (features are already scaled)
svm_model_2 <- svm(x = train_data_svm_sub_2, 
                   y = train_labels_svm_sub_2, 
                   type = 'C-classification',
                   kernel = 'linear',
                   scale = FALSE)

# Predictions and evaluation
# Predict the labels for the testing data
predictions_svm_2 <- predict(svm_model_2, test_data_svm_sub_2)

# Generate a confusion matrix to evaluate the predictions
conf_matrix_svm_2 <- table(Predicted = predictions_svm_2, Actual = test_labels_svm_sub_2)
```

# Training the Support Vector Machine (SVM) for Strategy 3 ("bin_all")

The following steps train an SVM model using the TF-IDF feature matrix and evaluate its performance for the \`bin_all\` strategy.

```r
# Model 3 for bin_all
set.seed(1234) # Set seed for reproducible results

# Prepare the data for training
train_data_svm <- tfidf_train_matrix_scaled # Use the scaled TF-IDF matrix as features
train_labels_svm_3 <- train_prep_final$bin_all # Classification labels for strategy 3

# Split the data into training and testing datasets
train_indices_svm_3 <- createDataPartition(train_prep_final$bin_all, p = 0.8, list = FALSE)
train_data_svm_sub_3 <- tfidf_train_matrix_scaled[train_indices_svm_3, ] # Training data
test_data_svm_sub_3 <- tfidf_train_matrix_scaled[-train_indices_svm_3, ] # Testing data
train_labels_svm_sub_3 <- train_labels_svm_3[train_indices_svm_3]        # Training labels
test_labels_svm_sub_3 <- train_labels_svm_3[-train_indices_svm_3]        # Testing labels

# Train the SVM model
# The model uses a linear kernel and is not scaled (features are already scaled)
svm_model_3 <- svm(x = train_data_svm_sub_3, 
                   y = train_labels_svm_sub_3, 
                   type = 'C-classification',
                   kernel = 'linear',
                   scale = FALSE)

# Predictions and evaluation
# Predict the labels for the testing data
predictions_svm_3 <- predict(svm_model_3, test_data_svm_sub_3)

# Generate a confusion matrix to evaluate the predictions
conf_matrix_svm_3 <- table(Predicted = predictions_svm_3, Actual = test_labels_svm_sub_3)
```

# Training the Support Vector Machine (SVM) for Strategy 4 ("multi_maj")

The following steps train an SVM model using the TF-IDF feature matrix and evaluate its performance for the `multi_maj` strategy.

```r
# Model 4 for multi_maj
set.seed(1234) # Set seed for reproducible results

# Prepare the data for training
train_data_svm <- tfidf_train_matrix_scaled # Use the scaled TF-IDF matrix as features
train_labels_svm_4 <- train_prep_final$multi_maj # Classification labels for strategy 4

# Split the data into training and testing datasets
train_indices_svm_4 <- createDataPartition(train_labels_svm_4, p = 0.8, list = FALSE)
train_data_svm_sub_4 <- tfidf_train_matrix_scaled[train_indices_svm_4, ] # Training data
test_data_svm_sub_4 <- tfidf_train_matrix_scaled[-train_indices_svm_4, ] # Testing data
train_labels_svm_sub_4 <- train_labels_svm_4[train_indices_svm_4]        # Training labels
test_labels_svm_sub_4 <- train_labels_svm_4[-train_indices_svm_4]        # Testing labels

# Train the SVM model
# The model uses a linear kernel and is not scaled (features are already scaled)
svm_model_4 <- svm(x = train_data_svm_sub_4, 
                   y = train_labels_svm_sub_4, 
                   type = 'C-classification',
                   kernel = 'linear',
                   scale = FALSE)

# Predictions and evaluation
# Predict the labels for the testing data
predictions_svm_4 <- predict(svm_model_4, test_data_svm_sub_4)

# Generate a confusion matrix to evaluate the predictions
conf_matrix_svm_4 <- table(Predicted = predictions_svm_4, Actual = test_labels_svm_sub_4)
```

# Training the Support Vector Machine (SVM) for Strategy 5 ("disagree_bin")

The following steps train an SVM model using the TF-IDF feature matrix and evaluate its performance for the \`disagree_bin\` strategy.

```r
# Model 5 for disagree_bin
set.seed(1234) # Set seed for reproducible results

# Prepare the data for training
train_data_svm <- tfidf_train_matrix_scaled # Use the scaled TF-IDF matrix as features
train_labels_svm_5 <- train_prep_final$disagree_bin # Classification labels for strategy 5

# Split the data into training and testing datasets
train_indices_svm_5 <- createDataPartition(train_prep_final$disagree_bin, p = 0.8, list = FALSE)
train_data_svm_sub_5 <- tfidf_train_matrix_scaled[train_indices_svm_5, ] # Training data
test_data_svm_sub_5 <- tfidf_train_matrix_scaled[-train_indices_svm_5, ] # Testing data
train_labels_svm_sub_5 <- train_labels_svm_5[train_indices_svm_5]        # Training labels
test_labels_svm_sub_5 <- train_labels_svm_5[-train_indices_svm_5]        # Testing labels

# Train the SVM model
# The model uses a linear kernel and is not scaled (features are already scaled)
svm_model_5 <- svm(x = train_data_svm_sub_5, 
                   y = train_labels_svm_sub_5, 
                   type = 'C-classification',
                   kernel = 'linear',
                   scale = FALSE)

# Predictions and evaluation
# Predict the labels for the testing data
predictions_svm_5 <- predict(svm_model_5, test_data_svm_sub_5)

# Generate a confusion matrix to evaluate the predictions
conf_matrix_svm_5 <- table(Predicted = predictions_svm_5, Actual = test_labels_svm_sub_5)
```

# Model Evaluation: Evaluation Metrics

The foundation of all metrics is the **confusion matrix**, which compares the true labels with the predicted labels. The following metrics are commonly used for evaluating classification models:

## Confusion Matrix
The confusion matrix provides a detailed breakdown of the classification outcomes:

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **True Positive (TP)**  | Correctly predicted as positive | - |
| **False Positive (FP)** | Incorrectly predicted as positive | - |
| **True Negative (TN)**  | Correctly predicted as negative | - |
| **False Negative (FN)** | Incorrectly predicted as negative | - |

---

## Metrics

### **1. Precision**
- **Definition:** Measures how many of the objects predicted as positive are truly positive.
- **Formula:** `Precision = TP / (TP + FP)`

### **2. Recall (Sensitivity or True Positive Rate)**
- **Definition:** Measures how many of the actual positive objects are correctly predicted.
- **Formula:** `Recall = TP / (TP + FN)`

### **3. Accuracy**
- **Definition:** Measures the overall correctness of the predictions.
- **Formula:** `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

### **4. Error Rate**
- **Definition:** Measures the proportion of incorrect predictions.
- **Formula:** `Error Rate = (FP + FN) / (TP + TN + FP + FN)`

### **5. True Positive Rate (TPR)**
- **Definition:** Measures the proportion of true positives out of all actual positives.
- **Formula:** `TPR = TP / (TP + FN)`

### **6. False Positive Rate (FPR)**
- **Definition:** Measures the proportion of false positives out of all actual negatives.
- **Formula:** `FPR = FP / (FP + TN)`

---

## Additional Metrics

### **7. ROC Curve (Receiver Operating Characteristic Curve)**
- The ROC curve is a plot of the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various threshold levels.
- **AUC (Area Under the Curve):** The area under the ROC curve provides a single value to evaluate the model. A perfect classifier has an AUC of 1.

### **8. F1 Score**
- **Definition:** The F1 score is the harmonic mean of Precision and Recall, balancing the trade-off between the two.
- **Formula:** `F1 Score = 2 * (Precision * Recall) / (Precision + Recall)`

---

## Notes
- The choice of metrics depends on the problem domain. For example, Precision is critical in tasks like medical diagnosis to minimize false positives, while Recall is important in detecting fraudulent activities where missing positives is costly.
- It is recommended to combine metrics like Precision, Recall, and F1 Score for a holistic evaluation of model performance.

# SVM Model Evaluation and Analysis

The following steps evaluate and analyze the performance of each SVM model across the five strategies (\`bin_maj\`, \`bin_one\`, \`bin_all\`, \`multi_maj\`, \`disagree_bin\`).

```r
# Define a function for SVM model evaluation
evaluate_svm_model <- function(conf_matrix, model, predictions, test_labels) {
  # Print the confusion matrix and model details
  print(conf_matrix)
  print(model)
  
  # Compute and print the evaluation metrics
  conf_matrix <- confusionMatrix(predictions, test_labels)
  print(conf_matrix)
}

# Model 1: bin_maj
evaluate_svm_model(conf_matrix_svm_1, svm_model_1, predictions_svm_1, test_labels_svm_sub_1)

# Model 2: bin_one
evaluate_svm_model(conf_matrix_svm_2, svm_model_2, predictions_svm_2, test_labels_svm_sub_2)

# Model 3: bin_all
evaluate_svm_model(conf_matrix_svm_3, svm_model_3, predictions_svm_3, test_labels_svm_sub_3)

# Model 4: multi_maj
evaluate_svm_model(conf_matrix_svm_4, svm_model_4, predictions_svm_4, test_labels_svm_sub_4)

# Model 5: disagree_bin
evaluate_svm_model(conf_matrix_svm_5, svm_model_5, predictions_svm_5, test_labels_svm_sub_5)
```

# Introduction to Algorithms

## **Random Forest**

**Random Forest** is a powerful and versatile ensemble learning algorithm based on decision trees. It combines the predictions of multiple decision trees to create robust and generalizable models. The method leverages two key concepts:

- **Bagging (Bootstrap Aggregation):** Each decision tree is trained on a random subset of the dataset, reducing variance and preventing overfitting.
- **Feature Randomization:** At each node, a random subset of features is selected to increase diversity among the trees.

### **Strengths of Random Forest:**
- Resilient to outliers and non-linear relationships.
- Automatically computes feature importance.
- Performs well on large and complex datasets.

### **Use Case**
In our analysis, Random Forest is used to classify text data (represented by TF-IDF features) based on different strategies such as `bin_maj`, `bin_one`, etc.


# Training the Random Forest Model (RF) for Strategy 1 ("bin_maj")

The following steps train a Random Forest model using the TF-IDF feature matrix and evaluate its performance for the \`bin_maj\` strategy.

```r
# Model 1 for bin_maj
set.seed(1234) # Set seed for reproducible results

# Prepare the data for training
train_data <- tfidf_train_matrix_scaled # Use the scaled TF-IDF matrix as features
train_labels_1 <- train_prep_final$bin_maj # Classification labels for strategy 1

# Train the Random Forest model
# The model uses 500 trees (ntree = 500) for classification
rf_model_1 <- randomForest(x = train_data, y = train_labels_1, ntree = 500)

# Evaluate the model
# Split the data into training and testing datasets
train_indices_1 <- createDataPartition(train_prep_final$bin_maj, p = 0.8, list = FALSE)
train_data_sub_1 <- tfidf_train_matrix_scaled[train_indices_1, ] # Training data
test_data_sub_1 <- tfidf_train_matrix_scaled[-train_indices_1, ] # Testing data
train_labels_sub_1 <- train_labels_1[train_indices_1]            # Training labels
test_labels_sub_1 <- train_labels_1[-train_indices_1]            # Testing labels

# Train the Random Forest model again on the training subset
rf_model_1 <- randomForest(x = train_data_sub_1, y = train_labels_sub_1, ntree = 500)

# Predictions and evaluation
# Predict the labels for the testing data
predictions_1 <- predict(rf_model_1, test_data_sub_1)

# Generate a confusion matrix to evaluate the predictions
conf_matrix_1 <- table(Predicted = predictions_1, Actual = test_labels_sub_1)
```
# Training the Random Forest Model (RF) for Strategy 2 ("bin_one")

The following steps train a Random Forest model using the TF-IDF feature matrix and evaluate its performance for the `bin_one` strategy.

```r
# Model 2 for bin_one
set.seed(1234) # Set seed for reproducible results

# Prepare the data for training
# Use the scaled TF-IDF matrix as features
train_labels_2 <- train_prep_final$bin_one # Classification labels for strategy 2

# Train the Random Forest model
# The model uses 500 trees (ntree = 500) for classification
rf_model_2 <- randomForest(x = train_data, y = train_labels_2, ntree = 500)

# Evaluate the model
# Split the data into training and testing datasets
train_indices_2 <- createDataPartition(train_prep_final$bin_one, p = 0.8, list = FALSE)
train_data_sub_2 <- tfidf_train_matrix_scaled[train_indices_2, ] # Training data
test_data_sub_2 <- tfidf_train_matrix_scaled[-train_indices_2, ] # Testing data
train_labels_sub_2 <- train_labels_2[train_indices_2]            # Training labels
test_labels_sub_2 <- train_labels_2[-train_indices_2]            # Testing labels

# Train the Random Forest model again on the training subset
rf_model_2 <- randomForest(x = train_data_sub_2, y = train_labels_sub_2, ntree = 500)

# Predictions and evaluation
# Predict the labels for the testing data
predictions_2 <- predict(rf_model_2, test_data_sub_2)

# Generate a confusion matrix to evaluate the predictions
conf_matrix_2 <- table(Predicted = predictions_2, Actual = test_labels_sub_2)
```

# Training the Random Forest Model (RF) for Strategy 3 ("bin_all")

The following steps train a Random Forest model using the TF-IDF feature matrix and evaluate its performance for the \`bin_all\` strategy.

```r
# Model 3 for bin_all
set.seed(1234) # Set seed for reproducible results

# Prepare the data for training
# Use the scaled TF-IDF matrix as features
train_labels_3 <- train_prep_final$bin_all # Classification labels for strategy 3

# Train the Random Forest model
# The model uses 500 trees (ntree = 500) for classification
rf_model_3 <- randomForest(x = train_data, y = train_labels_3, ntree = 500)

# Evaluate the model
# Split the data into training and testing datasets
train_indices_3 <- createDataPartition(train_prep_final$bin_all, p = 0.8, list = FALSE)
train_data_sub_3 <- tfidf_train_matrix_scaled[train_indices_3, ] # Training data
test_data_sub_3 <- tfidf_train_matrix_scaled[-train_indices_3, ] # Testing data
train_labels_sub_3 <- train_labels_3[train_indices_3]            # Training labels
test_labels_sub_3 <- train_labels_3[-train_indices_3]            # Testing labels

# Train the Random Forest model again on the training subset
rf_model_3 <- randomForest(x = train_data_sub_3, y = train_labels_sub_3, ntree = 500)

# Predictions and evaluation
# Predict the labels for the testing data
predictions_3 <- predict(rf_model_3, test_data_sub_3)

# Generate a confusion matrix to evaluate the predictions
conf_matrix_3 <- table(Predicted = predictions_3, Actual = test_labels_sub_3)
```

# Training the Random Forest Model (RF) for Strategy 4 ("multi_maj")

The following steps train a Random Forest model using the TF-IDF feature matrix and evaluate its performance for the `multi_maj` strategy.

```r
# Model 4 for multi_maj
set.seed(1234) # Set seed for reproducible results

# Prepare the data for training
# Use the scaled TF-IDF matrix as features
train_labels_4 <- train_prep_final$multi_maj # Classification labels for strategy 4

# Train the Random Forest model
# The model uses 500 trees (ntree = 500) for classification
rf_model_4 <- randomForest(x = train_data, y = train_labels_4, ntree = 500)

# Evaluate the model
# Split the data into training and testing datasets
train_indices_4 <- createDataPartition(train_prep_final$multi_maj, p = 0.8, list = FALSE)
train_data_sub_4 <- tfidf_train_matrix_scaled[train_indices_4, ] # Training data
test_data_sub_4 <- tfidf_train_matrix_scaled[-train_indices_4, ] # Testing data
train_labels_sub_4 <- train_labels_4[train_indices_4]            # Training labels
test_labels_sub_4 <- train_labels_4[-train_indices_4]            # Testing labels

# Train the Random Forest model again on the training subset
rf_model_4 <- randomForest(x = train_data_sub_4, y = train_labels_sub_4, ntree = 500)

# Predictions and evaluation
# Predict the labels for the testing data
predictions_4 <- predict(rf_model_4, test_data_sub_4)

# Generate a confusion matrix to evaluate the predictions
conf_matrix_4 <- table(Predicted = predictions_4, Actual = test_labels_sub_4)
```

# Training the Random Forest Model (RF) for Strategy 5 ("disagree_bin")

The following steps train a Random Forest model using the TF-IDF feature matrix and evaluate its performance for the \`disagree_bin\` strategy.

```r
# Model 5 for disagree_bin
set.seed(1234) # Set seed for reproducible results

# Prepare the data for training
# Use the scaled TF-IDF matrix as features
train_labels_5 <- train_prep_final$disagree_bin # Classification labels for strategy 5

# Train the Random Forest model
# The model uses 500 trees (ntree = 500) for classification
rf_model_5 <- randomForest(x = train_data, y = train_labels_5, ntree = 500)

# Evaluate the model
# Split the data into training and testing datasets
train_indices_5 <- createDataPartition(train_prep_final$disagree_bin, p = 0.8, list = FALSE)
train_data_sub_5 <- tfidf_train_matrix_scaled[train_indices_5, ] # Training data
test_data_sub_5 <- tfidf_train_matrix_scaled[-train_indices_5, ] # Testing data
train_labels_sub_5 <- train_labels_5[train_indices_5]            # Training labels
test_labels_sub_5 <- train_labels_5[-train_indices_5]            # Testing labels

# Train the Random Forest model again on the training subset
rf_model_5 <- randomForest(x = train_data_sub_5, y = train_labels_sub_5, ntree = 500)

# Predictions and evaluation
# Predict the labels for the testing data
predictions_5 <- predict(rf_model_5, test_data_sub_5)

# Generate a confusion matrix to evaluate the predictions
conf_matrix_5 <- table(Predicted = predictions_5, Actual = test_labels_sub_5)
```

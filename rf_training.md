## **Random Forest**

**Random Forest** is a powerful and versatile ensemble learning algorithm based on decision trees. It combines the predictions of multiple decision trees to create robust and generalizable models. The method leverages two key concepts:

- **Bagging (Bootstrap Aggregation):** Each decision tree is trained on a random subset of the dataset, reducing variance and preventing overfitting.
- **Feature Randomization:** At each node, a random subset of features is selected to increase diversity among the trees.

### **Strengths of Random Forest:**
- Resilient to outliers and non-linear relationships.
- Automatically computes feature importance.
- Performs well on large and complex datasets.

### **Use Case**
In our analysis, Random Forest is used to classify text data (represented by TF-IDF features) based on different strategies such as \`bin_maj\`, \`bin_one\`, etc.


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
- **Formula:** \`Precision = TP / (TP + FP)\`

### **2. Recall (Sensitivity or True Positive Rate)**
- **Definition:** Measures how many of the actual positive objects are correctly predicted.
- **Formula:** \`Recall = TP / (TP + FN)\`

### **3. Accuracy**
- **Definition:** Measures the overall correctness of the predictions.
- **Formula:** \`Accuracy = (TP + TN) / (TP + TN + FP + FN)\`

### **4. Error Rate**
- **Definition:** Measures the proportion of incorrect predictions.
- **Formula:** \`Error Rate = (FP + FN) / (TP + TN + FP + FN)\`

### **5. True Positive Rate (TPR)**
- **Definition:** Measures the proportion of true positives out of all actual positives.
- **Formula:** \`TPR = TP / (TP + FN)\`

### **6. False Positive Rate (FPR)**
- **Definition:** Measures the proportion of false positives out of all actual negatives.
- **Formula:** \`FPR = FP / (FP + TN)\`

---

## Additional Metrics

### **7. ROC Curve (Receiver Operating Characteristic Curve)**
- The ROC curve is a plot of the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various threshold levels.
- **AUC (Area Under the Curve):** The area under the ROC curve provides a single value to evaluate the model. A perfect classifier has an AUC of 1.

### **8. F1 Score**
- **Definition:** The F1 score is the harmonic mean of Precision and Recall, balancing the trade-off between the two.
- **Formula:** \`F1 Score = 2 * (Precision * Recall) / (Precision + Recall)\`

---

## Notes
- The choice of metrics depends on the problem domain. For example, Precision is critical in tasks like medical diagnosis to minimize false positives, while Recall is important in detecting fraudulent activities where missing positives is costly.
- It is recommended to combine metrics like Precision, Recall, and F1 Score for a holistic evaluation of model performance.

# Random Forest Model Evaluation and Analysis

The following steps evaluate and analyze the performance of each Random Forest model across the five strategies (\`bin_maj\`, \`bin_one\`, \`bin_all\`, \`multi_maj\`, \`disagree_bin\`). Additionally, feature importance is computed and visualized for each model.

```r
# Define a function for Random Forest model evaluation
evaluate_rf_model <- function(conf_matrix, model) {
  # Print the confusion matrix and model details
  print(conf_matrix)
  print(model)
  
  # Display feature importance
  importance(model)
  
  # Plot feature importance
  varImpPlot(model)
}

# Model 1: bin_maj
evaluate_rf_model(conf_matrix_1, rf_model_1)

# Model 2: bin_one
evaluate_rf_model(conf_matrix_2, rf_model_2)

# Model 3: bin_all
evaluate_rf_model(conf_matrix_3, rf_model_3)

# Model 4: multi_maj
evaluate_rf_model(conf_matrix_4, rf_model_4)

# Model 5: disagree_bin
evaluate_rf_model(conf_matrix_5, rf_model_5)
```

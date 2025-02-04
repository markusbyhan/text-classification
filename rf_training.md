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
# RF Strategy 1 with 5-fold Cross Validation

# Set the seed for reproducible results
set.seed(1234)

# Prepare the training data and target labels.
# Note: 'final_traindata_matrix' contains the training features.
# Initially, target_labels is assigned from df_knn_final$bin_maj, then overwritten.
target_labels <- df_knn_final$bin_maj  # First assignment
tfidf_matrix <- trainingsdata         # TF-IDF matrix (not used later)
target_labels <- competition_final$bin_maj  # Overwrites the previous target_labels

# Split data into training and testing sets (80% train, 20% test)
train_indices <- createDataPartition(target_labels, p = 0.8, list = FALSE)
train_data <- final_traindata_matrix[train_indices, ]
test_data <- final_traindata_matrix[-train_indices, ]
train_labels <- target_labels[train_indices]
test_labels <- target_labels[-train_indices]

# Define training control using 5-fold cross validation
train_control <- trainControl(method = "cv", number = 5)

# Train a Random Forest model using 5-fold cross validation.
# - method = "rf": Specifies the Random Forest algorithm.
# - tuneLength = 10: Tests 10 different tuning parameter values.
# - scale = FALSE: No internal scaling is applied.
rf_model_1 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "rf",
  trControl = train_control,
  tuneLength = 10,
  scale = FALSE
)

# Make predictions on the test set.
rf_predictions_1 <- predict(rf_model_1, test_data)

# Create and display the confusion matrix.
conf_matrix_rf_1 <- confusionMatrix(rf_predictions_1, test_labels)
cat("Confusion Matrix:\n")
conf_matrix_rf_1 <- as.matrix(conf_matrix_rf_1$table)
print(conf_matrix_rf_1)

# Calculate precision and recall for each class.
precision <- diag(conf_matrix_rf_1) / rowSums(conf_matrix_rf_1)
recall <- diag(conf_matrix_rf_1) / colSums(conf_matrix_rf_1)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate the F1-score for each class and the macro F1-score.
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Display evaluation metrics.
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Analyze feature correlations.
correlationMatrix <- cor(final_traindata_matrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate variable importance from the RF model.
importance <- varImp(rf_model_1, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```
# Training the Random Forest Model (RF) for Strategy 2 ("bin_one")

The following steps train a Random Forest model using the TF-IDF feature matrix and evaluate its performance for the `bin_one` strategy.

```r
# RF Strategy 2 with 5-fold Cross Validation

# Set the seed for reproducible results
set.seed(1234)

# Prepare the training data and target labels.
# 'final_traindata_matrix' contains the training features.
# Here, target_labels are defined using the 'bin_one' column.
target_labels <- df_knn_final$bin_one

# Split data into training (80%) and testing (20%) sets.
train_indices <- createDataPartition(target_labels, p = 0.8, list = FALSE)
train_data <- final_traindata_matrix[train_indices, ]
test_data <- final_traindata_matrix[-train_indices, ]
train_labels <- target_labels[train_indices]
test_labels <- target_labels[-train_indices]

# Define training control using 5-fold cross validation.
train_control <- trainControl(method = "cv", number = 5)

# Train a Random Forest model using 5-fold cross validation.
rf_model_2 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "rf",
  trControl = train_control,
  tuneLength = 10,  # Number of candidate tuning parameters.
  scale = FALSE
)

# Make predictions on the test set.
rf_predictions_2 <- predict(rf_model_2, test_data)

# Create and display the confusion matrix.
conf_matrix_rf_2 <- confusionMatrix(rf_predictions_2, test_labels)
cat("Confusion Matrix:\n")
conf_matrix_rf_2 <- as.matrix(conf_matrix_rf_2$table)
print(conf_matrix_rf_2)

# Calculate precision and recall for each class.
precision <- diag(conf_matrix_rf_2) / rowSums(conf_matrix_rf_2)
recall <- diag(conf_matrix_rf_2) / colSums(conf_matrix_rf_2)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate F1-scores and the Macro F1-Score.
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Display evaluation metrics.
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Analyze feature correlations.
correlationMatrix <- cor(final_traindata_matrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate variable importance from the RF model.
importance <- varImp(rf_model_2, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# Training the Random Forest Model (RF) for Strategy 3 ("bin_all")

The following steps train a Random Forest model using the TF-IDF feature matrix and evaluate its performance for the \`bin_all\` strategy.

```r
# RF Strategy 3 with 5-fold Cross Validation

# Set the seed for reproducible results
set.seed(1234)

# Prepare the training data and target labels.
# Here, target_labels are defined using the 'bin_all' column.
target_labels <- df_knn_final$bin_all

# Split the data into training (80%) and testing (20%) sets.
train_indices <- createDataPartition(target_labels, p = 0.8, list = FALSE)
train_data <- final_traindata_matrix[train_indices, ]
test_data <- final_traindata_matrix[-train_indices, ]
train_labels <- target_labels[train_indices]
test_labels <- target_labels[-train_indices]

# Define training control using 5-fold cross validation.
train_control <- trainControl(method = "cv", number = 5)

# Train a Random Forest model using 5-fold cross validation.
rf_model_3 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "rf",
  trControl = train_control,
  tuneLength = 10,
  scale = FALSE
)

# Make predictions on the test set.
rf_predictions_3 <- predict(rf_model_3, test_data)

# Create and display the confusion matrix.
conf_matrix_rf_3 <- confusionMatrix(rf_predictions_3, test_labels)
cat("Confusion Matrix:\n")
conf_matrix_rf_3 <- as.matrix(conf_matrix_rf_3$table)
print(conf_matrix_rf_3)

# Calculate precision and recall for each class.
precision <- diag(conf_matrix_rf_3) / rowSums(conf_matrix_rf_3)
recall <- diag(conf_matrix_rf_3) / colSums(conf_matrix_rf_3)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate F1-scores and the Macro F1-Score.
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Display evaluation metrics.
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Analyze feature correlations.
correlationMatrix <- cor(final_traindata_matrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate variable importance from the RF model.
importance <- varImp(rf_model_3, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# Training the Random Forest Model (RF) for Strategy 4 ("multi_maj")

The following steps train a Random Forest model using the TF-IDF feature matrix and evaluate its performance for the `multi_maj` strategy.

```r
# RF Strategy 4 with 5-fold Cross Validation

# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels.
# Here, target_labels are defined using the 'multi_maj' column.
target_labels <- df_knn_final$multi_maj

# Split the data into training (80%) and testing (20%) sets.
train_indices <- createDataPartition(target_labels, p = 0.8, list = FALSE)
train_data <- final_traindata_matrix[train_indices, ]
test_data <- final_traindata_matrix[-train_indices, ]
train_labels <- target_labels[train_indices]
test_labels <- target_labels[-train_indices]

# Define training control using 5-fold cross validation.
train_control <- trainControl(method = "cv", number = 5)

# Train a Random Forest model using 5-fold cross validation.
rf_model_4 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "rf",
  trControl = train_control,
  tuneLength = 10,
  scale = FALSE
)

# Make predictions on the test set.
rf_predictions_4 <- predict(rf_model_4, test_data)

# Convert predictions and true labels to factors with levels 0 to 4.
rf_predictions_4 <- factor(rf_predictions_4, levels = 0:4)
test_labels_f <- factor(test_labels, levels = 0:4)

# Create and display the confusion matrix.
conf_matrix_rf_4 <- confusionMatrix(rf_predictions_4, test_labels_f)
cat("Confusion Matrix:\n")
print(conf_matrix_rf_4$table)

# Calculate precision and recall for each class.
conf_matrix_rf_4_table <- as.matrix(conf_matrix_rf_4$table)
precision <- diag(conf_matrix_rf_4_table) / rowSums(conf_matrix_rf_4_table)
recall <- diag(conf_matrix_rf_4_table) / colSums(conf_matrix_rf_4_table)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate F1-scores and the Macro F1-Score.
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Display evaluation metrics.
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Analyze feature correlations.
correlationMatrix <- cor(final_traindata_matrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate variable importance from the RF model.
importance <- varImp(rf_model_4, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# Training the Random Forest Model (RF) for Strategy 5 ("disagree_bin")

The following steps train a Random Forest model using the TF-IDF feature matrix and evaluate its performance for the \`disagree_bin\` strategy.

```r
# RF Strategy 5 with 5-fold Cross Validation

# Set the seed for reproducible results
set.seed(1234)

# Prepare the training data and target labels.
# Here, target_labels are defined using the 'disagree_bin' column.
target_labels <- df_knn_final$disagree_bin

# Split the data into training (80%) and testing (20%) sets.
train_indices <- createDataPartition(target_labels, p = 0.8, list = FALSE)
train_data <- final_traindata_matrix[train_indices, ]
test_data <- final_traindata_matrix[-train_indices, ]
train_labels <- target_labels[train_indices]
test_labels <- target_labels[-train_indices]

# Define training control using 5-fold cross validation.
train_control <- trainControl(method = "cv", number = 5)

# Train a Random Forest model using 5-fold cross validation.
rf_model_5 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "rf",
  trControl = train_control,
  tuneLength = 10,
  scale = FALSE
)

# Make predictions on the test set.
rf_predictions_5 <- predict(rf_model_5, test_data)

# Create and display the confusion matrix.
conf_matrix_rf_5 <- confusionMatrix(rf_predictions_5, test_labels)
cat("Confusion Matrix:\n")
conf_matrix_rf_5 <- as.matrix(conf_matrix_rf_5$table)
print(conf_matrix_rf_5)

# Calculate precision and recall for each class.
precision <- diag(conf_matrix_rf_5) / rowSums(conf_matrix_rf_5)
recall <- diag(conf_matrix_rf_5) / colSums(conf_matrix_rf_5)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate F1-scores and the Macro F1-Score.
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Display evaluation metrics.
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Analyze feature correlations.
correlationMatrix <- cor(final_traindata_matrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate variable importance from the RF model.
importance <- varImp(rf_model_5, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
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

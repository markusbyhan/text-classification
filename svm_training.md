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
# SVM Strategy 1 with 10-fold Cross Validation

# Set the seed for reproducible results
set.seed(1234)

# Prepare training data and target labels
# - final_traindata_matrix contains the training features (assumed preprocessed and scaled)
# - target_labels contains the class labels from df_knn_final$bin_maj
target_labels <- df_knn_final$bin_maj

# Define training control for 10-fold cross validation using caret
# - method = "cv" specifies cross-validation
# - number = 10 specifies the number of folds (10-fold CV)
train_control <- trainControl(method = "cv", number = 10)

# Train an SVM model with a linear kernel using 10-fold cross validation
# SVM parameters explained:
# - method = "svmLinear": This uses a support vector machine with a linear kernel.
#   The linear kernel is useful when the data is linearly separable or when interpretability is important.
# - trControl = train_control: Specifies the resampling method (here, 10-fold CV) for performance estimation.
# - tuneLength = 10: Instructs caret to try 10 different candidate values for the tuning parameter(s).
#   For svmLinear, the primary tuning parameter is usually the cost (C) parameter, which controls the penalty for misclassification.
# - scale = FALSE: Indicates that the model should not apply its own scaling since the data is assumed to be pre-scaled.
svm_model_1 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "svmLinear",
  trControl = train_control,
  tuneLength = 10,
  scale = FALSE
)

# Print the cross-validation results
print(svm_model_1)

# Optionally, display the best tuning parameters:
# best_model <- svm_model_1$bestTune
# cat("Best Model:\n")
# print(best_model)

# Display full results from cross-validation
print(svm_model_1$results)

# Make predictions on the training data (or test data if available)
svm_predictions_1 <- predict(svm_model_1, final_traindata_matrix)

# Create and print the confusion matrix
conf_matrix_1 <- confusionMatrix(svm_predictions_1, target_labels)
cat("Confusion Matrix:\n")
# Convert the confusion matrix to a matrix (if necessary)
conf_matrix_1 <- as.matrix(conf_matrix_1$table)
print(conf_matrix_1)

# Calculate precision and recall for each class
precision <- diag(conf_matrix_1) / rowSums(conf_matrix_1)
recall    <- diag(conf_matrix_1) / colSums(conf_matrix_1)

# Replace any NaN values with 0
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate the F1-score for each class
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0

# Calculate the Macro-F1 score (average F1-score across classes)
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Display the evaluation metrics
cat("F1-scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# ---------------------------
# Variable Importance and Correlation Analysis
# ---------------------------

# Calculate the correlation matrix of the features (assuming competition_tf_sa is defined)
correlationMatrix <- cor(competition_tf_sa)
# Find indices of highly correlated attributes (cutoff set to 0.5)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate variable importance from the SVM model using caret's varImp function
importance <- varImp(svm_model_1, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
# Plot the variable importance
plot(importance)
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
# SVM Strategy 2 with 10-fold Cross Validation

# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels for Strategy 2.
# - final_traindata_matrix contains the training features.
# - target_labels are defined by the 'bin_one' column from df_knn_final.
target_labels <- df_knn_final$bin_one

# Define training control using 10-fold cross validation.
train_control <- trainControl(method = "cv", number = 10)

# Train an SVM model with a linear kernel.
# Explanation of parameters:
# - method = "svmLinear": Uses a support vector machine with a linear kernel.
# - tuneLength = 10: Searches over 10 candidate values for the cost parameter (C).
# - scale = FALSE: Assumes the data is already scaled.
svm_model_2 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "svmLinear",
  trControl = train_control,
  tuneLength = 10,
  scale = FALSE
)

# Display the cross-validation results.
print(svm_model_2)
print(svm_model_2$results)

# Make predictions on the training data.
svm_predictions_2 <- predict(svm_model_2, final_traindata_matrix)

# Create and display the confusion matrix.
conf_matrix_2 <- confusionMatrix(svm_predictions_2, target_labels)
cat("Confusion Matrix:\n")
conf_matrix_2 <- as.matrix(conf_matrix_2$table)
print(conf_matrix_2)

# Calculate precision and recall for each class.
precision <- diag(conf_matrix_2) / rowSums(conf_matrix_2)
recall <- diag(conf_matrix_2) / colSums(conf_matrix_2)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate the F1-score for each class and the macro F1-score.
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Print evaluation metrics.
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Perform correlation analysis on the features (assuming 'competition_tf_sa' is defined).
correlationMatrix <- cor(competition_tf_sa)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate variable importance from the SVM model and plot it.
importance <- varImp(svm_model_2, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# Training the Support Vector Machine (SVM) for Strategy 3 ("bin_all")

The following steps train an SVM model using the TF-IDF feature matrix and evaluate its performance for the \`bin_all\` strategy.

```r
# SVM Strategy 3 with 10-fold Cross Validation

# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels for Strategy 3.
# Here, the target labels come from the 'bin_all' column of df_knn_final.
target_labels <- df_knn_final$bin_all

# Define training control with 10-fold cross validation.
train_control <- trainControl(method = "cv", number = 10)

# Train an SVM model with a linear kernel.
svm_model_3 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "svmLinear",
  trControl = train_control,
  tuneLength = 10,
  scale = FALSE
)

# Display cross-validation results.
print(svm_model_3)

# Make predictions on the training data.
svm_predictions_3 <- predict(svm_model_3, final_traindata_matrix)

# Create and display the confusion matrix.
conf_matrix_3 <- confusionMatrix(svm_predictions_3, target_labels)
cat("Confusion Matrix:\n")
conf_matrix_3 <- as.matrix(conf_matrix_3$table)
print(conf_matrix_3)

# Calculate precision and recall for each class.
precision <- diag(conf_matrix_3) / rowSums(conf_matrix_3)
recall <- diag(conf_matrix_3) / colSums(conf_matrix_3)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate F1-scores and the Macro F1-Score.
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Print evaluation metrics.
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Perform correlation analysis on the training features.
correlationMatrix <- cor(final_traindata_matrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate and plot variable importance.
importance <- varImp(svm_model_3, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# Training the Support Vector Machine (SVM) for Strategy 4 ("multi_maj")

The following steps train an SVM model using the TF-IDF feature matrix and evaluate its performance for the `multi_maj` strategy.

```r
# SVM Strategy 4 with 10-fold Cross Validation

# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels for Strategy 4.
# Target labels are derived from the 'multi_maj' column in df_knn_final.
target_labels <- df_knn_final$multi_maj

# Define training control with 10-fold cross validation.
train_control <- trainControl(method = "cv", number = 10)

# Train an SVM model with a linear kernel.
svm_model_4 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "svmLinear",
  trControl = train_control,
  tuneLength = 10,
  scale = FALSE
)

# Display cross-validation results.
print(svm_model_4)

# Make predictions on the training data.
svm_predictions_4 <- predict(svm_model_4, final_traindata_matrix)

# Convert predictions and target labels to factors with levels 0 to 4.
svm_predictions_4 <- factor(svm_predictions_4, levels = 0:4)
target_labels <- factor(target_labels, levels = 0:4)

# Create and display the confusion matrix.
conf_matrix_4 <- confusionMatrix(svm_predictions_4, target_labels)
cat("Confusion Matrix:\n")
conf_matrix_4 <- as.matrix(conf_matrix_4$table)
print(conf_matrix_4)

# Calculate precision and recall for each class.
precision <- diag(conf_matrix_4) / rowSums(conf_matrix_4)
recall <- diag(conf_matrix_4) / colSums(conf_matrix_4)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate F1-scores and the Macro F1-Score.
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Print evaluation metrics.
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Perform correlation analysis on the training features.
correlationMatrix <- cor(final_traindata_matrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate and plot variable importance.
importance <- varImp(svm_model_4, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# Training the Support Vector Machine (SVM) for Strategy 5 ("disagree_bin")

The following steps train an SVM model using the TF-IDF feature matrix and evaluate its performance for the \`disagree_bin\` strategy.

```r
# SVM Strategy 5 with 10-fold Cross Validation

# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels for Strategy 5.
# Target labels are taken from the 'disagree_bin' column of df_knn_final.
target_labels <- df_knn_final$disagree_bin

# Define training control with 10-fold cross validation.
train_control <- trainControl(method = "cv", number = 10)

# Train an SVM model with a linear kernel.
svm_model_5 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "svmLinear",
  trControl = train_control,
  tuneLength = 10,
  scale = FALSE
)

# Display cross-validation results.
print(svm_model_5)

# Make predictions on the training data.
svm_predictions_5 <- predict(svm_model_5, final_traindata_matrix)

# Create and display the confusion matrix.
conf_matrix_5 <- confusionMatrix(svm_predictions_5, target_labels)
cat("Confusion Matrix:\n")
conf_matrix_5 <- as.matrix(conf_matrix_5$table)
print(conf_matrix_5)

# Calculate precision and recall for each class.
precision <- diag(conf_matrix_5) / rowSums(conf_matrix_5)
recall <- diag(conf_matrix_5) / colSums(conf_matrix_5)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate F1-scores and the Macro F1-Score.
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Print evaluation metrics.
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Perform correlation analysis on the training features.
correlationMatrix <- cor(final_traindata_matrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate and plot variable importance.
importance <- varImp(svm_model_5, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# Model Evaluation: Evaluation Metrics

The foundation of all metrics is the **confusion matrix**, which compares the true labels with the predicted labels. The following metrics are commonly used for evaluating classification models:

## Confusion Matrix
The confusion matrix provides a detailed breakdown of the classification outcomes:

# Confusion Matrix

A confusion matrix is a table that is used to evaluate the performance of a classification model by comparing the actual class labels with the predicted class labels.

For a **binary classification** problem, the confusion matrix typically looks like this:

|                      | **Predicted: Negative** | **Predicted: Positive** |
|----------------------|-------------------------|-------------------------|
| **Actual: Negative** | True Negatives (TN)     | False Positives (FP)    |
| **Actual: Positive** | False Negatives (FN)    | True Positives (TP)     |

For a **multi-class classification** problem, the matrix is expanded as follows:

|                        | **Predicted: Class 1** | **Predicted: Class 2** | ... | **Predicted: Class N** |
|------------------------|------------------------|------------------------|-----|------------------------|
| **Actual: Class 1**    | a₁₁                   | a₁₂                   | ... | a₁ₙ                   |
| **Actual: Class 2**    | a₂₁                   | a₂₂                   | ... | a₂ₙ                   |
| ...                    | ...                    | ...                    | ... | ...                    |
| **Actual: Class N**    | aₙ₁                   | aₙ₂                   | ... | aₙₙ                   |

**Legend:**

- **aᵢⱼ**: The number of instances with actual class *i* that were predicted as class *j*.
- In a binary problem:
  - **TN (True Negatives):** Correctly predicted negatives.
  - **FP (False Positives):** Incorrectly predicted positives (Type I error).
  - **FN (False Negatives):** Incorrectly predicted negatives (Type II error).
  - **TP (True Positives):** Correctly predicted positives.
- In a multi-class problem, the diagonal entries (a₁₁, a₂₂, ..., aₙₙ) indicate the number of correct predictions for each class, while the off-diagonal entries indicate misclassifications.

*Note:* Replace "Class 1", "Class 2", ..., "Class N" with your specific class labels when applying this template.
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


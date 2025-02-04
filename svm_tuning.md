## Introduction: The Importance of Hyperparameter Tuning in SVM Models

In machine learning, **hyperparameter tuning** is essential for optimizing model performance. For Support Vector Machines (SVM), parameters such as the **cost parameter (C)** and **class weights** have a 
significant impact on the decision boundaries and overall predictive power. Without proper tuning, an SVM model may either overfit the training data (capturing noise) or underfit (failing to capture the 
underlying patterns), which leads to poor generalization on unseen data.
By using techniques like **10-fold cross validation**, we can systematically explore various hyperparameter values to identify the settings that best balance the trade-off between model complexity and 
accuracy. This process helps in improving key evaluation metrics such as **precision**, **recall**, and the **F1-score**.
The code blocks below demonstrate how to tune SVM models using the `svmLinearWeights` method, adjusting parameters like **C** and **weight** to optimize performance across different SVM strategies.

# SVM Tuning Strategy 1 with 10-fold Cross Validation

```r
# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels.
# 'final_traindata_matrix' contains the training features.
target_labels <- df_knn_final$bin_maj  # Classification column for strategy 1

# Define training control for 10-fold cross validation
train_control <- trainControl(method = "cv", number = 10)

# Train the SVM model using the 'svmLinearWeights' method.
# Parameters:
# - C: Cost parameter (set to 0.5)
# - weight: Class weight (set to 2)
# - scale: No scaling is applied (assumes data is pre-scaled)
svm_model_1 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "svmLinearWeights",
  trControl = train_control,
  C = 0.5,
  weight = 2,
  # tuneGrid = grid,  # Uncomment and define 'grid' if tuning multiple parameters
  scale = FALSE
)

# Display the best tuning parameters
best_model <- svm_model_1$bestTune
cat("Best Model:\n")
print(best_model)

# Display the cross-validation results
print(svm_model_1)
print(svm_model_1$results)

# Make predictions on the training data
svm_predictions_1 <- predict(svm_model_1, final_traindata_matrix)

# Create and display the confusion matrix
conf_matrix_1 <- confusionMatrix(svm_predictions_1, target_labels)
cat("Confusion Matrix:\n")
conf_matrix_1 <- as.matrix(conf_matrix_1)
print(conf_matrix_1)

# Calculate precision and recall for each class
precision <- diag(conf_matrix_1) / rowSums(conf_matrix_1)
recall <- diag(conf_matrix_1) / colSums(conf_matrix_1)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate the F1-score for each class and the Macro F1-Score (average)
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Display evaluation metrics
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Analyze feature correlations in the training data
correlationMatrix <- cor(final_traindata_matrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate and display variable importance
importance <- varImp(svm_model_1, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# SVM Tuning Strategy 2 with 10-fold Cross Validation

```r
# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels.
target_labels <- df_knn_final$bin_one  # Classification column for strategy 2

# Define training control for 10-fold cross validation
train_control <- trainControl(method = "cv", number = 10)

## Optionally, define a grid for parameter tuning (currently commented out)
# grid <- expand.grid(C = seq(0.01, 1, by = 0.01))

# Train the SVM model using the 'svmLinearWeights' method.
# Parameters:
# - C: Cost parameter set to 0.25
# - weight: Class weight set to 1
# - scale: Data is assumed to be pre-scaled
svm_model_2 <- train(
  x = final_traindata_matrix, 
  y = target_labels,
  method = "svmLinearWeights",
  trControl = train_control,
  C = 0.25,
  weight = 1,
  scale = FALSE
)

# Display the cross-validation results and best model tuning parameters
print(svm_model_2)
best_model <- svm_model_2$bestTune
cat("Best Model:\n")
print(best_model)
print(svm_model_2$results)

# Make predictions on the training data
svm_predictions_2 <- predict(svm_model_2, final_traindata_matrix)

# Create and display the confusion matrix
conf_matrix_2 <- confusionMatrix(svm_predictions_2, target_labels)
cat("Confusion Matrix:\n")
conf_matrix_2 <- as.matrix(conf_matrix_2)
print(conf_matrix_2)

# Calculate precision and recall for each class
precision <- diag(conf_matrix_2) / rowSums(conf_matrix_2)
recall <- diag(conf_matrix_2) / colSums(conf_matrix_2)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate F1-scores and Macro F1-Score
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Display evaluation metrics
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Analyze feature correlations in the training data
correlationMatrix <- cor(final_traindata_matrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate and display variable importance
importance <- varImp(svm_model_2, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# SVM Tuning Strategy 3 with 10-fold Cross Validation

```r
# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels.
# 'final_traindata_matrix' is the TF-IDF matrix.
target_labels <- df_knn_final$bin_all  # Classification column for strategy 3

# Define training control for 10-fold cross validation
train_control <- trainControl(method = "cv", number = 10)

# Train the SVM model using the 'svmLinearWeights' method.
# Parameters:
# - C: Cost parameter set to 0.25
# - weight: Class weight set to 1
# - scale: No scaling is applied
svm_model_3 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "svmLinearWeights",
  trControl = train_control,
  C = 0.25,
  weight = 1,
  scale = FALSE
)

# Display the cross-validation results and best tuning parameters
print(svm_model_3)
best_model <- svm_model_3$bestTune
cat("Best Model:\n")
print(best_model)

# Make predictions on the training data
svm_predictions_3 <- predict(svm_model_3, final_traindata_matrix)

# Create and display the confusion matrix
conf_matrix_3 <- confusionMatrix(svm_predictions_3, target_labels)
cat("Confusion Matrix:\n")
conf_matrix_3 <- as.matrix(conf_matrix_3)
print(conf_matrix_3)

# Calculate precision and recall for each class
precision <- diag(conf_matrix_3) / rowSums(conf_matrix_3)
recall <- diag(conf_matrix_3) / colSums(conf_matrix_3)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate F1-scores and Macro F1-Score
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Display evaluation metrics
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Analyze feature correlations
correlationMatrix <- cor(final_traindata_matrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate and display variable importance
importance <- varImp(svm_model_3, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# SVM Tuning Strategy 4 with 10-fold Cross Validation

```r
# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels.
# 'final_traindata_matrix' is the TF-IDF matrix.
target_labels <- df_knn_final$multi_maj  # Classification column for strategy 4

# Define training control for 10-fold cross validation
train_control <- trainControl(method = "cv", number = 10)

# Train the SVM model using the 'svmLinearWeights' method.
# Parameters:
# - C: Cost parameter set to 0.5
# - weight: Class weight set to 2
# - scale: Data is assumed to be pre-scaled
svm_model_4 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "svmLinearWeights",
  trControl = train_control,
  C = 0.5,
  weight = 2,
  scale = FALSE
)

# Display the cross-validation results and best tuning parameters
print(svm_model_4)
best_model <- svm_model_4$bestTune
cat("Best Model:\n")
print(best_model)

# Make predictions on the training data
svm_predictions_4 <- predict(svm_model_4, final_traindata_matrix)

# Convert predictions and target labels to factors with levels 0 to 4
svm_predictions_4 <- factor(svm_predictions_4, levels = 0:4)
target_labels <- factor(target_labels, levels = 0:4)

# Create and display the confusion matrix
conf_matrix_4 <- confusionMatrix(svm_predictions_4, target_labels)
cat("Confusion Matrix:\n")
conf_matrix_4 <- as.matrix(conf_matrix_4)
print(conf_matrix_4)

# Calculate precision and recall for each class
precision <- diag(conf_matrix_4) / rowSums(conf_matrix_4)
recall <- diag(conf_matrix_4) / colSums(conf_matrix_4)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate F1-scores and Macro F1-Score
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Display evaluation metrics
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Analyze feature correlations
correlationMatrix <- cor(final_traindata_matrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate and display variable importance
importance <- varImp(svm_model_4, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# SVM Tuning Strategy 5 with 10-fold Cross Validation

```r
# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels.
# 'final_traindata_matrix' is the TF-IDF matrix.
target_labels <- df_knn_final$disagree_bin  # Classification column for strategy 5

# Define training control for 10-fold cross validation
train_control <- trainControl(method = "cv", number = 10)

# Train the SVM model using the 'svmLinearWeights' method.
# Parameters:
# - C: Cost parameter set to 1
# - weight: Class weight set to 1
# - scale: No scaling applied
svm_model_5 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "svmLinearWeights",
  trControl = train_control,
  C = 1,
  weight = 1,
  scale = FALSE
)

# Display the cross-validation results and best tuning parameters
print(svm_model_5)
best_model <- svm_model_5$bestTune
cat("Best Model:\n")
print(best_model)

# Make predictions on the training data
svm_predictions_5 <- predict(svm_model_5, final_traindata_matrix)

# Create and display the confusion matrix
conf_matrix_5 <- confusionMatrix(svm_predictions_5, target_labels)
cat("Confusion Matrix:\n")
conf_matrix_5 <- as.matrix(conf_matrix_5)
print(conf_matrix_5)

# Calculate precision and recall for each class
precision <- diag(conf_matrix_5) / rowSums(conf_matrix_5)
recall <- diag(conf_matrix_5) / colSums(conf_matrix_5)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate F1-scores and Macro F1-Score
f1_scores <- 2 * (precision * recall) / (precision + recall)
f1_scores[is.na(f1_scores)] <- 0
macro_f1_score <- mean(f1_scores, na.rm = TRUE)

# Display evaluation metrics
cat("F1-Scores for each class:\n", f1_scores, "\n")
cat("Macro F1-Score:\n", macro_f1_score, "\n")
cat("Precision for each class:\n", precision, "\n")
cat("Recall for each class:\n", recall, "\n")

# Analyze feature correlations
correlationMatrix <- cor(final_traindata_matrix)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.5)
cat("Indexes of highly correlated features:\n")
print(highlyCorrelated)

# Estimate and display variable importance
importance <- varImp(svm_model_5, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

## The Importance of Hyperparameter Tuning in Random Forest Models

In machine learning, **hyperparameter tuning** is vital for optimizing model performance. For Random Forest models, parameters such as the number of predictors to consider at each split 
(`mtry`), the splitting rule, and the minimum node size can greatly influence the accuracy and robustness of the model. Proper tuning helps prevent overfitting and underfitting by balancing 
model complexity with predictive power.
Using techniques like **5-fold cross validation**, we systematically explore different hyperparameter values through a grid search. This process enables us to identify the combination of 
parameters that maximize performance metrics—such as precision, recall, and the F1-score—on unseen data. The following code blocks demonstrate how to tune Random Forest models using the 
`ranger` method with 5-fold cross validation for different target variables.

# RF Tuning Strategy 1 with 5-fold Cross Validation

```r
# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels.
# 'final_traindata_matrix' contains the training features.
target_labels <- df_knn_final$bin_maj  # Classification column for strategy 1

# Split the data into training (80%) and testing (20%) sets.
train_indices <- createDataPartition(target_labels, p = 0.8, list = FALSE)
train_data <- final_traindata_matrix[train_indices, ]
test_data  <- final_traindata_matrix[-train_indices, ]
train_labels <- target_labels[train_indices]
test_labels  <- target_labels[-train_indices]

# Define training control using 5-fold cross validation.
train_control <- trainControl(method = "cv", number = 5)

# Determine the number of predictors in the training data.
num_features <- ncol(train_data)

# Create a tuning grid for the 'mtry' parameter.
# In this example, we explore mtry values from 1 to the floor of the square root of the number of features.
tune_grid <- expand.grid(
  mtry = seq(1, floor(sqrt(num_features)), by = 1),
  # num.trees = c(500, 1000),      # Example values for num.trees (optional)
  splitrule = c("gini", "extratrees"),  # Example values for the split rule
  min.node.size = c(1, 5, 10)      # Example values for minimum node size
)

# Train the Random Forest model using the 'ranger' method.
# The parameter 'importance = "impurity"' is used to assess variable importance.
rf_model_1 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "ranger",
  trControl = train_control,
  tuneGrid = tune_grid,
  importance = "impurity"
)

# Make predictions on the test set.
rf_predictions_1 <- predict(rf_model_1, test_data)

# Create and display the confusion matrix.
conf_matrix_rf_1 <- confusionMatrix(rf_predictions_1, test_labels)
cat("Confusion Matrix:\n")
conf_matrix_rf_1 <- as.matrix(conf_matrix_rf_1)
print(conf_matrix_rf_1)

# Calculate precision and recall for each class.
precision <- diag(conf_matrix_rf_1) / rowSums(conf_matrix_rf_1)
recall    <- diag(conf_matrix_rf_1) / colSums(conf_matrix_rf_1)
precision[is.na(precision)] <- 0
recall[is.na(recall)] <- 0

# Calculate the F1-score for each class and the macro F1-score (average).
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

# Estimate and display variable importance.
importance <- varImp(rf_model_1, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# RF Tuning Strategy 2 with 5-fold Cross Validation

```r
# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels.
target_labels <- df_knn_final$bin_one  # Classification column for strategy 2

# Split the data into training (80%) and testing (20%) sets.
train_indices <- createDataPartition(target_labels, p = 0.8, list = FALSE)
train_data <- final_traindata_matrix[train_indices, ]
test_data  <- final_traindata_matrix[-train_indices, ]
train_labels <- target_labels[train_indices]
test_labels  <- target_labels[-train_indices]

# Define training control using 5-fold cross validation.
train_control <- trainControl(method = "cv", number = 5)

# Determine the number of predictors.
num_features <- ncol(train_data)

# Create a tuning grid for 'mtry'.
tune_grid <- expand.grid(
  mtry = seq(1, floor(sqrt(num_features)), by = 1),
  # num.trees = c(500, 1000),  # Example values for num.trees (optional)
  splitrule = c("gini", "extratrees"),  # Example values for split rule
  min.node.size = c(1, 5, 10)  # Example values for min.node.size
)

# Train the Random Forest model using the 'ranger' method.
rf_model_2 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "ranger",
  trControl = train_control,
  tuneGrid = tune_grid,
  importance = "impurity"
)

# Make predictions on the test set.
rf_predictions_2 <- predict(rf_model_2, test_data)

# Create and display the confusion matrix.
conf_matrix_rf_2 <- confusionMatrix(rf_predictions_2, test_labels)
cat("Confusion Matrix:\n")
conf_matrix_rf_2 <- as.matrix(conf_matrix_rf_2)
print(conf_matrix_rf_2)

# Calculate precision and recall for each class.
precision <- diag(conf_matrix_rf_2) / rowSums(conf_matrix_rf_2)
recall    <- diag(conf_matrix_rf_2) / colSums(conf_matrix_rf_2)
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

# Estimate and display variable importance.
importance <- varImp(rf_model_2, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# RF Tuning Strategy 3 with 5-fold Cross Validation

```r
# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels.
target_labels <- df_knn_final$bin_all  # Classification column for strategy 3

# Split the data into training (80%) and testing (20%) sets.
train_indices <- createDataPartition(target_labels, p = 0.8, list = FALSE)
train_data <- final_traindata_matrix[train_indices, ]
test_data  <- final_traindata_matrix[-train_indices, ]
train_labels <- target_labels[train_indices]
test_labels  <- target_labels[-train_indices]

# Define training control using 5-fold cross validation.
train_control <- trainControl(method = "cv", number = 5)

# Determine the number of predictors.
num_features <- ncol(train_data)

# Create a tuning grid for 'mtry'.
tune_grid <- expand.grid(
  mtry = seq(1, floor(sqrt(num_features)), by = 1),
  # num.trees = c(500, 1000),  # Optional values for num.trees
  splitrule = c("gini", "extratrees"),  # Example split rules
  min.node.size = c(1, 5, 10)  # Example values for min.node.size
)

# Train the Random Forest model using the 'ranger' method.
rf_model_3 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "ranger",
  trControl = train_control,
  tuneGrid = tune_grid,
  importance = "impurity"
)

# Make predictions on the test set.
rf_predictions_3 <- predict(rf_model_3, test_data)

# Create and display the confusion matrix.
conf_matrix_rf_3 <- confusionMatrix(rf_predictions_3, test_labels)
cat("Confusion Matrix:\n")
conf_matrix_rf_3 <- as.matrix(conf_matrix_rf_3)
print(conf_matrix_rf_3)

# Calculate precision and recall for each class.
precision <- diag(conf_matrix_rf_3) / rowSums(conf_matrix_rf_3)
recall    <- diag(conf_matrix_rf_3) / colSums(conf_matrix_rf_3)
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

# Estimate and display variable importance.
importance <- varImp(rf_model_3, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# RF Tuning Strategy 4 with 5-fold Cross Validation

```r
# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels.
target_labels <- df_knn_final$multi_maj  # Classification column for strategy 4

# Split the data into training (80%) and testing (20%) sets.
train_indices <- createDataPartition(target_labels, p = 0.8, list = FALSE)
train_data <- final_traindata_matrix[train_indices, ]
test_data  <- final_traindata_matrix[-train_indices, ]
train_labels <- target_labels[train_indices]
test_labels  <- target_labels[-train_indices]

# Define training control using 5-fold cross validation.
train_control <- trainControl(method = "cv", number = 5)

# Determine the number of predictors.
num_features <- ncol(train_data)

# Create a tuning grid for 'mtry'.
tune_grid <- expand.grid(
  mtry = seq(1, floor(sqrt(num_features)), by = 1),
  # num.trees = c(500, 1000),  # Optional values for num.trees
  splitrule = c("gini", "extratrees"),  # Example split rules
  min.node.size = c(1, 5, 10)  # Example values for min.node.size
)

# Train the Random Forest model using the 'ranger' method.
rf_model_4 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "ranger",
  trControl = train_control,
  tuneGrid = tune_grid,
  importance = "impurity"
)

# Make predictions on the test set.
rf_predictions_4 <- predict(rf_model_4, test_data)

# Create and display the confusion matrix.
conf_matrix_rf_4 <- confusionMatrix(rf_predictions_4, test_labels)
cat("Confusion Matrix:\n")
conf_matrix_rf_4 <- as.matrix(conf_matrix_rf_4)
print(conf_matrix_rf_4)

# Calculate precision and recall for each class.
precision <- diag(conf_matrix_rf_4) / rowSums(conf_matrix_rf_4)
recall    <- diag(conf_matrix_rf_4) / colSums(conf_matrix_rf_4)
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

# Estimate and display variable importance.
importance <- varImp(rf_model_4, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

# RF Tuning Strategy 5 with 5-fold Cross Validation

```r
# Set the seed for reproducibility
set.seed(1234)

# Prepare the training data and target labels.
target_labels <- df_knn_final$disagree_bin  # Classification column for strategy 5

# Split the data into training (80%) and testing (20%) sets.
train_indices <- createDataPartition(target_labels, p = 0.8, list = FALSE)
train_data <- final_traindata_matrix[train_indices, ]
test_data  <- final_traindata_matrix[-train_indices, ]
train_labels <- target_labels[train_indices]
test_labels  <- target_labels[-train_indices]

# Define training control using 5-fold cross validation.
train_control <- trainControl(method = "cv", number = 5)

# Determine the number of predictors.
num_features <- ncol(train_data)

# Create a tuning grid for 'mtry'.
tune_grid <- expand.grid(
  mtry = seq(1, floor(sqrt(num_features)), by = 1),
  # num.trees = c(500, 1000),  # Optional values for num.trees
  splitrule = c("gini", "extratrees"),  # Example split rules
  min.node.size = c(1, 5, 10)  # Example values for min.node.size
)

# Train the Random Forest model using the 'ranger' method.
rf_model_5 <- train(
  x = final_traindata_matrix,
  y = target_labels,
  method = "ranger",
  trControl = train_control,
  tuneGrid = tune_grid,
  importance = "impurity"
)

# Make predictions on the test set.
rf_predictions_5 <- predict(rf_model_5, test_data)

# Create and display the confusion matrix.
conf_matrix_rf_5 <- confusionMatrix(rf_predictions_5, test_labels)
cat("Confusion Matrix:\n")
conf_matrix_rf_5 <- as.matrix(conf_matrix_rf_5)
print(conf_matrix_rf_5)

# Calculate precision and recall for each class.
precision <- diag(conf_matrix_rf_5) / rowSums(conf_matrix_rf_5)
recall    <- diag(conf_matrix_rf_5) / colSums(conf_matrix_rf_5)
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

# Estimate and display variable importance.
importance <- varImp(rf_model_5, scale = FALSE)
cat("Variable Importance:\n")
print(importance)
plot(importance)
```

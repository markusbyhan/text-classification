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

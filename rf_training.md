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

## Subtask 1: Prediction Using Trained SVM Models

In this section, we use the trained SVM models to generate predictions on the test dataset (`testdata`). For each SVM model (using different strategies), we compute the predictions 
and print them to the console. Finally, we compile all the predictions into a final data frame (`germeval_comp_task1_svm_final`) for further evaluation and analysis.

```r
# Predict using each trained SVM model on the test dataset
predictions_comp_svm_1 <- predict(svm_model_1, testdata)
print(predictions_comp_svm_1)

predictions_comp_svm_2 <- predict(svm_model_2, testdata)
print(predictions_comp_svm_2)

predictions_comp_svm_3 <- predict(svm_model_3, testdata)
print(predictions_comp_svm_3)

predictions_comp_svm_4 <- predict(svm_model_4, testdata)
print(predictions_comp_svm_4)

predictions_comp_svm_5 <- predict(svm_model_5, testdata)
print(predictions_comp_svm_5)

# Combine the predictions into a final data frame
germeval_comp_task1_svm_final <- data.frame(
  id = germ_comp_test$id,            # Use the 'id' from the test dataset
  bin_maj = predictions_comp_svm_1,   # Predictions from SVM model 1 (majority strategy)
  bin_one = predictions_comp_svm_2,   # Predictions from SVM model 2 (minimum one strategy)
  bin_all = predictions_comp_svm_3,   # Predictions from SVM model 3 (all annotators strategy)
  multi_maj = predictions_comp_svm_4, # Predictions from SVM model 4 (multi-majority strategy)
  disagree_bin = predictions_comp_svm_5  # Predictions from SVM model 5 (disagreement strategy)
)

# Display the final data frame containing all predictions
print(germeval_comp_task1_svm_final)
```

## Subtask 1: Prediction Using Trained RF Models

In this section, we use the trained Random Forest (RF) models to generate predictions on the test dataset (`testdata`). We compute predictions using five different 
strategies (e.g., `bin_maj`, `bin_one`, etc.) and then compile all the predictions into a final data frame (`germeval_comp_task1_rf_final`). This combined output can 
be used for further evaluation and comparison with the SVM model results.

```r
# Predict using each trained RF model on the test dataset
predictions_comp_rf_1 <- predict(rf_model_1, testdata)
print(predictions_comp_rf_1)

predictions_comp_rf_2 <- predict(rf_model_2, testdata)
print(predictions_comp_rf_2)

predictions_comp_rf_3 <- predict(rf_model_3, testdata)
print(predictions_comp_rf_3)

predictions_comp_rf_4 <- predict(rf_model_4, testdata)
print(predictions_comp_rf_4)

predictions_comp_rf_5 <- predict(rf_model_5, testdata)
print(predictions_comp_rf_5)

# Combine the predictions into a final data frame.
# The 'id' column is taken from the test dataset 'germ_comp_test'
germeval_comp_task1_rf_final <- data.frame(
  id = germ_comp_test$id,
  bin_maj = predictions_comp_rf_1,
  bin_one = predictions_comp_rf_2,
  bin_all = predictions_comp_rf_3,
  multi_maj = predictions_comp_rf_4,
  disagree_bin = predictions_comp_rf_5
)

# Display the final data frame with the RF predictions
print(germeval_comp_task1_rf_final)
```

# Save the combined SVM predictions to a TSV file

```r
write.table(germeval_comp_task1_svm_final, 
            "germeval_task1_svm_competition.tsv", 
            sep = "\t", row.names = FALSE, quote = FALSE)

# Save the combined RF predictions to a TSV file
write.table(germeval_comp_task1_rf_final, 
            "germeval_task1_rf_competition.tsv", 
            sep = "\t", row.names = FALSE, quote = FALSE)
```

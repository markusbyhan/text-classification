# k-NN Imputation and Final Data Preparation

This script demonstrates how to prepare a final dataframe, handle missing values using k-NN imputation, and ensure data completeness with a 
combination of advanced and simple imputation techniques. The steps include creating a final dataframe, applying k-NN imputation iteratively, 
and restoring labels to the processed data.

## Step 1: Create a Final Dataframe with All Label Variations

```R
train_comp_final <- data.frame(
  text = germ_comp_clean$text,
  A001 = germ_comp_clean$A001,
  A002 = germ_comp_clean$A002,
  A003 = germ_comp_clean$A003,
  A004 = germ_comp_clean$A004,
  A005 = germ_comp_clean$A005,
  A007 = germ_comp_clean$A007,
  A008 = germ_comp_clean$A008,
  A009 = germ_comp_clean$A009,
  A010 = germ_comp_clean$A010,
  A012 = germ_comp_clean$A012,
  bin_maj = train_prep_comp_2$bin_maj,
  bin_one = train_prep_comp_3$bin_one,
  bin_all = train_prep_comp_4$bin_all,
  multi_maj = train_prep_comp_5$multi_maj,
  disagree_bin = train_prep_comp_6$disagree_bin
)

# Inspect the structure of the final dataframe
glimpse(train_comp_final)

# Step 2: Preparing Data for k-NN Imputation
comp_knn <- select(train_comp_final, -c(bin_maj:disagree_bin))  # Exclude specific label columns
comp_knn <- select(comp_knn, -c(text))  # Remove the text column

# Transpose the data for easier handling in k-NN imputation
comp_knn_t <- t(comp_knn)

# Step 3: Define a Function for k-NN Imputation on Transposed Data
knn_impute_transposed <- function(comp_knn_t, k = 3) {
  imputed_data_t <- comp_knn_t
  for (i in 1:nrow(comp_knn_t)) {
    for (j in 1:ncol(comp_knn_t)) {
      if (is.na(comp_knn_t[i, j])) {
        # Calculate Euclidean distances to all other rows (columns in the original data)
        distances <- apply(comp_knn_t[-i, ], 1, function(row) {
          sqrt(sum((comp_knn_t[i, ] - row)^2, na.rm = TRUE))
        })
        # Find the indices of the k nearest neighbors without NAs
        nearest_neighbors <- order(distances, na.last = NA)[1:k]
        nearest_neighbors <- nearest_neighbors[!is.na(distances[nearest_neighbors])]
        # Impute with the mean of the k nearest neighbors
        if (length(nearest_neighbors) > 0) {
          imputed_value <- mean(comp_knn_t[nearest_neighbors, j], na.rm = TRUE)
          if (!is.nan(imputed_value)) {  # Check if the imputed value is valid
            imputed_data_t[i, j] <- imputed_value
          }
        }
      }
    }
  }
  return(imputed_data_t)
}

# Step 4: Apply k-NN Imputation Multiple Times
data_t_imputed <- knn_impute_transposed(comp_knn_t, k = 3)
data_t_imputed_2 <- knn_impute_transposed(data_t_imputed, k = 3)
data_t_imputed_3 <- knn_impute_transposed(data_t_imputed_2, k = 3)
data_t_imputed_4 <- knn_impute_transposed(data_t_imputed_3, k = 2)

# Step 5: Transpose Back the Imputed Data
data_imputed <- t(data_t_imputed)
data_imputed_2 <- t(data_t_imputed_2) %>% round()
data_imputed_3 <- t(data_t_imputed_3) %>% round()
data_imputed_4 <- t(data_t_imputed_4) %>% round()

# Convert the final imputed data into a dataframe
df_knn <- data.frame(data_imputed_4)

# Step 6: Handle Remaining NAs with Simple Mean Imputation
simple_impute_remaining_nas <- function(df_knn) {
  imputed_data <- df_knn
  for (j in 1:ncol(df_knn)) {
    if (any(is.na(df_knn[, j]))) {
      imputed_data[is.na(df_knn[, j]), j] <- mean(df_knn[, j], na.rm = TRUE)
    }
  }
  return(imputed_data)
}

# Apply simple mean imputation to fill remaining missing values
data_imputed_final <- simple_impute_remaining_nas(df_knn)

# Round the values to integers
data_imputed_final <- round(data_imputed_final)

# Ensure there are no NAs remaining
any(is.na(data_imputed_final))  # Should return FALSE

# Step 7: Add Back Label Columns to the Final Dataframe
df_knn_final <- data_imputed_final %>%
  mutate(
    bin_maj = train_prep_comp_2$bin_maj,
    bin_one = train_prep_comp_3$bin_one,
    bin_all = train_prep_comp_4$bin_all,
    multi_maj = train_prep_comp_5$multi_maj,
    disagree_bin = train_prep_comp_6$disagree_bin
  )
```

# Calculating Average Ratings and Binary Classification

The following steps calculate average ratings for each row, add a binary classification column, and ensure the data is prepared for machine learning tasks.

```r
# Step 1: Calculate Average Ratings
# Calculate the average rating for each row based on the selected rating columns.
# Missing values (NA) are ignored during the calculation.
data_imputed_final <- data_imputed_final %>%
  mutate(average_rating = rowMeans(select(., all_of(rating_cols)), na.rm = TRUE))

# Step 2: Create a Binary Classification Column
# Add a new binary column "binary_class" where rows with an average rating > 0.2 are classified as 1, and others as 0.
data_imputed_final$binary_class <- ifelse(data_imputed_final$average_rating > 0.2, 1, 0)

# Step 3: Convert Binary Classification to a Factor
# Convert the binary classification column into a factor to prepare it for machine learning tasks.
data_imputed_final$binary_class <- as.factor(data_imputed_final$binary_class)

# Step 4: Ensure Valid Factor Levels
# Adjust factor levels to ensure they are valid R variable names and compatible with models or visualizations.
data_imputed_final$binary_class <- as.factor(data_imputed_final$binary_class)
```

# Strategy 3 ("bin_all"): All Annotators

This strategy determines whether all annotators assigned a label greater than \`0\`.

In this modified version of the function, we use the \`all()\` function within the \`apply()\` function to check whether all valid values 
(\`valid_values\`, i.e., excluding \`NA\`s) are greater than \`0\`. If this condition is met, the function returns \`1\`; if at least one value 
is \`0\` or all values are \`NA\`, it returns \`0\`.
This approach is stricter than the previous ones, as it requires every valid element in the row to meet the condition for the function to return \`1\`.

```r
# Restrict to columns A001 - A012
train_prep_4 <- train_prep %>%
  select(A001:A012)

# Function for decision-making based on all annotators
all_non_zero <- function(train_prep_4) {
  # Apply the function to each row of the dataframe
  apply(train_prep_4, 1, function(row) {
    # Remove NA values
    valid_values <- row[!is.na(row)]
    # Check if all values are non-zero
    if (all(valid_values > 0)) {
      return(1)
    } else {
      return(0)
    }
  })
}

# Apply the function to the dataframe
results_all <- all_non_zero(train_prep_4)

# Add the results as a new column `bin_all` to train_prep_4
train_prep_4$bin_all <- results_all
train_prep_4$bin_all <- as.factor(train_prep_4$bin_all)
class(train_prep_4$bin_all)
```

# Strategy 2 ("bin_one"): At Least 1 Annotator

This strategy determines whether at least one annotator assigned a label other than \`0\`.

In this adjusted version of the function, we use the \`any()\` function within \`apply()\` to check whether at least one value in the list of valid 
values (\`valid_values\`) is greater than 0. If this condition is met, the function returns \`1\`; otherwise, it returns \`0\`.
This logic aligns with the requirement and is effective for decision-making when the goal is to check for the presence of any non-zero value rather 
than determining a majority.  

```r
# Restrict to columns A001 - A012
train_prep_3 <- train_prep %>%
  select(A001:A012)

# Function to check for at least 1 non-zero value
one_non_zero <- function(train_prep_3) {
  # Apply the function to each row of the dataframe
  apply(train_prep_3, 1, function(row) {
    # Remove NA values
    valid_values <- row[!is.na(row)]
    # Check if at least one value is non-zero
    if (any(valid_values > 0)) {
      return(1)
    } else {
      return(0)
    }
  })
}

# Apply the function to train_prep_3
results_one <- one_non_zero(train_prep_3)

# Add the results as a new column `bin_one` to train_prep_3
train_prep_3$bin_one <- results_one
train_prep_3$bin_one <- as.factor(train_prep_3$bin_one)
class(train_prep_3$bin_one)
```

# Strategy 1 ("bin_maj"): Majority Decision
  
Checks each row of the dataframe to determine whether the majority of valid values (excluding NAs) is `0` or not.  
Returns `1` if the majority of values are not `0`, and `0` if the majority of values are `0`. If no clear majority exists (e.g., an equal number 
of `0` and non-`0` values), both labels (`1` and `0`) can be considered correct. This can be handled by returning a specific value or through 
special treatment in the evaluation function.  

```r
# This section implements a majority decision strategy to determine binary labels based on user ratings.

# Restrict to columns A001 - A012
train_prep_comp_2 <- germ_comp_clean %>%
  select(A001:A012)

# Majority decision function
majority_decision <- function(train_prep_comp_2) {
  # Apply the function to each row of the dataframe
  apply(train_prep_comp_2, 1, function(row) {
    # Remove NA values
    valid_values <- row[!is.na(row)]
    # Count the number of 0 values
    count_zero <- sum(valid_values == 0)
    # Count the number of non-0 values
    count_non_zero <- sum(valid_values > 0)
    # Determine the majority
    if (count_zero > count_non_zero) {
      # Majority is 0
      return(0)
    } else if (count_zero < count_non_zero) {
      # Majority is non-0
      return(1)
    } else {
      # No clear majority, can be treated as a special case
      # For example, return NA or another indicator
      return(0)
    }
  })
}

# Execute the function
results <- majority_decision(train_prep_comp_2)

# Add columns to define final labels based on different strategies
train_prep_comp_2$bin_maj <- results
train_prep_comp_2$bin_maj <- as.factor(train_prep_comp_2$bin_maj)
```

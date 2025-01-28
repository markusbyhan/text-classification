# Strategy 4 ("multi_maj"): Multi-Majority

This strategy determines the majority label across all annotators for each text. If no clear majority exists (e.g., a tie or all values 
are \`NA\`), the result is set to \`NA\`.

**Determination of a Multi-Majority**

In this function, we use \`table()\` to create a frequency table of the values in each row (excluding \`NA\`). The \`which.max()\` 
function is used to identify the most frequently occurring label. If there is a single clear majority label, it is returned. 
If multiple labels occur with the same frequency or all values are \`NA\`, the function returns \`NA\` to indicate that no clear 
majority exists.

```r
# Restrict to columns A001 - A012
train_prep_5 <- train_prep %>%
  select(A001:A012)

# Function to determine the majority label
multi_majority <- function(train_prep_5) {
  apply(train_prep_5, 1, function(row) {
    # Remove NA values
    valid_values <- row[!is.na(row)]
    # Find the most common value, if it exists
    most_common <- which.max(table(valid_values))
    labels <- names(most_common)
    # Check if there is a clear majority
    if (length(labels) == 1) {
      # A clear majority exists
      return(as.numeric(labels))
    } else {
      # No clear majority
      return(NA)  # Or another special value to represent "any label is valid"
    }
  })
}

# Apply the function to the dataframe
results_multi <- multi_majority(train_prep_5)

# Add the results as a new column `multi_maj` to train_prep_5
train_prep_5$multi_maj <- results_multi
train_prep_5$multi_maj <- as.factor(train_prep_5$multi_maj)
class(train_prep_5$multi_maj)
```

# Preparing for Lemmatization with TreeTagger

The following steps prepare the tokens for lemmatization using the TreeTagger tool. Lemmatization provides a more context-aware text analysis compared to stemming. For example, the word "besser" 
would be lemmatized to its base form "gut," which cannot be achieved through simple stemming.
Here is the link to Helmut Schmidt's tool: [TreeTagger](https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/). Additionally, I recommend visiting 
[IMS Stuttgart](https://www.ims.uni-stuttgart.de/forschung/ressourcen/werkzeuge/treetagger/) for further information.

The \`koRpus\` package integrates with TreeTagger to perform lemmatization.

## Steps for Preparation

```r
# Set up the TreeTagger environment with the language set to German
set.kRp.env(TT.cmd = "C:\\TreeTagger\\cmd\\tree-tagger.exe", lang = "de")

# Save the tokens as a text file for TreeTagger input
write.table(tokens_comp_train_flat, file = "tokens_comp_train_flat.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# Create a file for tokens formatted for TreeTagger
fileConn <- file("tokens_comp_tree_tagger.txt")

# Initialize a list to store tokens with sentence boundaries
output_lines <- c()

# Loop through the rows of the tokens DataFrame
for (i in 1:nrow(tokens_comp_train_flat)) {
  # Split the tokens into individual words
  tokens <- unlist(strsplit(tokens_comp_train_flat$Tokens[i], " "))
  
  # Add tokens to the output
  output_lines <- c(output_lines, tokens)
  
  # Add a blank line after each sentence except the last
  if (i < nrow(tokens_comp_train_flat)) {
    output_lines <- c(output_lines, "")
  }
}

# Write tokens and sentence boundaries to the file
writeLines(output_lines, fileConn)

# Close the file connection
close(fileConn)

# Create a temporary file
input_file <- tempfile(fileext = ".txt")

# Write tokens and sentence boundaries to the temporary file
writeLines(output_lines, input_file)
```
## Steps for Lemmatization

```r
# Define the path to TreeTagger and its parameters
tree_tagger_path <- "path/to/TreeTagger" # Replace with your TreeTagger installation path
input_file <- file.path(tree_tagger_path, "lib", "tokens_tree_tagger.txt")
output_file <- "path/to/output/lemmatized_output_comp.txt" # Define the output file path

# Configure the TreeTagger environment
set.kRp.env(
  TT.cmd = file.path(tree_tagger_path, "cmd", "tree-tagger.exe"), # Path to TreeTagger executable
  lang = "de",
  TT.tknz = file.path(tree_tagger_path, "cmd", "utf8-tokenize.perl") # Path to tokenizer script
)

# Execute TreeTagger for lemmatization
tagged.results_comp <- treetag(
  file = sentence_vector_comp, # Input file containing tokens
  treetagger = "manual",
  format = "obj",
  lang = "de",
  stemmer = SnowballC::wordStem,
  TT.options = list(
    path = tree_tagger_path, 
    preset = "de",
    tokenizer = "utf8-tokenize.perl",  # Path to tokenizer script
    tagger = "tree-tagger.exe",  # Path to TreeTagger executable
    params = "german.par"  # Path to parameter file
  ),
  apply.sentc.end = TRUE,
  sentc.end = c("")
)

# Run the TreeTagger system command
taggedText(tagged_results_comp)

# Convert tagged tokens into a data frame
# Extract tokens, lemmas, and their IDs from `tagged_results_comp`
tree_tokens_comp <- as.data.frame(tagged_results_comp@tokens[["token"]]) %>%
  mutate(
    lemma = tagged_results_comp@tokens[["lemma"]],  # Add lemma information
    id = tagged_results_comp@tokens[["idx"]]       # Add token IDs
  )

# Include the original tokens explicitly in the data frame
tree_tokens_comp <- tree_tokens_comp %>%
  mutate(tokens = tagged_results_comp@tokens[["token"]])

# Replace tokens with lemmas, if available
# If the lemma is "<unknown>", retain the original token
tree_tokens_comp <- tree_tokens_comp %>%
  mutate(new_token = ifelse(lemma == "<unknown>", tagged_results_comp@tokens[["token"]], lemma))

# Normalize specific tokens (e.g., "frauen", "Frauen", "Frau" -> "frau")
# This ensures consistency in token representation
tree_tokens_comp <- tree_tokens_comp %>%
  mutate(new_token = ifelse(new_token %in% c("frauen", "Frauen", "Frau"), "frau", new_token))

# Remove the original first column (presumably redundant)
tree_tokens_comp <- tree_tokens_comp[, -1]

# Reorder columns for better organization
# Ensure the ID and tokens are prioritized
tree_tokens_comp <- tree_tokens_comp %>%
  select(id, tokens, everything())

```
# Function to replace tokens in a given text

```r
# `text`: The text in which tokens will be replaced
# `old_tokens`: A vector of tokens to be replaced
# `new_tokens`: A vector of replacement tokens
replace_tokens <- function(text, old_tokens, new_tokens) {
  for (i in seq_along(old_tokens)) {
    # Create a regex pattern to match whole words only
    pattern <- paste0("\\b", old_tokens[i], "\\b")
    # Replace the token with its corresponding new token
    text <- gsub(pattern, new_tokens[i], text, ignore.case = TRUE)
  }
  return(text)  # Return the modified text
}

# Function to replace tokens in a nested list
```

```r
# `text_list`: A nested list of texts (e.g., sentences)
# `tree_tokens_comp`: A data frame with `tokens` and `new_token` columns
replace_tokens_in_list <- function(text_list, tree_tokens_comp) {
  old_tokens <- tree_tokens_comp$tokens      # Tokens to be replaced
  new_tokens <- tree_tokens_comp$new_token  # Corresponding replacement tokens
  
  # Apply the replacement function to each sentence in the nested list
  lapply(text_list, function(sublist) {
    lapply(sublist, function(sentence) {
      replace_tokens(sentence, old_tokens, new_tokens)
    })
  })
}

# Apply the replacement function to the nested list
# `processed_lemma_comp`: The nested list of texts to process
# `tree_tokens_comp`: The data frame containing the token mappings
processed_lemma_clean <- replace_tokens_in_list(processed_lemma_comp, tree_tokens_comp)
```

```r
# Initialize DataFrame columns
# `index`: Holds the index of each list element
# `token_counts`: Stores the count of valid tokens per list element
# `token_strings`: Stores concatenated tokens as a single string for each list element
index <- integer()
token_counts <- integer()
token_strings <- character()

# Loop through the nested list and its sublists to populate the DataFrame columns
for (i in seq_along(processed_lemma_clean)) {
  # Filter tokens with more than one character or empty strings
  # Ensures valid tokens while retaining empty strings if present
  valid_tokens <- processed_lemma_clean[[i]][nchar(processed_lemma_clean[[i]]) > 1 | processed_lemma_clean[[i]] == ""]
  
  # Populate index, token counts, and concatenated token strings
  current_index <- length(index) + 1
  index[current_index] <- current_index
  token_counts[current_index] <- length(valid_tokens)
  token_strings[current_index] <- paste(valid_tokens, collapse = " ")
}

# Create the DataFrame
lemma_comp <- data.frame(
  Index = index,                  # Unique index for each list element
  TokenCount = token_counts,      # Number of valid tokens
  Tokens = token_strings,         # Concatenated token strings
  stringsAsFactors = FALSE        # Do not convert strings to factors
)

# Convert the DataFrame to a Corpus for text processing
# The Corpus is required for processing with the `tm` package
lemma_comp_corpus <- Corpus(VectorSource(lemma_comp$Tokens))
```

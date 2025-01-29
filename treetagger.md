# Preparing for Lemmatization with TreeTagger

The following steps prepare the tokens for lemmatization using the TreeTagger tool. Lemmatization provides a more context-aware text analysis compared to stemming. For example, the word "besser" 
would be lemmatized to its base form "gut," which cannot be achieved through simple stemming.

The \`koRpus\` package integrates with TreeTagger to perform lemmatization.

## Steps for Preparation

```r
# Set up the TreeTagger environment with the language set to German
set.kRp.env(TT.cmd = "C:\\TreeTagger\\cmd\\tree-tagger.exe", lang = "de")

# Save the tokens as a text file for TreeTagger input
write.table(tokens_flat, file = "tokens_flat.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# Create a file for tokens formatted for TreeTagger
fileConn <- file("tokens_tree_tagger.txt")

# Initialize a list to store tokens with sentence boundaries
output_lines <- c()

# Loop through the rows of the tokens DataFrame
for (i in 1:nrow(tokens_flat)) {
  # Split the tokens into individual words
  tokens <- unlist(strsplit(tokens_flat$Tokens[i], " "))
  
  # Add tokens to the output
  output_lines <- c(output_lines, tokens)
  
  # Add a blank line after each sentence except the last
  if (i < nrow(tokens_flat)) {
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
output_file <- "path/to/output/lemmatized_output.txt" # Define the output file path

# Configure the TreeTagger environment
set.kRp.env(
  TT.cmd = file.path(tree_tagger_path, "cmd", "tree-tagger.exe"), # Path to TreeTagger executable
  lang = "de",
  TT.tknz = file.path(tree_tagger_path, "cmd", "utf8-tokenize.perl") # Path to tokenizer script
)

# Execute TreeTagger for lemmatization
tagged.results <- treetag(
  file = "tokens_tree_tagger.txt", # Input file containing tokens
  treetagger = "manual",
  format = "obj",
  lang = "de",
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
system(tagged.results)
```

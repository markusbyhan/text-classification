# Creating a Corpus Object for the Training Data

To process the training data using the \`tm\` package, it is necessary to create a corpus object.

```r
# Create a corpus for the training data
# The Corpus() function transforms the text data into a format suitable for text mining
# The VectorSource() function converts the text column into a format that Corpus() can process
corpus_comp_train <- Corpus(VectorSource(germ_comp_clean$text))
```

# Text Preprocessing

The following functions are applied to preprocess the text data within the corpus object.

```r
# Function 1: Basic text preprocessing
text_preprocessing_1 <- function(corpus_comp_train) {
  # Convert text to lowercase for consistency
  corpus_comp_train <- tm_map(corpus_comp_train, content_transformer(tolower))
  # Remove extra whitespace
  corpus_comp_train <- tm_map(corpus_comp_train, stripWhitespace)
  # Remove punctuation marks
  corpus_comp_train <- tm_map(corpus_comp_train, removePunctuation)
  # Remove numerical digits
  corpus_comp_train <- tm_map(corpus_comp_train, removeNumbers)
  return(corpus_comp_train)
}

# Apply the first preprocessing function to the corpus
corpus_comp_train <- text_preprocessing_1(corpus_comp_train)

# Function 2: Advanced text preprocessing
text_preprocessing_2 <- function(corpus_comp_train) {
  corpus_comp_train <- tm_map(corpus_comp_train, content_transformer(function(text) {
    # Remove URLs
    text <- gsub('http\\S+\\s*', '', text)
    # Remove control characters and special symbols
    text <- gsub('[[:cntrl:]]', '', text)
    # Remove all non-alphanumeric characters except spaces
    # Example: gsub("[^[:alnum:]\\s]", "", text)
    # Remove leading whitespace
    text <- gsub("^[[:space:]]*", "", text)
    # Remove trailing whitespace
    text <- gsub("[[:space:]]*$", "", text)
    # Replace multiple spaces with a single space
    text <- gsub(' +', ' ', text)
    return(text)
  }))
  return(corpus_comp_train)
}

# Apply the second preprocessing function to the corpus
corpus_comp_train <- text_preprocessing_2(corpus_comp_train)
```
Bild einfügen! corpus_comp_train_2.png

# Stopword Handling

The following step removes stopwords from the text in the corpus using the pre-defined list `all_stopwords`.

```r
# Remove stopwords from the corpus
# The removeWords function takes the list of stopwords (all_stopwords) and removes them from the text
corpus_comp_train_2 <- tm_map(corpus_comp_train, removeWords, all_stopwords)

# Inspect the 5th document in the processed corpus to verify the removal of stopwords
inspect(corpus_comp_train_2[5])
```
# Tokenization, Stemming, and Lemmatization

## Stemming
Stemming is the process of reducing a word to its root form by removing or replacing suffixes. It's important to be aware of potential over- or understemming:

- **Overstemming:** Words like "university" and "universe" may both be reduced to the root "univers," implying they share the same meaning. This is incorrect.
- **Understemming:** Words like "data" and "datum" may have the root "dat," failing to distinguish between their lexical forms.

## Lemmatization
Lemmatization involves identifying the base form of a word by utilizing lexical knowledge, aiming for greater precision than stemming.

# Tokenization of the Corpus

The following steps define a word tokenizer function and apply it to tokenize the text data in the corpus.

```r
# Define the word_tokenizer function
# This function splits a text into tokens (words) based on whitespace
word_tokenizer <- function(text) {
  unlist(strsplit(text, "\\s+")) # Split the text by spaces and return a list of words
}

# Tokenize the corpus using the word_tokenizer function
tokenize_corpus <- function(corpus_comp_train_2) {
  tokens <- list() # Initialize an empty list to store tokens
  for (i in 1:length(corpus_comp_train_2)) {
    # Convert the text in the corpus to a character string
    text <- as.character(corpus_comp_train_2[[i]])
    # Apply the word tokenizer to the text
    tokens[[i]] <- word_tokenizer(text)
  }
  return(tokens) # Return the list of tokenized texts
}

# Apply the tokenization function to the processed corpus
tokens_comp_train <- tokenize_corpus(corpus_comp_train_2)
```

Bild einfügen! tokens_comp_train.png

# Converting the Token List into a DataFrame

The following steps iterate through the token list to populate the columns of a DataFrame with token-related information.

```r
# Initialization of variables
# `index`: Stores the indices of the list elements
# `token_counts`: Stores the count of valid tokens for each list element
# `token_strings`: Stores the valid tokens as a concatenated string
index <- numeric(length(tokens_comp_train))
token_counts <- numeric(length(tokens_comp_train))
token_strings <- character(length(tokens_comp_train))

# Loop through the `tokens_comp_train` list
for (i in seq_along(tokens_comp_train)) {
  # Store the current index in the `index` list
  index[i] <- i
  
  # Filter tokens that have more than one character
  # `nchar` checks the length of each token
  valid_tokens <- tokens_comp_train[[i]][nchar(tokens_comp_train[[i]]) > 1]
  
  # Store the number of valid tokens
  token_counts[i] <- length(valid_tokens)
  
  # If valid tokens are present, concatenate them into a string
  # Otherwise, store an empty string
  if (length(valid_tokens) > 0) {
    token_strings[i] <- paste(valid_tokens, collapse = " ")
  } else {
    token_strings[i] <- ""
  }
}

# Create a dataframe with the collected information
tokens_comp_train_flat <- data.frame(
  Index = index,                  # Indices
  TokenCount = token_counts,      # Number of valid tokens
  Tokens = token_strings,         # Tokens as a single string
  stringsAsFactors = FALSE        # Do not convert strings to factors
)

# Further processing of the dataframe
# `Corpus`: Creates a text corpus from the tokens
corpus_comp_clean <- Corpus(VectorSource(tokens_comp_train_flat$Tokens))

# Tokenize the text corpus
tokens_comp_clean <- tokenize_corpus(corpus_comp_clean)
```

Bild einfügen: tokens_comp_clean

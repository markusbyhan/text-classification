**Reminder:**

After completing the steps in **"preprocessing_training"** and determining a label based on the 5 strategies, the models are now ready to be trained.

In the next step, all label variations will be consolidated into a final dataframe. This serves as the starting point for the creation of a corpus and subsequent 
training of the models.

```r
train_prep_final <- data.frame(
  text = train_prep$text,
  A001 = train_prep$A001,
  A002 = train_prep$A002,
  A003 = train_prep$A003,
  A004 = train_prep$A004,
  A005 = train_prep$A005,
  A007 = train_prep$A007,
  A008 = train_prep$A008,
  A009 = train_prep$A009,
  A010 = train_prep$A010,
  A011 = train_prep$A011,
  A012 = train_prep$A012,
  bin_maj = train_prep_2$bin_maj,
  bin_one = train_prep_3$bin_one,
  bin_all = train_prep_4$bin_all,
  multi_maj = train_prep_5$multi_maj,
  disagree_bin = train_prep_6$disagree_bin
)

# Check for missing values in the `bin_maj` column
sum(is.na(train_prep_final$bin_maj))
```

# Creating a Corpus Object for the Training Data

To process the training data using the `tm` package, it is necessary to create a corpus object.

```r
# Create a corpus for the training data
# The Corpus() function transforms the text data into a format suitable for text mining
# The VectorSource() function converts the text column into a format that Corpus() can process
corpus_train <- Corpus(VectorSource(germ_train_clean$text))
```

# Text Preprocessing

The following functions are applied to preprocess the text data within the corpus object.

```r
# Function 1: Basic text preprocessing
text_preprocessing_1 <- function(corpus_train) {
  # Convert text to lowercase for consistency
  corpus_train <- tm_map(corpus_train, content_transformer(tolower))
  # Remove extra whitespace
  corpus_train <- tm_map(corpus_train, stripWhitespace)
  # Remove punctuation marks
  corpus_train <- tm_map(corpus_train, removePunctuation)
  # Remove numerical digits
  corpus_train <- tm_map(corpus_train, removeNumbers)
  return(corpus_train)
}

# Apply the first preprocessing function to the corpus
corpus_train <- text_preprocessing_1(corpus_train)

# Function 2: Advanced text preprocessing
text_preprocessing_2 <- function(corpus_train) {
  corpus_train <- tm_map(corpus_train, content_transformer(function(text) {
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
  return(corpus_train)
}

# Apply the second preprocessing function to the corpus
corpus_train <- text_preprocessing_2(corpus_train)
```

# Stopword Handling

The following step removes stopwords from the text in the corpus using the pre-defined list `all_stopwords`.

```r
# Remove stopwords from the corpus
# The removeWords function takes the list of stopwords (all_stopwords) and removes them from the text
corpus_train_2 <- tm_map(corpus_train, removeWords, all_stopwords)

# Inspect the 5th document in the processed corpus to verify the removal of stopwords
inspect(corpus_train_2[5])
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
tokenize_corpus <- function(corpus_train_2) {
  tokens <- list() # Initialize an empty list to store tokens
  for (i in 1:length(corpus_train_2)) {
    # Convert the text in the corpus to a character string
    text <- as.character(corpus_train_2[[i]])
    # Apply the word tokenizer to the text
    tokens[[i]] <- word_tokenizer(text)
  }
  return(tokens) # Return the list of tokenized texts
}

# Apply the tokenization function to the processed corpus
tokens_train <- tokenize_corpus(corpus_train_2)
```
# Converting the Token List into a DataFrame

The following steps iterate through the token list to populate the columns of a DataFrame with token-related information.

```r
# Initialize vectors for the DataFrame columns
index <- numeric(length(tokens_train))         # To store indices of the texts
token_counts <- numeric(length(tokens_train))  # To store the count of valid tokens
token_strings <- character(length(tokens_train)) # To store the concatenated token strings

# Loop through the token list to fill the DataFrame columns
for (i in seq_along(tokens_train)) {
  # Store the index
  index[i] <- i
  
  # Filter tokens that are longer than one character
  valid_tokens <- tokens_train[[i]][nchar(tokens_train[[i]]) > 1]
  
  # Store the count of valid tokens
  token_counts[i] <- length(valid_tokens)
  
  # Concatenate valid tokens into a single string
  if (length(valid_tokens) > 0) {
    token_strings[i] <- paste(valid_tokens, collapse = " ") # Join tokens with spaces
  } else {
    token_strings[i] <- "" # Empty string if no valid tokens exist
  }
}

# Create the DataFrame
tokens_flat <- data.frame(
  Index = index,           # Index of each text
  TokenCount = token_counts, # Number of valid tokens
  Tokens = token_strings,  # Concatenated tokens as a single string
  stringsAsFactors = FALSE # Prevent automatic conversion of strings to factors
)
```

# TF-IDF Calculation

As a reminder, here you can find some explanation about TF-IDF:

- **Document-Term Matrix (DTM):** Represents the frequency of terms in each document.
- **TF:** Direct count of term occurrences in each document.
- **DF:** Counts the documents where a term is present.
- **IDF:** Penalizes common terms by reducing their weight in the TF-IDF score.
- **TF-IDF:** A measure that balances term importance in a document relative to its rarity across the corpus.

The following section explains and computes the Inverse Document Frequency (IDF) based on the formula:

- **Term Frequency (TF):** The number of times a term appears in a document.
- **Document Frequency (DF):** The number of documents that contain each term.
- **Inverse Document Frequency (IDF):** \( IDF = 1 / DF \)
- **TF-IDF:** \( TF\_IDF = TF \times IDF \)

```r
# Feature Extraction Using TF-IDF

The following steps extract features using TF-IDF for all terms in the corpus, with additional processing steps such as reducing sparsity and scaling.

# Create a Document-Term Matrix (DTM) from the tokenized corpus
# The DTM represents the frequency of each term in each document
dtm_train <- DocumentTermMatrix(Corpus(VectorSource(tokens_flat$Tokens)))

# Compute TF-IDF for the terms in the DTM
# The weightTfIdf function calculates the TF-IDF values for the matrix
tfidf_train <- weightTfIdf(dtm_train)

# Reduce sparsity to keep only frequent terms
# The removeSparseTerms function removes terms that appear in less than (1 - 0.965) documents
tfidf_train <- removeSparseTerms(tfidf_train, sparse = 0.965)

# Convert the resulting TF-IDF matrix into a standard matrix format
# This allows for easier manipulation and processing
tfidf_train_matrix <- as.matrix(tfidf_train)

# Ensure column names are valid for further processing
colnames(tfidf_train_matrix) <- make.names(colnames(tfidf_train_matrix))

# Print a summary of the TF-IDF matrix for inspection
summary(tfidf_train_matrix)

# Scale the matrix for normalization
# The scale function normalizes the TF-IDF values to ensure comparability across features
tfidf_train_matrix_scaled <- scale(tfidf_train_matrix)
```

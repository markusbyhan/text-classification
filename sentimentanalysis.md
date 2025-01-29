# Sentiment Analysis with the "syuzhet" Package

The following steps perform a sentiment analysis using the **syuzhet** package. Sentiment scores are calculated based on the NRC Sentiment Lexicon, which provides sentiment categories such as joy, anger, sadness, and more.

## Steps for Sentiment Analysis

```r
# Specify the path to the .txt file containing the text data
# Replace "path/to/tree_tokens_data.txt" with the actual path to your text file
sentiment_path <- "path/to/tree_tokens_data.txt"

# Read the .txt file into a string for processing
# The `scan` function reads the file line by line and converts it into a character vector
text_string <- scan(file = sentiment_path, fileEncoding = "UTF-8", what = character(), sep = "\n", allowEscapes = TRUE)

# Tokenize the text into individual words
# The `get_tokens` function splits the string into tokens (words) for analysis
text_words <- get_tokens(text_string)

# Check the total number of words
# Use `length` to count the number of tokens
length(text_words) # Prints the number of words

# Convert the text into sentences for sentence-based analysis
# The `get_sentences` function splits the text into sentences, which can be used for sentence-level sentiment analysis
sentence_vector <- get_sentences(text_words)

# Perform sentiment analysis using the NRC Sentiment Lexicon
# The `get_nrc_sentiment` function assigns sentiment scores to individual words based on the lexicon
# Set `lang = "german"` to use the German NRC Lexicon
sentiment_scores <- get_nrc_sentiment(text_words, lang = "german")

# Display the first few rows of sentiment scores
# Use `head` to preview the first rows of the resulting data frame
head(sentiment_scores)

# Summarize the sentiment scores
# Use `summary` to provide a statistical overview of the sentiment scores
summary(sentiment_scores)
```

# Visualizing Sentiment Analysis

The following steps provide a visual representation of the sentiment analysis results using bar plots. Additionally, the most frequent words associated with specific emotions and sentiments are extracted and displayed.

---

## Steps for Visualization

### 1. Emotion Distribution
A bar plot is used to display the proportional distribution of emotions in the training data.

```r
# Bar plot for emotion distribution
barplot(
  colSums(prop.table(sentiment_scores[, 1:8])), # Proportional sum of emotions
  space = 0.2,
  horiz = FALSE,
  las = 1,
  cex.names = 0.7,
  col = brewer.pal(n = 8, name = "Set3"), # Color palette for emotions
  main = "Training Data",
  sub = "Emotion Distribution",
  xlab = "Emotions", 
  ylab = NULL
)
```

### 2. Positive/Negative Sentiment Distribution
```r
barplot(
  colSums(prop.table(sentiment_scores[, 9:10])), # Proportional sum of positive/negative sentiments
  space = 0.2,
  horiz = FALSE,
  las = 1,
  cex.names = 0.7,
  col = brewer.pal(n = 2, name = "Set3"), # Color palette for sentiments
  main = "Training Data",
  sub = "Positive/Negative Distribution",
  xlab = "Sentiments", 
  ylab = NULL
)
```

### Frequent Words by Sentiment
```r
Anger
anger_words <- text_words[sentiment_scores$anger > 0]
anger_word_order <- sort(table(unlist(anger_words)), decreasing = TRUE)
head(anger_word_order, n = 12)
length(anger_word_order)

Anticipation
anticipation_words <- text_words[sentiment_scores$anticipation > 0]
anticipation_word_order <- sort(table(unlist(anticipation_words)), decreasing = TRUE)
head(anticipation_word_order, n = 12)
length(anticipation_word_order)

Disgust
disgust_words <- text_words[sentiment_scores$disgust > 0]
disgust_word_order <- sort(table(unlist(disgust_words)), decreasing = TRUE)
head(disgust_word_order, n = 12)
length(disgust_word_order)

Fear
fear_words <- text_words[sentiment_scores$fear > 0]
fear_word_order <- sort(table(unlist(fear_words)), decreasing = TRUE)
head(fear_word_order, n = 12)
length(fear_word_order)

Joy
joy_words <- text_words[sentiment_scores$joy > 0]
joy_word_order <- sort(table(unlist(joy_words)), decreasing = TRUE)
head(joy_word_order, n = 12)
length(joy_word_order)

Sadness
sadness_words <- text_words[sentiment_scores$sadness > 0]
sadness_word_order <- sort(table(unlist(sadness_words)), decreasing = TRUE)
head(sadness_word_order, n = 12)
length(sadness_word_order)

Surprise
surprise_words <- text_words[sentiment_scores$surprise > 0]
surprise_word_order <- sort(table(unlist(surprise_words)), decreasing = TRUE)
head(surprise_word_order, n = 12)
length(surprise_word_order)

Trust
trust_words <- text_words[sentiment_scores$trust > 0]
trust_word_order <- sort(table(unlist(trust_words)), decreasing = TRUE)
head(trust_word_order, n = 12)
length(trust_word_order)

Negative Sentiment
negative_words <- text_words[sentiment_scores$negative > 0]
negative_word_order <- sort(table(unlist(negative_words)), decreasing = TRUE)
head(negative_word_order, n = 12)
length(negative_word_order)

Positive Sentiment
positive_words <- text_words[sentiment_scores$positive > 0]
positive_word_order <- sort(table(unlist(positive_words)), decreasing = TRUE)
head(positive_word_order, n = 12)
length(positive_word_order)
```

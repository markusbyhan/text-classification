# Sentiment Analysis with the "syuzhet" Package

The following steps perform a sentiment analysis using the **syuzhet** package. Sentiment scores are calculated based on the NRC Sentiment Lexicon, which provides sentiment categories such as joy, anger, sadness, and more.

## Steps for Sentiment Analysis

```r
# Function to save a DataFrame to a text file
save_dataframe_to_text <- function(tokens_comp_train_flat, file_name) {
  # Open a connection to the specified file in write mode
  # This allows writing text content to the file
  file_conn <- file(file_name, open = "wt")
  
  # Iterate through each row of the "Tokens" column
  for (i in 1:nrow(tokens_comp_train_flat)) {
    # Write the sentence (tokens) to the file
    # Each row of the "Tokens" column is written as a separate line
    writeLines(tokens_comp_train_flat$Tokens[i], file_conn)
    
    # Add a blank line after each sentence
    # Blank lines improve readability or can be required by certain NLP tools
    writeLines("", file_conn)
  }
  
  # Close the file connection
  # Ensures that all data is written to the file and resources are released
  close(file_conn)
}

# Save the DataFrame to a text file
# This step writes the tokenized data from the DataFrame to a text file
# The resulting file can be used for further sentiment analysis or NLP processing
save_dataframe_to_text(tokens_comp_train_flat, "tokens_comp_train_flat.txt")
```

```r
# Extract sentences from the tokenized data
# `get_sentences` splits the tokenized text into individual sentences
sentence_vector_comp <- get_sentences(tokens_comp_train_flat$Tokens)

# Check the number of sentences extracted
# This helps to validate that sentences are correctly split
length(sentence_vector_comp)

# Perform sentiment analysis using the NRC Sentiment Lexicon
# `get_nrc_sentiment` assigns sentiment scores for each emotion category
# `lang = "german"` specifies that the analysis uses German lexicons
sentiment_scores_comp_sen <- get_nrc_sentiment(tokens_comp_train_flat$Tokens, lang = "german")

# Display the first few rows of sentiment scores
# This provides a quick look at the sentiment distribution for initial analysis
head(sentiment_scores_comp_sen)

# Summarize the sentiment scores
# This gives an overview of sentiment distributions across the dataset
summary(sentiment_scores_comp_sen)
```
Bild einfügen: summary_sentiment

### Frequent Words by Sentiment
```r
Anger
anger_words <- text_words[sentiment_scores_comp_sen$anger > 0]
anger_word_order <- sort(table(unlist(anger_words)), decreasing = TRUE)
head(anger_word_order, n = 12)
length(anger_word_order)

Anticipation
anticipation_words <- text_words[sentiment_scores_comp_sen$anticipation > 0]
anticipation_word_order <- sort(table(unlist(anticipation_words)), decreasing = TRUE)
head(anticipation_word_order, n = 12)
length(anticipation_word_order)

Disgust
disgust_words <- text_words[sentiment_scores_comp_sen$disgust > 0]
disgust_word_order <- sort(table(unlist(disgust_words)), decreasing = TRUE)
head(disgust_word_order, n = 12)
length(disgust_word_order)

Fear
fear_words <- text_words[sentiment_scores_com_sen$fear > 0]
fear_word_order <- sort(table(unlist(fear_words)), decreasing = TRUE)
head(fear_word_order, n = 12)
length(fear_word_order)

Joy
joy_words <- text_words[sentiment_scores_comp_sen$joy > 0]
joy_word_order <- sort(table(unlist(joy_words)), decreasing = TRUE)
head(joy_word_order, n = 12)
length(joy_word_order)

Sadness
sadness_words <- text_words[sentiment_scores_comp_sen$sadness > 0]
sadness_word_order <- sort(table(unlist(sadness_words)), decreasing = TRUE)
head(sadness_word_order, n = 12)
length(sadness_word_order)

Surprise
surprise_words <- text_words[sentiment_scores_comp_sen$surprise > 0]
surprise_word_order <- sort(table(unlist(surprise_words)), decreasing = TRUE)
head(surprise_word_order, n = 12)
length(surprise_word_order)

Trust
trust_words <- text_words[sentiment_scores_comp_sen$trust > 0]
trust_word_order <- sort(table(unlist(trust_words)), decreasing = TRUE)
head(trust_word_order, n = 12)
length(trust_word_order)

Negative Sentiment
negative_words <- text_words[sentiment_scores_comp_sen$negative > 0]
negative_word_order <- sort(table(unlist(negative_words)), decreasing = TRUE)
head(negative_word_order, n = 12)
length(negative_word_order)

Positive Sentiment
positive_words <- text_words[sentiment_scores_comp_sen$positive > 0]
positive_word_order <- sort(table(unlist(positive_words)), decreasing = TRUE)
head(positive_word_order, n = 12)
length(positive_word_order)
```
# Visualizing Sentiment Analysis

The following steps provide a visual representation of the sentiment analysis results using bar plots. Additionally, the most frequent words associated with specific emotions and sentiments are extracted and displayed.

---

## Steps for Visualization

### 1. Emotion Distribution
A bar plot is used to display the proportional distribution of emotions in the training data.

```r
# Calculate proportions of sums for sentiment scores
# This computes the proportional distribution of each sentiment category
proportions <- colSums(prop.table(sentiment_comp_sen[, 1:10]))

# Define the standard spacing for bars
# Spaces control the visual gaps between bars in the barplot
spaces <- rep(0.2, 9)  # Define equal spacing for 9 gaps (between 10 bars)

# Increase the spacing specifically between the 8th and 9th bars
# This highlights the separation between specific sentiment categories
spaces[9] <- 1  # Add larger gap between bars 8 and 9

# Define colors for all bars
# Use a predefined palette for the first 8 bars
colors <- brewer.pal(n = 8, name = "Set3")

# Ensure the palette length matches the number of bars (10 in total)
if (length(colors) < 10) {
  colors <- rep(colors, length.out = 10)
}

# Customize colors for the 9th and 10th bars
colors[9] <- "#eb3434"    # Custom red color for the 9th bar
colors[10] <- "#099e22"   # Custom green color for the 10th bar

# Create the bar plot
barplot(
  proportions,           # Data to visualize
  space = spaces,        # Apply individual spacing
  horiz = FALSE,         # Vertical bars
  las = 1,               # Rotate axis labels to be horizontal
  cex.names = 0.5,       # Adjust size of category labels
  col = colors,          # Apply color palette
  main = "Sentiment analysis of training data",  # Chart title
  xlab = "Distribution of emotions",            # X-axis label
  ylab = "Share of the corpus"                  # Y-axis label
)
```
Bild einfügen: Sentiment_train

```

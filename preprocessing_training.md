# 1. Pre-Processing
The following steps include reading the training dataset and stopword lists for preprocessing tasks.

## 1.1 Reading the Training Dataset and preparing the Stopword Lists
```r
file_path <- "C:/path/to/training-dataset/germeval-competition-traindev.jsonl" # Define file paths
con <- file(file_path, "r")
germ_comp_train <- stream_in(con)
close(con)

# 1.2 Stopword Removal
# Based on the repository: https://github.com/stopwords-iso/stopwords-iso
stopword_list_1 <- "C:/path/to/stopwords/stopwords-de.txt" 
stopwords_1 <- readLines(stopword_list_1)

# Based on the repository: https://github.com/solariz/german_stopwords/blob/master/german_stopwords_plain.txt
stopword_list_2 <- "C:/path/to/stopwords/german_stopwords_plain.txt" 
stopwords_2 <- readLines(stopword_list_2)

# Built-in stopword list
stopwords_german <- stopwords(language = "de") 

# Combine all stopwords
all_stopwords <- unique(c(stopwords_1, stopwords_2, stopwords_german))
```

# 1.2 Transformation of the "annotations" Column

The following code transforms the \`annotations\` column in the dataset by unnesting it and restructuring it to have separate columns for each user.

```r
germ_comp_train_2 <- germ_comp_train %>%
  mutate(row = row_number()) %>%         # Add a row number to uniquely identify each row
  unnest(annotations) %>%                # Unnest the annotations column
  pivot_wider(names_from = user,         # Create separate columns for each user
              values_from = label) %>%   # Extract the labels for each user
  select(-row)                           # Remove the temporary row number column
```
![Sample Image](D:\BA_Github\germ_train_2.png "germ_train_2")

Reorder the columns corresponding to users or moderators into the correct sequence. Subsequently, clean the categorizations by removing descriptive texts 
from the labels and converting them into numerical values for classification, based on the conducted assessments:
- **0 = None**  
- **1 = Minimal**  
- **2 = Present**  
- **3 = Strong**  
- **4 = Extreme**
  
Finally, duplicates are checked and removed if necessary. The removal of rows with missing values is not applicable, as not all users have provided a label for every text.

# 1.3 Cleaning and Preprocessing the Data Structure

The following steps adjust the column order, clean the labels to retain only numeric values, and convert column types for further analysis. Finally, duplicates are removed.

```r
# Adjusting the column order:
germ_comp_train_3 <- germ_comp_train_2 %>%
  select(-starts_with("A")) %>%  # Select all columns except those starting with "A001-A012"
  bind_cols(
    germ_comp_train_2 %>%
    select(starts_with("A")) %>%  # Select only columns that start with "A"
    select(sort(names(.)))  # Sort column names alphabetically
  )

# Cleaning the labels to retain numeric values:
germ_comp_train_4 <- germ_comp_train_3 %>%
  mutate(across(starts_with("A"), ~if_else(is.na(.), ., str_extract(., "^[0-9]+"))))

# Converting column types to numeric:
germ_comp_train_4 <- germ_comp_train_4 %>%
  mutate(across(starts_with("A"), as.numeric))

# Removing duplicates:
germ_comp_clean <- unique(germ_comp_train_4)
```
![Sample Image](D:\BA_Github\germ_train_clean.png "germ_train_clean")

## 1.4 Let's Take a Look at Basic Information About the Training Dataset

```r
# 3.1 To Do: Statistics of the Training Dataset

# Create a data frame to calculate statistics
comp_stats <- germ_comp_clean %>%
  select(id, text) %>%
  mutate(word_count = str_count(text, "\\S+"))  # Count the number of words per text

# Calculate word statistics (min, max, average) before removing stopwords
comp_stats_2 <- data.frame(
  texte_training = nrow(comp_stats),             # Total number of texts
  total_word_count = sum(comp_stats$word_count), # Total word count
  max_words = max(comp_stats$word_count),        # Maximum words in a text
  min_words = min(comp_stats$word_count),        # Minimum words in a text
  avg_words = mean(comp_stats$word_count)        # Average words per text
)

# Count the number of ratings per text
rating_counts <- germ_comp_clean %>%
  rowwise() %>%
  mutate(rating_count = sum(!is.na(c_across(A001:A012)))) %>%
  ungroup() %>%
  select(id, rating_count)

# Attach the rating count to germ_train_clean
germ_comp_clean <- germ_comp_clean %>%
  left_join(rating_counts, by = "id")

# Select columns for ratings starting with 'A'
rating_cols <- grep("^A", names(germ_comp_clean), value = TRUE)

# Calculate the average rating for each row and add as a new column "average_rating"
germ_comp_clean <- germ_comp_clean %>%
  mutate(average_rating = rowMeans(select(., all_of(rating_cols)), na.rm = TRUE))

# Group texts into 5 groups based on word count: 1 - 173, average: 33
germ_comp_statistics <- comp_stats %>%
  mutate(quantity_words =
           ifelse(comp_stats$word_count %in% 1:30, "1 - 30",
           ifelse(comp_stats$word_count %in% 31:60, "31 - 60",
           ifelse(comp_stats$word_count %in% 61:90, "61 - 90",
           ifelse(comp_stats$word_count %in% 91:120, "91 - 120",
           ifelse(comp_stats$word_count %in% 121:150, "121 - 150",
                  "more than 150")))))) %>%
  count(quantity_words)

# User Annotations:
# Per user: Total number of ratings, distribution of ratings from 0 - 4 (bar chart, users on the X-axis)
gts <- germ_comp_clean %>%
  select(starts_with("A0"))

texts_with_label <- colSums(!is.na(gts))
texts_without_label <- colSums(is.na(gts))

gts_2 <- data.frame(
  texts_with_label, texts_without_label)

gts_2 <- gts_2 %>%
  arrange(desc(texts_with_label))

# Distribution of the number of ratings: texts with 4 - 11 ratings
range(germ_comp_clean$rating_count)
value_counts <- table(factor(germ_comp_clean$rating_count, levels = 4:11))

# Convert the table to a data frame
gts_3 <- as.data.frame(value_counts)

# Rename the columns for better clarity
names(gts_3) <- c("Value", "Frequency")

# Calculate the frequency of ratings for each column (0 - 4)
gts_4 <- lapply(gts, function(x) table(factor(x, levels = 0:4)))

# Create a data frame from the frequencies
gts_5 <- do.call(cbind, gts_4)
gts_5 <- as.data.frame(gts_5)
gts_5$Label <- rownames(gts_5)
gts_5 <- gts_5 %>%
  mutate(Sum = rowSums(.[1:10]))
```
![Sample Image](D:\BA_Github\gts_5.png "gts_5")

## 1.5 Visualizations of the Training Data

The following visualization provides an overview of users and the number of ratings per label.

```r
# Overview of users with the number of ratings per label
daten_long <- gts_5 %>%
  select(A001:A012, Label) %>%  # Select the first 11 columns and the ID
  pivot_longer(cols = -Label, names_to = "Column", values_to = "Value") # Reshape the data

ggplot(daten_long, aes(x = Column, y = Value, fill = as.factor(Label))) +
  geom_col(position = "stack", stat = "identity") +
  scale_fill_manual(values = c("#22763F", "#7AAF60", "#F5CA63", "#F17F4B", "#BE2A3E")) +
  theme_minimal() +
  labs(x = "User", y = "Number of Ratings", 
       title = "Overview of Rating Distribution per User") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  guides(fill = guide_legend(title = "Ratings"))
```
![Sample Image](D:\BA_Github\distribution_ratings_text.png "gts_5")
![Sample Image](D:\BA_Github\distribution_text_by_word_count.png "gts_5")
![Sample Image](D:\BA_Github\labeldistribution.png "gts_5")

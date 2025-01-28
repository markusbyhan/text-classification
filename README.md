# Automatic sexism detection in German-language texts 
I would like to provide you with a concise overview of the development, training, and fine-tuning of two machine learning algorithms. This work was conducted as part of my bachelor’s thesis in 2024.

### Introduction: What was the topic?

I focused on the development and evaluation of classifiers for detecting sexism and misogyny in online comments written in German, published in the comment section of an Austrian online newspaper. 
The challenge was to create an algorithm capable of identifying subtle and implicit sexism, which often avoids direct detection by steering clear of overtly offensive language. This research was based 
on an existing corpus annotated by humans, who assessed the presence and intensity of sexism and misogyny. A particular emphasis was placed on handling and interpreting differing opinions among annotators. 
The goal was to train robust models that could also reflect the diversity of opinions when evaluating new texts.

### Objective of the Thesis:
The aim was to develop classifiers for predicting a binary label that indicates the presence or absence of sexism. In the labeled training data, the binary label was derived using different strategies to 
consolidate the individual ratings of the annotators. For example, under the "bin_one" strategy, a text was classified as sexist if at least one annotator marked it as such. A text was only labeled as 
non-sexist if all annotators unanimously agreed on this classification. The severity of sexism, rated on a scale from 1 to 4, did not influence the binary classification.  
Additionally, the study aimed to investigate how majority opinions could be effectively utilized for classification. Using the labeled training data, features with significant importance relative to the 
entire corpus were extracted using **Term Frequency - Inverse Document Frequency (TF-IDF)**. Furthermore, the texts were analyzed for emotions and overall sentiment (positive/negative) through sentiment analysis, 
and these features were combined with the TF-IDF data. Based on this feature set, **SVM (Support Vector Machine)** and **Random Forest** models were trained and evaluated across all five annotation strategies.

### Basics:
I won’t go into detail on all the fundamentals of machine learning here, as that would make this overview less concise. Instead, I’ve listed some helpful links below to provide you with in-depth background knowledge:

### Fields of Machine Learning:
- [Types of Machine Learning Algorithms](https://towardsdatascience.com/types-of-machine-learning-algorithms-you-should-know-953a08248861)
- [Supervised vs Unsupervised Learning](https://www.ibm.com/topics/supervised-learning)

### Methods in Machine Learning:
- [An Overview of Machine Learning Algorithms](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)
- [10 Popular Machine Learning Algorithms](https://www.simplilearn.com/tutorials/machine-learning-tutorial/machine-learning-algorithms)

### Support Vector Machine:
- [Introduction to Support Vector Machines](https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/)
- [How SVM Works in Machine Learning](https://scikit-learn.org/stable/modules/svm.html)

### Random Forest:
- [Random Forest Algorithm Explained](https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd)
- [Random Forest in Machine Learning](https://www.geeksforgeeks.org/random-forest-in-machine-learning/)

## Setup:
Package installation

```r
# Installing Basic Packages
install.packages("jsonlite")    # Read JSON files
install.packages("dplyr")       # Data Manipulation
install.packages("gridExtra")   # Extras for tables
install.packages("utils")       # Utilities for R

# Installing Text Preprocessing Packages
install.packages("tidyverse")
install.packages("tidytext")    # Text mining
install.packages("tm")          # Text mining
install.packages("quanteda")    # Quantitative text data analysis
install.packages("SnowballC")   # Stemmer
install.packages("textstem")    # Stemming and Lemmatizing
install.packages("slam")        # Sparse lightweight arrays and matrices
install.packages("syuzhet")
install.packages("koRpus")      # Text analysis
install.koRpus.lang("de")       # German language package

# Installing Visualization Packages
install.packages("ggplot2")
install.packages("wordcloud2")
install.packages("RColorBrewer")

# Installing Machine Learning Packages
install.packages("text2vec")
install.packages("e1071")
install.packages("caret")
install.packages("caTools")
install.packages("randomForest")
```

## After installing: Package loading

```r
# Loading the Libraries
library(jsonlite)
library(dplyr)
library(gridExtra)

library(tidyverse)
library(tidytext)
library(tm)
library(quanteda)
library(SnowballC)
library(koRpus)
library(koRpus.lang.de)
library(textstem)
library(slam)

library(ggplot2)
library(wordcloud2)
library(RColorBrewer)

library(e1071)
library(caret)
library(caTools)
library(randomForest)
library(syuzhet)

library(lubridate)
library(stringr)
library(tidyr)
library(readr)
```

## Notes:
- The Basic Packages section includes utilities required for general data manipulation.
- The Text Preprocessing Packages focus on text mining and analysis tasks.
- The Visualization Packages provide tools for creating plots and visualizations.
- The Machine Learning Packages enable you to build and evaluate models effectively.

Use this code block in your R environment to install and load the necessary libraries.

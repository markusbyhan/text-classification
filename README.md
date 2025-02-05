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


# Containing files
## Testdata
- Competition Phase = germeval-competition-test.jsonl
- Development Phae =germeval-development-test.jsonl

## Trainingdata
- Competition Phase = germeval-competition-traindev.jsonl
- Development Phase  germeval-development-train.jsonl

## Setup and preprocessing
- Introduction = README.md
- Task description = task_prequisites.md
- Packages = package_setup.md
- Basis preprocessing I = preprocessing_training.md
- Basic preprocessing II = trainingset.md


## 5 Stategies for textclassification
- All Annotators = train_bin_all.md
- Majority Decision = train_bin_maj.md
- At least 1 Annotator = train_bin_one.md
- Disagreement = train_disagree_bin.md
- Multi-Majority = train_multi_maj.md

## Handling the missing values
- knn_imputation.md

## Sentimentanalysis with the "syuzhet"-Package
- sentimentanalysis.md

## Lemmatizing the Tokens with Treetagger
- treetagger.md

## Kombining TF-IDF and Sentimentanalysis
- tf_idf.md 

## Training and Evaluation of the models
- svm_training.md
- rf_training.md

## Tuning and Evaluation of the models
- svm_tuning.md
- rf_tuning.md

## Prediction of the Testdata
- prediction.md

## Bachelorthesis
- Kolloquium (german) = BA_Präsentation.pdf
- Thesis (german) = mwe-bachelor.pdf

## Important Notice

This repository is still under my review. After three-quarters of a year and 6,000 lines of code, I need to take a step back, consolidate some steps and variables, and re-evaluate the entire process.

### What’s the lesson here?
Structured coding moves you forward, while wild experimentation and constant changes do not...

In particular, the areas of **Treetagger, TF-IDF, sentiment analysis, and application to the test data** are not yet finalized but will be reviewed in the coming weeks.

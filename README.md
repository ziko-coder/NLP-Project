# NLP Project: Suicide Detection from Reddit Data
This NLP project focuses on detecting suicide-related content from text data collected from Reddit. The project involves various preprocessing techniques, TF-IDF vectorization, and machine-learning models for text classification.

## Dataset
The dataset used in this project is named "Suicide_Detection.csv" and is collected from Reddit. It contains text data related to suicide posts or discussions. The dataset is stored in a CSV file format and is loaded into a Pandas DataFrame for analysis and modeling.


## Project Overview
1. Data Overview:
  * Reads the dataset using Pandas and displays information about the DataFrame.
  * Samples are a subset of data for analysis.
  * Removes unnecessary columns and checks for null values and duplicates.
  * Prints the count of each class in the dataset.
2. Preprocessing:
  * Converts the text data to lowercase.
  * Removes punctuation and special characters from the text.
  * Removes stop words from the text using NLTK's stopwords list.
  * Tokenizes the text into individual words.
  * Performs stemming on the words using PorterStemmer from NLTK.
  * Saves the preprocessed dataset to a new CSV file.
3. TF-IDF Vectorization:
  * Prepares the preprocessed text data for machine learning by converting it into a numerical representation.
  * Uses scikit-learn's TfidfVectorizer to transform the text into a TF-IDF matrix.
4. Machine Learning:
  * Splits the data into training and testing sets.
  * Trains a VotingClassifier, which combines three Naive Bayes classifiers (GaussianNB, BernoulliNB, and MultinomialNB) using soft voting.
  * Evaluates the performance of the VotingClassifier on both training and testing sets.
  * Trains an SVM (Support Vector Machine) classifier using scikit-learn's SVC.
  * Evaluates the performance of the SVM model on both training and testing sets.
  * Saves the trained models using pickle for future use.
5. Example:
  * Provides an example function to preprocess new text inputs and make predictions using the trained SVM model.

## Dependencies
The following libraries and frameworks are used in this project:
* pandas    
* matplotlib
* seaborn
* nltk
* sklearn

## Conclusion
This NLP project demonstrates the process of preprocessing text data, converting it into numerical features using TF-IDF vectorization, and training machine learning models for text classification. The goal of the project is to detect suicide-related content from Reddit data. By following the steps in the Jupyter Notebook, you can analyze the dataset, train the models, and make predictions on new text inputs.
For more details, refer to the provided Jupyter Notebook file and the code comments within it.  

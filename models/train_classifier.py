import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
import sqlalchemy as db
import string
import sys
import pickle


def load_data(database_filepath):
    """
    Load Distaster Response data from the Disaster Response Database
    
    Input: 
        database_filepath: Database containing Disaster Response dataset
    Output:
          X - Disaster messages (text)
          Y - Dataframe. Categorisation of messages. Contains 36 Sparse Columns
              which are indicated with 1 or 0 depending on the message categorisation 
          category_names = List of the column names of the 36 Categories

    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("df",con=engine)
    df = df[df['related'].notna()]
    X = df['message'].values
    Y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    
    return X,Y,category_names


def tokenize(text):
    """
    Pre-Process text data into suitable formats for feature extraction and modelling 
    
    Input: 
         text: Messages from disaster responses. 
    Output: 
         clean_tokens: Normalized, Cleaned and Lemmatized Tokens from distater response messages 
    """
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))

    fil_tokens = [w for w in tokens if not w in stop_words]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in fil_tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Train ML model to predict category from messages.
    
    Output:
         model - Trained model that has been estimated using Pipeline Estimators
    """

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance 
    
    Input: 
         X_test - Test Features (Messages)
         Y_test - Test Target Variables (Categories)
         category_names - Target Names 
    Output: 
          classification report for the model for each category with the precision, recall and F1 score
    
    """
    y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))



def save_model(model, model_filepath):
    """
    Save .pkl object for the model 
    Input: 
         model - trained ML model 
         model_filepath - location for .pkl object
         
    Output: 
         A pickle object
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
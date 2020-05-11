# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


import pickle


def load_data(database_filepath):
    """
    Process: 
        Loading the data from the sqlite database, creating a pandas dataframe. 
        Defining X and y variables (from cleaned data) as well as category names
    
    Args:
        database_filepath (str): Filepath to sqlite database
        
    Returns:
        X, y, category_names 
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster', engine)
    # df_sample = df.sample(n=2620)
    X = df['message'].values
    y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns
    
    return X, y, category_names


def tokenize(text):
    """
    Process:
        This function tokenizes the text, to clean it and prepare it for the ML pipeline.
        The process involves tokenization, removal of stop words, stemming and lemmatization
    
    Args:
        text (str): The text to clean
        
    Returns:
        lemmed: Cleaned text ready for ML
    
    """
    # normalization
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    # tokenization
    words = word_tokenize(text)
    
    # stop words
    stop_words = stopwords.words("english")
    
    # remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    # words in their root fom
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed]
    
    return lemmed


def build_model():
    """
    Process:
        Building a new pipeline based on better result metrics (see jupyter notebook)
        A gridsearch optimized model is also provided but commented out. This is based
        on the original pipeline (RandomForestClassifier) instead of the actual returned 
        output (AdaBoostClassifier)
    
    Args:
        None
    
    Returns:
        pipeline2: Improved original pipeline (Model using AdaBoost classifier)
    """
# build the pipeline

#     pipeline = Pipeline([
#             ('vect', CountVectorizer(tokenizer=tokenize)),
#             ('tfidf', TfidfTransformer()),
#             ('clf', MultiOutputClassifier(RandomForestClassifier()))
#         ])
    
#     # using GridSearchCV to find better parameters to use
#     parameters = {'clf__estimator__n_estimators': [20, 50],
#                     'clf__estimator__criterion' :['gini', 'entropy'],
#                     'clf__estimator__max_depth' : [2, 5, None],
#                     }

#     cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2) 
#     cv.fit(X_train, y_train)
#     return cv

    # AdaBoost pipeline
    pipeline2 = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    return pipeline2


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Process: 
        Calculates and returns the test accuracy, precision, recall and F1 score for the optimal model
    
    Args:
        model (classification model): Model used
        X_test (dataframe): Dataframe of test features
        Y_test (dataframe): Dataframe of test target
        category_names (list): List of category names

    Returns:
        None
    """
    # using AdaBoostClassifier

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    
    return


def save_model(model, model_filepath):
    """
    Process:
        Saves the model into a pickle file
    
    Args:
        model (ML model): Fitted model
        model_filepath (str): Filepath to save model to
    
    Returns:
        None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)
        return


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
    

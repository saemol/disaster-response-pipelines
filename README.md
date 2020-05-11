# disaster-response-pipelines
## Installation

- Python 3 and Anaconda

- Flask
- Plotly
- Packages:
  - NLTK
  - Scikit-learn
  - Pickle
  - SQlite

## 

## Project Motivation

This project was part of an assignment in Udacity's Nanodegree program in Data Science. The goal is to analyze disaster response data from Figure Eight and to build a web app that performs classification on disaster messages. A machine learning pipeline algorithm has been used to perform the classification task.

## 

## Project components

##### 1. ETL

The first part of the data pipeline is the Extract, Transform, and Load process. Here, the dataset is loaded, cleaned, and then stored in a SQLite database. To load the data into an SQLite database, the pandas dataframe `.to_sql()` method has been used.



##### 2. ML Pipeline

Python script `train_classifier.py`, writes a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file



##### 3. Flask Web App

Outputs the classification results of the classification via an API. Also includes 3 plotly charts.

----------------------------------------------------------------------------------------------------------------------------------------------------------

### File Descriptions

ETL Pipeline Preparation.ipynb: Notebook where ETL is performed

ML Pipeline Preparation.ipynb: Notebook where pipeline is built and deployed

### Folder Descriptions

- models: includes Picke-file of classifier (classifier.pkl) and python file with ML pipeline (train_classifier.py)
- data: disaster data
- app: Flask web app



## Authors

- **Saeed Molavi** https://github.com/saemol

## 

## Results

AdaBoost results:

![AdaBoost](https://github.com/saemol/disaster-response-pipelines/blob/master/screenshots/AdaBoost%20results.png)



Classification:

![disaster](https://github.com/saemol/disaster-response-pipelines/blob/master/screenshots/disaster.png)



Plotly charts:![context](https://github.com/saemol/disaster-response-pipelines/blob/master/screenshots/context.png)

![distribution](https://github.com/saemol/disaster-response-pipelines/blob/master/screenshots/newplot.png)

## Acknowledgments

- Figure Eight for the data
- Udacity for the course
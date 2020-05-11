# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Process:
        Loading messages and categories, using pandas to merge them into a dataframe
    
    Args:
        messages_filepath (str): filepath to messages.csv
        categories_filepath (str): filepath to categories.csv
    
    Returns:
        df (dataframe): Pandas dataframe of messages and categories merged
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Process:
        Cleaning the dataframe and preparing it for the ML pipeline
    
    Args:
        df (dataframe)
    
    Returns:
        df (cleaned)
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # extract the category column names
    category_colnames = row.str.split('-').str[0].values.tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to 0s and 1s
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)  
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, sort=False)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    #change wrong value of 2.0 to 1.0 according to instructions from mentor
    df[df.related == 2] = 0
    
    return df


def save_data(df, database_filename):
    """
    Process:
        Save the clean dataset into an sqlite database
    
    Args:
        df (dataframe): The final and cleaned dataframe of responses
        database_filename (str): Filepath of stored sqlite database
    
    Returns:
        None
    """
    # export to SQL
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster', engine, index=False, if_exists='replace')
    return

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
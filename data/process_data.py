import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    input:
        messages_filepath: The path of messages data.
        categories_filepath: The path of categories data.
    output:

        df: The dataframe of merged data
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on="id")
    return df

def clean_data(df):
    '''
    input:
        df: The dataframe of merged data
    output:
        df: Cleaned dataset
    '''
    categories = df['categories'].str.split(";",expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda r : r[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda r : r[-1:])
        categories[column] = categories[column].astype(int)
    df = df.drop(['categories'],axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df[df.duplicated() == False]
    return df

def save_data(df, database_filename):
    '''
    input:
        df: The dataframe of merged data
        database_filename: The path where the data is stored to
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterDataTable', engine, index=False)  


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
import pandas as pd
import numpy as np
import re
import nltk
import sys
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



def load_data(database_filepath):
    '''
    input:
        database_filepath: The path of sql database
    output:
        X: Message as training data
        Y: Label
        category_names: Categorical names of labels
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterDataTable',engine)
    X = df.message.values
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = list(df.columns[4:])
    return X,Y,category_names

def tokenize(text):
    '''
    input:
        text: Message before tokenization.
    output:
        clean_tokens: A list of tokens after tokenization.
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    stop_words = stopwords.words("english")
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    input:
        None
    output:
        cv: Model after GridSearch
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'clf__estimator__n_estimators': [50, 100, 200],
              'clf__estimator__min_samples_split': [2, 3, 4],
              'clf__estimator__min_samples_leaf': [1, 2, 4],
              'clf__estimator__max_depth': [10, 20, None]
              }
    cv_model = GridSearchCV(pipeline, param_grid=parameters,return_train_score=False)
    return cv_model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    input:
       model: Trained model
       X_test: Message list of test data
       Y_test: Label of test data
       category_names: Category names
    '''
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    for i in range(Y_test.shape[1]):
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(category_names[i]))
        print(classification_report(Y_test.iloc[:,i], y_pred.iloc[:,i]))


def save_model(model, model_filepath):
    '''
    input:
        model: Trained model
        model_filepath: The path to save model
    '''
    joblib.dump(model, model_filepath)


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
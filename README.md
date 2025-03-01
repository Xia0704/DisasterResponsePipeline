# Disaster Response Pipeline Project

### Summary of the project
This project analyzes disaster data from Figure Eight to build a model for an API that classifies disaster messages to different categories.

### File Description
    .
    ├── app     
    │   ├── run.py                           # File to run app
    │   └── templates   
    │       ├── go.html                     
    │       └── master.html                   
    ├── data                   
    │   ├── disaster_categories.csv          # Categories data
    │   ├── disaster_messages.csv            # Messages data
    │   └── process_data.py                  # Data cleaning
    ├── models
    │   └── train_classifier.py             # ML training        
    └── README.md    



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

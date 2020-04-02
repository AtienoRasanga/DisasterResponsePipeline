# Disaster Response Pipeline Project

## Project Aim
This project aims to classify messages received during disasters into suitable categories for quick action by relevant organisations. 

## Project Description 
The project uses NLP (Natural Language Processing) methodologies to process text messages and then applies Machine Learning to 
classify the messages into the 36 available categories such as Food, Fire, Hospital etc.
 
The project used Random Forest to classify the preprocessed messaged into the 36 categories. 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

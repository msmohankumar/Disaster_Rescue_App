# Disaster Response Pipeline Project (Udacity - Data Science Nanodegree)

![intro](https://github.com/msmohankumar/Disaster_Rescue_App/assets/153971484/16b042bd-c631-437c-92c3-731a03ee306c)

[(https://github.com/msmohankumar/Disaster_Response_App/assets/153971484/1af397f7-7751-4353-b9be-0d0948c30ee6)](https://github.com/msmohankumar/Disaster_Rescue_App/blob/main/screenshots/intro.png)



## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting-started)
    - [Dependencies](#dependencies)
    - [Installing](#installing)
    - [Executing Program](#executing-program)
3. [Additional Material](#additional-material)
4. [Authors](#authors)
5. [License](#license)
6. [Acknowledgement](#acknowledgement)
7. [Screenshots](#screenshots)

## Description
This project is part of the Data Science Nanodegree Program by Udacity. It involves building a Natural Language Processing (NLP) model to categorize messages from real-life disaster events in real-time. The dataset consists of pre-labelled tweets and messages.

The project is divided into the following key sections:
- Processing data: Building an ETL pipeline to extract, clean, and store the data in a SQLite database.
- Building a machine learning pipeline: Training a classifier to categorize text messages into various categories.
- Running a web app: Displaying model results in real-time.

## Getting Started

### Dependencies
- Python 3.5+
- NumPy, SciPy, Pandas, Scikit-Learn
- NLTK
- SQLAlchemy
- Pickle
- Flask, Plotly

### Installing
Clone the git repository:

- Root Directory
    - data: Contains data files and data processing scripts.
        - disaster_categories.csv: Categories data file.
        - disaster_messages.csv: Messages data file.
        - DisasterResponse.db
        - process_data.py: ETL pipeline script to clean and process data.
    - models: Contains machine learning model scripts.
        - train_classifier.py: Script to train the classifier and save the model.
        - classifier.pkl
    - screenshots: Contains screenshots of the web app.
        - intro.png: Introduction screenshot.
        - sample_input.png: Sample input screenshot.
        - sample_output.png: Sample output screenshot.
        - main_page.png: Main page screenshot.
        - process_data.png: Process data screenshot.
        - train_classifier_data.png: Train classifier screenshot.
    - app: Contains the web application files.
        - run.py: Script to run the web app.
        - templates: HTML templates for the web app.
            - master.html: Main page template.
            - go.html: Classification result template.
    - README.md: Project README file.


### Executing Program
1. Run the ETL pipeline to clean data and store the processed data in the database:
    ```
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```

2. Run the ML pipeline to load data from the database, train the classifier, and save the classifier as a pickle file:
    ```
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```

3. Run the web app:
    ```
    python run.py
    ```
   Access the web app at:
 * Running on http://127.0.0.1:3001
 * Running on http://192.168.29.170:3001


## Authors
M S Mohan Kumar

## License
This project is licensed under the MIT License.

## Acknowledgement
- Udacity for providing an excellent Data Science Nanodegree Program.
  

## Screenshots
- Sample Input
  ![sample_input](https://github.com/msmohankumar/Disaster_Rescue_App/assets/153971484/0ed4d427-7aca-4f4b-a4c5-5ab6cdb19f39)
- Sample Output
![sample_output](https://github.com/msmohankumar/Disaster_Rescue_App/assets/153971484/7e7d9bc7-506f-4390-95a4-1970e2eefd81)
- Main Page
![main_page](https://github.com/msmohankumar/Disaster_Rescue_App/assets/153971484/578c9df0-5450-4ff7-b3dd-251a51bae3e6)
- Process Data
  ![Process_data](https://github.com/msmohankumar/Disaster_Rescue_App/assets/153971484/875546e3-3909-4f08-940a-f2e6046ac936)
- **Train Classifier without Category Level Precision Recall**
  ![Train_classifier_data](https://github.com/msmohankumar/Disaster_Rescue_App/assets/153971484/37e0d223-eaca-4f93-92fa-e27665712739)


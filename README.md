# Disaster Response Pipeline Project (Udacity - Data Science Nanodegree)

![intro](https://github.com/msmohankumar/Disaster_Response_App/assets/153971484/1af397f7-7751-4353-b9be-0d0948c30ee6)



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
  ![sample_input](https://github.com/msmohankumar/Disaster_Response_App/assets/153971484/f0275d3d-ed81-40f2-8943-2ee55dd9e28d)
- Sample Output
![sample_output](https://github.com/msmohankumar/Disaster_Response_App/assets/153971484/cb90b6d5-7a22-45d9-94a8-2d6d5e179500)
- Main Page
![main_page](https://github.com/msmohankumar/Disaster_Response_App/assets/153971484/8b1b8e8d-b3d7-453f-9035-18f595093604)
- Process Data
  ![process_data](https://github.com/msmohankumar/Disaster_Response_App/assets/153971484/365a9206-45ee-46a9-986f-0b61d4da132c)
- **Train Classifier without Category Level Precision Recall**
  ![Train_data](https://github.com/msmohankumar/Disaster_Response_App/assets/153971484/46599e89-c691-4b94-8a16-e58ee410d43c)


import numpy as np
from flask import Flask, request, render_template
import pickle
import csv

import logging
logging.basicConfig(filename='flask_debug.log', level=logging.DEBUG)

app = Flask(__name__)

# load the model
with open('models/model.pkl', 'rb') as model_file:
    vectorizer, classifier = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/predict', methods=['POST'])
def predict():
    mentee_input = [x for x in request.form.values()]
    features = vectorizer.transform(mentee_input)
    prediction = classifier.predict(features)

    # Extracting individual prediction values
    name = prediction[1]
    prof = ""
    email = ""

    # open csv, parse through rows to identify the line with the name
    # in that line, make an array of the values separated by commas
    # identify the profession and email

    csv_file_path = 'user_data.csv'
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == name:
                prof = row[2]
                email = row[1]
                prediction_array = row


    # identify similarities

    # clean a
    a = mentee_input
    for i in range(len(a)):
        a[i] = a[i].strip().lower()
    logging.debug("a: %s", a)

    # clean b
    b = prediction_array
    for i in range(len(b)):
        b[i] = b[i].strip().lower()
    logging.debug("b: %s", b)

    similarities = list(set(a) & set(b))
    logging.debug("Similarities: %s", similarities)

    # debugging statements
    logging.debug("Mentee Input: %s", mentee_input)
    logging.debug("Features: %s", features)
    logging.debug("Prediction Result: %s", prediction)
    logging.debug("Similarities: %s", similarities)

    return render_template('mentor-mentee.html', name=name, prof=prof, email=email, similarities = similarities[-1])

if __name__ == "__main__":
    app.run(debug=True)

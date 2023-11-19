import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# specify csv path
csv_file_path = 'user_data.csv'

# read csv file and extract text and labels
corpus = []
labels = []
with open(csv_file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        labels.append(row[0])  # name is in first column
        text = ' '.join(row[2:])  # description is in rest of the columns
        corpus.append(text)

# bag of words model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# train a classifier
classifier = MultinomialNB()
classifier.fit(X, labels)

# save the model
with open('models/model.pkl', 'wb') as model_file:
    pickle.dump((vectorizer, classifier), model_file)

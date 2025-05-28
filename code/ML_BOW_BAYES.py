from sklearn.feature_extraction.text import CountVectorizer
from static_data import *
from tools import load_data
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from memory_profiler import profile
from tools import result_report, get_date_time

def predict(sentences): 
    data_path1 = '../data/new_labeled_data1.txt'
    X_train, y_train = load_data(data_path1)
    # Load vectorizer object from local file
    vectorizer = joblib.load('vectorizer.pkl')
    
    # Transform training data
    X_train_bow = vectorizer.transform(X_train)
    
    # Create and train Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_bow, y_train)

    sentence_bow = vectorizer.transform(sentences)
    # Predict using the trained model
    predicted_labels = nb_classifier.predict(sentence_bow)
    return predicted_labels
    

# @profile
def ML_BOW_BAYES(train_data_path, test_data_path, result_dir, save_name):  # Training dataset path, test dataset path, result save directory, result save file name

    X_train, y_train = load_data(train_data_path)
    X_test, real_labels = load_data(test_data_path)

    # Load CountVectorizer object from local file
    vectorizer = joblib.load('vectorizer.pkl')

    # Transform training data
    X_train_bow = vectorizer.transform(X_train)

    # Transform test data
    X_test_bow = vectorizer.transform(X_test)

    # Create and train Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_bow, y_train)

    # Predict using the trained model
    predicted_labels = nb_classifier.predict(X_test_bow)

    save_path = result_dir + get_date_time() + '_' + save_name + '.xlsx'
    result_report("BoW/NB", save_path, predicted_labels, real_labels)
from sklearn.feature_extraction.text import TfidfVectorizer
from static_data import *
from tools import get_formatted_time, load_data
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from memory_profiler import profile
from tools import result_report, get_date_time

# @profile
def ML_TF_IDF_KNN(train_data_path, test_data_path, result_dir, save_name):

    X_train, y_train = load_data(train_data_path)
    X_test, real_labels = load_data(test_data_path)

    # Load TfidfVectorizer object from local file
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Transform training data
    X_train_tfidf = vectorizer.transform(X_train)

    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)

    # Create and train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_tfidf, y_train)

    # Predict using the trained model
    predicted_labels = knn.predict(X_test_tfidf)

    save_path = result_dir + get_date_time() + '_' + save_name + '.xlsx'
    result_report("TF-IDF/KNN", save_path, predicted_labels, real_labels)
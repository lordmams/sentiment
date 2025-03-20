import warnings
warnings.simplefilter('ignore')
import os
import pandas, unidecode, json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from cassandra.cluster import Cluster


class Main:


   def __init__(self):

        cluster = Cluster(['35.180.10.41'])
        session = cluster.connect('sentiment_analysis')
        
        # Get data from Cassandra
        rows = session.execute("SELECT avis, note FROM dataset")
        # Convert to pandas DataFrame
        data = [(row.avis, row.note) for row in rows]
        self.dataset = pandas.DataFrame(data, columns=['sentence', 'label'])
        
        # Verify that we got data
        if len(self.dataset) == 0:
            raise ValueError("No data was retrieved from the database")
        
        # Print the first few rows of data
        print("Sample of retrieved data:")
        print(self.dataset.head())
        print("\nTotal number of records:", len(self.dataset))
        print("\nLabel distribution:")
        print(self.dataset['label'].value_counts())

        self.vectorizer = None
        self.score = None
        self.model = None
        self.train()

   def train(self):
    # Separate dataset and expected output // SELECT avis FROM dataset
    sentences = self.dataset['sentence'].values
    # // SELECT note FROM dataset
    y = self.dataset['label'].values

    # Split datasets // Overfitting
    sentences_train, sentences_test, y_train, y_test =   train_test_split(sentences, y, test_size=0.25, random_state=1000)

    # Verctorization of training and testing data
    self.vectorizer = CountVectorizer()
    self.vectorizer.fit(sentences_train)
    X_train = self.vectorizer.transform(sentences_train)
    X_test  = self.vectorizer.transform(sentences_test)

    # Init model and fit it
    self.model = XGBClassifier(max_depth=2, n_estimators=30)
    self.model.fit(X_train, y_train)

   def predict(self, json_text):
    # predictions
    result = self.vectorizer.transform([unidecode.unidecode(json_text)])
    result = self.model.predict(result)

    if str(result[0]) == "0":
        sentiment = "NEGATIVE"

    elif str(result[0]) == "1":
        sentiment = "POSITIVE"

    return sentiment


if __name__ == "__main__":
   main = Main()
   print(main.predict("Depuis ce matin votre application ne marche pas, je n'arrive pas à déverrouiller ma voiture."))
   print(main.predict("Le service était excellent et le personnel très accueillant"))
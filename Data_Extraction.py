import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection as model_selection
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


class TrainMachine:

    @staticmethod
    def train_machine_model():
        df = pd.read_csv("C:\\Users\\m1055934\\Desktop\\sample.csv")

        num_true = len(df.loc[df.target == 1])
        num_false = len(df.loc[df.target == 0])
        print("Number of true cases: {0} ({1:2.2f}%)".format(num_true, (num_true / (num_true + num_false)) * 100))
        print("Number of false cases: {0} ({1:2.2f}%)".format(num_false, (num_false / (num_true + num_false)) * 100))

        feature_col_names = ['question_text']
        predicted_class_names = ['target']

        x = df[feature_col_names].values
        y = df[predicted_class_names].values
        split_test_size = 0.20


        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=split_test_size)

        print("{0:0.2f}% in training set".format((len(x_train) / len(df.index)) * 100))
        print("{0:0.2f}% in test set".format((len(x_test) / len(df.index)) * 100))

        vectorizer = TfidfVectorizer()
        train_vectors = vectorizer.fit_transform(x_train.ravel())
		
        test_vectors = vectorizer.transform(x_test.ravel())

        clf = MultinomialNB().fit(train_vectors, y_train.ravel())

        test_vector_trail = vectorizer.transform(trail_test)

        predicted = clf.predict(test_vectors)

        actual = np.array(y_test)

        count = 0

        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                count = count + 1

        print(accuracy_score(y_test, predicted))

        print(confusion_matrix(y_test, predicted))


if __name__ == "__main__":
    t_mac = TrainMachine()
    t_mac.train_machine_model()

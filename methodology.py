from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def logistic_regression(X_train, y_train, X_test, y_test):
    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(X_train, y_train)
    predictions = logreg.predict(X_test)
    print("Applying Logistic Regression:")
    print("Precision = {}".format(precision_score(y_test, predictions, average='macro')))
    print("Recall = {}".format(recall_score(y_test, predictions, average='macro')))
    print("Accuracy = {}".format(accuracy_score(y_test, predictions)))


def random_forest(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Applying Random Forest:")
    print("Precision = {}".format(precision_score(y_test, y_pred, average='macro')))
    print("Recall = {}".format(recall_score(y_test, y_pred, average='macro')))
    print("Accuracy = {}".format(accuracy_score(y_test, y_pred)))


def xgboost(X_train, y_train, X_test, y_test):
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)
    param = {
        'eta': 0.3,
        'max_depth': 3,
        'objective': 'multi:softprob',
        'num_class': 4}
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 25
    bst = xgb.train(param, xg_train, num_round, watchlist)
    pred = bst.predict(xg_test)
    best_pred = np.asarray([np.argmax(line) for line in pred])
    print("Applying XGBoost:")
    print("Precision = {}".format(precision_score(y_test, best_pred, average='macro')))
    print("Recall = {}".format(recall_score(y_test, best_pred, average='macro')))
    print("Accuracy = {}".format(accuracy_score(y_test, best_pred)))


def doc2vec(df):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df["review_clean"].apply(lambda x: x.split(" ")))]
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
    # Transform each document into a vector data
    doc2vec_df = df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    df = pd.concat([df, doc2vec_df], axis=1)
    return df


def tf_idf(df):
    tfidf = TfidfVectorizer(min_df=10)
    tfidf_result = tfidf.fit_transform(df["review_clean"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = df.index
    df = pd.concat([df, tfidf_df], axis=1)
    return df



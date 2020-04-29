import pandas as pd
import argparse
from methodology import logistic_regression, random_forest, xgboost, doc2vec, tf_idf
from sklearn.model_selection import train_test_split
from preprocess import preprocess_text


def main(csvfile):
    # Read data
    reviews_df = pd.read_csv(csvfile)
    # Removing NA's
    reviews_df = reviews_df.dropna()
    reviews_df = reviews_df.reset_index(drop=True)

    # Create labels
    # Divide Reviewer_Scores into four classes, 3 with score>7.5, 2 with score > 5, 2 with score >2.5, 0 with score <2.5
    reviews_df["Label"] = reviews_df["Reviewer_Score"].apply(lambda x:
                                                                     3 if x >7.5 else (
                                                                     2 if x>5 else(
                                                                     1 if x>2.5 else 0)))
    reviews_df = reviews_df[["Additional_Number_of_Scoring", "Average_Score", "Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts",
                            "Total_Number_of_Reviews_Reviewer_Has_Given","Negative_Review","Positive_Review","Label"]]
    # The whole dataset is too large and here only take 30% of the dataset
    reviews_df = reviews_df.sample(frac = 0.3, replace = False, random_state=42)

    # PART 1: Prediction without nlp feature
    print("Without NLP features:")
    # Feature selection
    features = ["Additional_Number_of_Scoring", "Average_Score", "Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given"]
    X_train, X_test, y_train, y_test = train_test_split(reviews_df[features], reviews_df["Label"], test_size=0.30, random_state=20)

    # Logistic Regression
    logistic_regression(X_train, y_train, X_test, y_test)

    # Random Forest
    random_forest(X_train, y_train, X_test, y_test)

    # XGBoost
    xgboost(X_train, y_train, X_test, y_test)

    # PART 2: Prediction with adding nlp features
    print("With NLP features:")
    # Append the positive and negative text reviews
    reviews_df["review"] = reviews_df["Negative_Review"] + reviews_df["Positive_Review"]
    # Remove 'No Negative' or 'No Positive' from text
    reviews_df["review"] = reviews_df["review"].apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))

    # Clean text data
    print("Start preprocessing textual columns...")
    reviews_df["review_clean"] = reviews_df["review"].apply(lambda x: preprocess_text(x))

    # Train a Doc2Vec model with text data
    print("Adding Doc2Vec...")
    reviews_df = doc2vec(reviews_df)

    # Add tf-idf columns
    print("Adding TF-IDF...")
    reviews_df = tf_idf(reviews_df)

    # Feature selection
    label = "Label"
    ignore_cols = [label, "review", "review_clean","Negative_Review","Positive_Review"]
    features_2 = [c for c in reviews_df.columns if c not in ignore_cols]
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(reviews_df[features_2], reviews_df["Label"], test_size=0.30, random_state=20)

    # Logistic Regression
    logistic_regression(X_train_2, y_train_2, X_test_2, y_test_2)

    # Random Forest
    random_forest(X_train_2, y_train_2, X_test_2, y_test_2)

    # XGBoost with nlp features
    xgboost(X_train_2, y_train_2, X_test_2, y_test_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvfile', type=argparse.FileType('r'), help='Input csv file', default="Hotel_Reviews.csv")
    args = parser.parse_args()

    main(args.csvfile)




## Project Details
##### By Ling Yang, Jingrong Tian

* This repository includes following files:
    - README
    - main.py: 
        
        Train models in two ways: with nlp features(doc2vec and tf-idf) and without nlp features, by three methodologies: logistic regression, random forest and XGBoost. Then evaluate the result by precision, recall and accuracy.
    - methodology.py: 
        
        Develop logistic regression, random forest and XGBoost. Develop Doc2Vec, TF-IDF.
    - preprocess.py: 
        
        Develop preprocess method for input textual reviews.
* Main libraries imported are:
    - pandas
    - sklearn
    - xgboost
    - gensim
    - nltk
* The dataset Hotel_Reviews.csv could be downloaded online from 
    - https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe
* Command to run the project:
    - `python main.py --csv_file Hotel_Reivews.csv`



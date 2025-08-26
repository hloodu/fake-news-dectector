## Fake News Detector

A machine learning web app that detects whether a news article is real or fake.

Built with Python, scikit-learn, and Streamlit. 
Also uses pandas, RandomForestClassifier, and joblib.
Data visualizations made with WordCloud, matplotlib, and seaborn.

Try the app [here](https://fake-news-dectector-hqvbgbfg2md5cbqx2fwvus.streamlit.app/)!

**How It Works:**
1. Uses a public data set of real and fake news ("True.csv" and "Fake.csv")
2. Cleans and vectorizes text with TF-IDF
3. Trains a Random Forest classifier for binary text classification
4. Streamlit interface allows users to paste text and see real/fake prediction

**Tested 3 different classifiers and compared confusion matrices to optimize model:**
- LogisticRegression: F1 Score = 0.98
- MultinomialNB: F1 Score = 0.93
- RandomForestClassifier: F1 Score = **1.00**
  
Since RandomForestClassifier had the highest F1 Score (which balances catching more fake news with preventing real news from being mislabelled), it's used in the final model.

**Dataset from Kaggle**: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

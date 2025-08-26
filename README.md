Fake News Detector

A machine learning web app that detects whether a news article is real or fake.

Built with Python, scikit-learn, and Streamlit. 
Also uses WordCloud, pandas, RandomForestClassifier, joblib, matplotlib, and seaborn.

Try the app here: 

How It Works:
1. Uses a public data set of real and fake news ("True.csv" and "Fake.csv")
2. Cleans and vectorizes text with TF-IDF
3. Trains a Random Forest classifier for binary text classification
4. Streamlit interface allows users to paste text and see real/fake prediction


Dataset from Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

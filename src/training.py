# combine and label data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from cleaning import clean_text

# load and label data
df_fake = pd.read_csv("data/Fake.csv")
df_real = pd.read_csv("data/True.csv")
df_fake['label'] = 1
df_real['label'] = 0
df = pd.concat([df_fake, df_real], ignore_index=True)       # merge dataframes and ensure indexes are correct
df = df.sample(frac=1, random_state=42).reset_index(drop=True)      # shuffle new dataframe, then reset indexes

df['cleaned'] = df['text'].apply(clean_text)    # clean data

# vectorize data (using inverse doc frequency)
vectorizer = TfidfVectorizer(stop_words = "english", max_features=5000)     # score each word based on relative importance
X = vectorizer.fit_transform(df['cleaned'])         # apply vectorizer to cleaned data; x = feature matrix
y = df['label']                 # set target variable (labels to predict)

# split data (20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# evaluate model
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save model, vectorizer
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
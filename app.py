import streamlit as st
import joblib
from src.cleaning import clean_text
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# get dataset for visuals
@st.cache_data
def load_data():
    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")
    fake['label'] = 1
    real['label'] = 0
    df = pd.concat([fake, real], ignore_index=True)
    return df

# load model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
df = load_data()

st.title("Fake News Detector")

# user input
st.header("Check an article:")
user_input = st.text_area("Paste a news article here:")

# use model to analyze input
if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label = "🟢 Real" if prediction == 0 else "🔴 Fake"
    st.markdown(f"### This article is: **{label}**")



st.header("Dataset Overview")

# graph class distribution
fig, ax = plt.subplots()
sns.countplot(data=df, x="label", ax=ax)
ax.set_xticklabels(["Real", "Fake"])
ax.set_title("Class Distribution (Real vs Fake)")
st.pyplot(fig)

df['cleaned'] = df['text'].apply(clean_text)


# create word clouds
fake_text = ' '.join(df[df['label'] == 1]['cleaned'])
real_text = ' '.join(df[df['label'] == 0]['cleaned'])

st.header("Word Clouds: Most Common Words")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fake News")
    fake_wc = WordCloud(width=300, height=200, background_color='white').generate(fake_text)
    st.image(fake_wc.to_array())

with col2:
    st.subheader("Real News")
    real_wc = WordCloud(width=300, height=200, background_color='white').generate(real_text)
    st.image(real_wc.to_array())


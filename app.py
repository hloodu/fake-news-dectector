import streamlit as st
import joblib
from src.cleaning import clean_text

st.title("Fake News Detector")

# load model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# user input
st.subheader("Check an article:")
user_input = st.text_area("Paste a news article here:")

# use model to analyze input
if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label = "real" if prediction == 0 else "fake"
    st.markdown(f"### This article is **{label}**")

# show visuals
st.divider()
st.header("Explore the Training Data")

st.subheader("Article Breakdown")
st.image("./images/datachart1.png")

st.subheader("Most Common Words:")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Fake News")
    st.image("./images/fake1.jpg")
with col2:
    st.subheader("Real News
    st.image("./images/real1.jpg")


# # code used to generate images: 

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from wordcloud import WordCloud

# # get dataset for visuals
# @st.cache_data
# def load_data():
#     fake = pd.read_csv("data/Fake.csv")
#     real = pd.read_csv("data/True.csv")
#     fake['label'] = 1
#     real['label'] = 0
#     df = pd.concat([fake, real], ignore_index=True)
#     return df

# df = load_data()

# # graph class distribution
# fig, ax = plt.subplots()
# sns.countplot(data=df, x="label", ax=ax)
# ax.set_xticklabels(["Real", "Fake"])
# ax.set_title("Class Distribution (Real vs Fake)")
# st.pyplot(fig)

# df['cleaned'] = df['text'].apply(clean_text)

# # create word clouds
# fake_text = ' '.join(df[df['label'] == 1]['cleaned'])
# real_text = ' '.join(df[df['label'] == 0]['cleaned'])

# st.header("Word Clouds: Most Common Words")

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown("**Fake News**")
#     fake_wc = WordCloud(width=300, height=200, background_color='white').generate(fake_text)
#     st.image(fake_wc.to_array())

# with col2:
#     st.markdown("**Real News**")
#     real_wc = WordCloud(width=300, height=200, background_color='white').generate(real_text)
#     st.image(real_wc.to_array())
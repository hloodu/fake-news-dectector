import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # remove punctuation
    return text
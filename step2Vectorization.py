

from sklearn.feature_extraction.text import TfidfVectorizer

def vecTfid(text):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(text)


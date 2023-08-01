import numpy as np
import pandas as pd
import regex as re
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def processHashtag(text):
    return re.sub(r'#\w+', '',text)

def removeSpecialCharsPonct(text):
    return re.sub(r'[^\w\s]', '', text)

def normalizeText(text):
    return text.lower()

def stopWordsRemove(text):
    words = text.split()
    words = [ word for word in words if word.lower() not in stopwords ]
    text = ' '.join(words)
    return text

def lemmatizationWord(text):
    lematizer = WordNetLemmatizer()
    words = word_tokenize(text)
    text_lematized = " ".join([lematizer.lemmatize(word, get_wordnet_pos(word)) for word in words])
    return text_lematized

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def pipelineClean(text):
    text = processHashtag(text)
    text = removeSpecialCharsPonct(text)
    text = normalizeText(text)
    text = stopWordsRemove(text)
    text = re.sub(r'\b(?:user|url)\b', '', text, flags=re.IGNORECASE).strip()
    text = lemmatizationWord(text)
    return text


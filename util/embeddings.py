import numpy as np
from util.constants import PATH_TO_GLOVE_FILE
import gensim.downloader as api

# Worked on by Tom and Devynn

# Load in GloVe pre-trained embeddings
def load_glove():
    embeddings_index = {}
    with open(PATH_TO_GLOVE_FILE, encoding='utf8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    return embeddings_index

# Load in Word2Vec pre-trained embeddings
def load_word2vec_pretrained():
    embeddings = api.load("word2vec-google-news-300")
    return embeddings


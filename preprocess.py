from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def preprocess_text(text, stem=False):
    """Preprocess one sentence: tokenizes, lowercases, applies the Porter stemmer,
     removes punctuation tokens and stopwords.
     Returns a list of strings."""
    stops = set(stopwords.words('english'))
    toks = word_tokenize(text)
    if stem:
        stemmer = PorterStemmer()
        toks = [stemmer.stem(tok) for tok in toks]
    toks_nopunc = [tok for tok in toks if tok not in string.punctuation]
    toks_nostop = [tok for tok in toks_nopunc if tok not in stops]
    return " ".join(toks_nostop)



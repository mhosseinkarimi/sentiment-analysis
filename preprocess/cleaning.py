import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from typing import List

def remove_punct(text: str) -> str:
    """Removes punctuation symbols from text.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
        Processed string
    """
    return re.sub('[^a-zA-z0-9\s]', '', text)

def remove_stopwords(text: str) -> str:
    """Removes words with high frequency in text but low semantic values.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
        Processed text
    """
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words])

def remove_prefix(text: str) -> str:
    """Removes some common twitter prefixes such as "rt" as a sign of the post being retweeted.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
        Processed text
    """
    if text.startswith("rt"):
        text = text[2:]
    return text

def convert_labels_to_numerical(label: str, classes: List[str]) -> int:
    """Converts the string categorical label to a numerical label.

    Parameters
    ----------
    label : str
        Categorical label
    classes : List[str]
        List of categories

    Returns
    -------
    int
        Numerical label
    """
    return classes.index(label)
from datasets import load_dataset
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# if you haven’t already downloaded the punkt models, run this once:
nltk.download('punkt')
nltk.download('stopwords')
def load_imdb_dataset():
    """
    Load the IMDB train/test splits from Hugging Face Datasets.
    Returns:
      train_texts, train_labels, test_texts, test_labels
    """
    ds = load_dataset("imdb")
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts  = ds["test"]["text"]
    test_labels = ds["test"]["label"]
    return train_texts, train_labels, test_texts, test_labels

# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    
    #Clean a raw text string: lowercasing, removing HTML, URLs, unwanted chars.
    #Extend this with your existing tokenization, stopword removal, etc.
    
    import re
    # lowercase
    text = text.lower()
    # remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # remove non-word characters (keep basic punctuation)
    text = re.sub(r'[^\w\s\.!?]', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_texts(texts: list[str]) -> list[str]:
    """
    Apply preprocess_text to a list of raw strings.
    """
    return [preprocess_text(t) for t in texts]

# ─── Vectorization ────────────────────────────────────────────────────────────

def vectorize_texts(processed_texts: list[str], vectorizer_path: str = None):
    #Args:
      #processed_texts: list of cleaned text strings.
      #vectorizer_path: path to load/save a pickle of TfidfVectorizer.
    #Returns:
      #X: numpy array of shape (n_samples, n_features)
      #vectorizer: fitted TfidfVectorizer
    
    if vectorizer_path and os.path.exists(vectorizer_path):
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        X = vectorizer.transform(processed_texts)
    else:
        vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1,2),
            smooth_idf=True,
            lowercase=False  # already lowercased in preprocess_text
        )
        X = vectorizer.fit_transform(processed_texts)
        if vectorizer_path:
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
    return X.toarray(), vectorizer


def load_and_preprocess_imdb(vectorizer_path: str = None):
   

    #Args:
      #vectorizer_path: optional path for saving/loading the vectorizer.
    #Returns:
     # X_train, y_train, X_test, y_test, vectorizer

    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset()
    processed_train = preprocess_texts(train_texts)
    processed_test  = preprocess_texts(test_texts)

    X_train, vectorizer = vectorize_texts(processed_train)
    X_test  = vectorizer.transform(processed_test).toarray()

    # now vectorizer is in memory; if you passed a path and want to overwrite it:
    if vectorizer_path:
         with open(vectorizer_path, "wb") as f:
             pickle.dump(vectorizer, f)

    return X_train, train_labels, X_test, test_labels, vectorizer









def preprocess_text(text):
    text = text.lower() # covert all text to lowercase
    tokens = word_tokenize(text) #split the sentances into each word, I.E tokens
    tokens = [word for word in tokens if word.isalnum() or word == "!" and word not in stopwords.words("english")] #remove any punctuation with word.isalum, and then remove the english stopwords. 
    #we also want to retain exclamation marks because they're indicitive of strong sentiment, later we can use them to increase the intensity of sentance. I.E ! means postive or negative, NOT NEUTRAL
    return " ".join(tokens) #join each token backtogether with a space between each one

#now we need to turn the data into a vector
#we do this using the TfidVectorizer or Term Frequency-inverse Document Frequency, which converts text into numbers based on how important each word is in the dataset
#There are two steps to this, term frequency or TF, which counts how often a word appears in a sentance
#And Inverse Document Frequency, which reduces the importance of common words by dividing their frequency accross all texts

#The values in the vector will represent the IMPORTANCE of a word relative to the corpus
#We will also be converting our vectorized text to tensors. This is an ESSENTIAL STEP (maybe elaborate on this more)
def vectorize_texts(texts):
    vectorizer = TfidfVectorizer(max_features=100) #Only keep the 100 most important words

    X = vectorizer.fit_transform(texts).toarray()

    return X, vectorizer












#run some tests to ensure that this func works
if __name__ == "__main__":
    sample_texts = [
        "I love this product! It's amazing!", 
        "This is the worst thing I've ever bought...", 
        "It's okay, not the best, but not the worst!"
    ]

    print("Original Texts:")
    for text in sample_texts:
        print(text)

    processed_texts = [preprocess_text(text) for text in sample_texts]

    print("\nProcessed Texts:")
    for text in processed_texts:
        print(text)

    X, vectorizer = vectorize_texts(processed_texts)

    print("\nVectorized Texts (Numerical Format):")
    print(X)  # Prints the numerical representation of the texts

    print("\nFeature Names (Words Used in Vectorization):")
    print(vectorizer.get_feature_names_out())  # Prints the words used in vectorization

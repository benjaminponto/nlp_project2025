#Text preprocessing and vectorization
#nltk is natural language ToolKit, we will use this for text processing
import nltk

from nltk.tokenize import word_tokenize
#get the tokenize lib. This is how we will be splitting the sentances in our dataset
from nltk.corpus import stopwords
#stopwords are common words like "the", "is" that dont carry much meaning
from sklearn.feature_extraction.text import TfidfVectorizer
#TfidfVectoizer converts text into numerical features

nltk.download("stopwords")

nltk.download('punkt_tab') #the punkt model didves a text into a list of sentances 

def preprocess_text(text):
    text = text.lower() # covert all text to lowercase
    tokens = word_tokenize(text) #split the sentances into each word, I.E tokens
    tokens = [word for word in tokens if word.isalnum() or word == "!" and word not in stopwords.words("english")] #remove any punctuation with word.isalum, and then remove the english stopwords
    return " ".join(tokens) #join each token backtogether with a space between each one

if __name__ == "__main__":
    sample_texts = [
        "I love this product! It's amazing.", 
        "This is the worst thing I've ever bought...", 
        "It's okay, not the best, but not the worst."
    ]

    print("Original Texts:")
    for text in sample_texts:
        print(text)

    processed_texts = [preprocess_text(text) for text in sample_texts]

    print("\nProcessed Texts:")
    for text in processed_texts:
        print(text)

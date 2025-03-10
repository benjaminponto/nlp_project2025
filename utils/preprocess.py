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
    tokens = [word for word in tokens if word.isalnum() or word == "!" and word not in stopwords.words("english")] #remove any punctuation with word.isalum, and then remove the english stopwords. 
    #we also want to retain exclamation marks because they're indicitive of strong sentiment, later we can use them to increase the intensity of sentance. I.E ! means postive or negative, NOT NEUTRAL
    return " ".join(tokens) #join each token backtogether with a space between each one

#now we need to turn the data into a vector
#we do this using the TfidVectorizer or Term Frequency-inverse Document Frequency, which converts text into numbers based on how important each word is in the dataset
#There are two steps to this, term frequency or TF, which counts how often a word appears in a sentance
#And Inverse Document Frequency, which reduces the importance of common words by dividing their frequency accross all texts

#The values in the vector will represent the IMPORTANCE of a word relative to the corpus
def vectorize_texts(texts):
    vectorizer = TfidfVectorizer(max_features=100) #Only keep the 100 most important words
    X = vectorizer.fit_transform(texts).toarray(), vectorizer
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

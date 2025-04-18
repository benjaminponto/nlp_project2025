import torch
from torch.utils.data import DataLoader, TensorDataset
from models.model import initialize_model, train_model
from utils.preprocess import vectorize_texts, preprocess_text

# create a sample dataset with 
sample_texts = [
    "I love this product! It's amazing!", 
    "This is the worst thing I've ever bought...", 
    "It's okay, not the best, but not the worst!"
]


#step 2 preprocess the texts 
processed_texts = [preprocess_text(text) for text in sample_texts]

# vectorize texts, i belive it retuirns a matrix of word counts and the vectorizer object
# the matrix is a 2D array where each row represents a text and each column represents a word in the vocabulary.
X, vectorizer = vectorize_texts(processed_texts)

# now we have to turn out vectors into tensors
X_train = torch.tensor(X, dtype=torch.float32)  # Feature Tensor
y_train = torch.tensor([0, 1, 2], dtype=torch.long)  # Labels (0 = Positive, 1 = Negative, 2 = Neutral)

# load the data? not sure why come back to this later
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)    

# specify our model 
input_size = X.shape[1]  # Number of features (words) after vectorization
hidden_size = 128  # Hidden layer neurons
output_size = 3  # Sentiment categories: Positive, Negative, Neutral

# call our intialize_model function to create our model, loss function, and optimizer
model, loss_fn, optimizer = initialize_model(input_size, hidden_size, output_size)


#this will return 5 epoch values and the loss for each epoch
train_model(model, loss_fn, optimizer, train_loader, num_epochs=5)

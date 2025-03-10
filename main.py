import torch
from torch.utils.data import DataLoader, TensorDataset
from models.model import initialize_model, train_model
from utils.preprocess import vectorize_texts, preprocess_text

# Step 1: Prepare Sample Raw Text Data
sample_texts = [
    "I love this product! It's amazing!", 
    "This is the worst thing I've ever bought...", 
    "It's okay, not the best, but not the worst!"
]

# Step 2: Preprocess the Texts
processed_texts = [preprocess_text(text) for text in sample_texts]

# Step 3: Vectorize the Processed Texts
X, vectorizer = vectorize_texts(processed_texts)

# Step 4: Convert to PyTorch Tensors
X_train = torch.tensor(X, dtype=torch.float32)  # Feature Tensor
y_train = torch.tensor([0, 1, 2], dtype=torch.long)  # Labels (0 = Positive, 1 = Negative, 2 = Neutral)

# Step 5: Create a PyTorch DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Step 6: Define Model Parameters
input_size = X.shape[1]  # Number of features (words) after vectorization
hidden_size = 128  # Hidden layer neurons
output_size = 3  # Sentiment categories: Positive, Negative, Neutral

# Step 7: Initialize the Model, Loss Function, and Optimizer
model, loss_fn, optimizer = initialize_model(input_size, hidden_size, output_size)

# Step 8: Train the Model
train_model(model, loss_fn, optimizer, train_loader, num_epochs=5)

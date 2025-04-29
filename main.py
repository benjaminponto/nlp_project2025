import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from models.model import initialize_model, train_model
from utils.preprocess import load_and_preprocess_imdb
from utils.predict import predict_sentiment

# 1) Load & preprocess IMDB data (features, labels, and vectorizer)
X_train_np, y_train, X_test_np, y_test, vectorizer = load_and_preprocess_imdb(
    vectorizer_path="vectorizer.pkl"
)

# 2) Convert NumPy arrays to torch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test  = torch.tensor(X_test_np, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.long)

# 3) Create DataLoaders
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),   batch_size=batch_size, shuffle=False)

# 4) Initialize and train the model
input_size  = X_train.shape[1]  # number of TF-IDF features
hidden_size = 128
output_size = 2  # IMDB: positive vs. negative
model, loss_fn, optimizer = initialize_model(input_size, hidden_size, output_size)
train_model(model, loss_fn, optimizer, train_loader, num_epochs=25)

# 5) Evaluate on the test split
model.eval()
test_loss, test_acc = train_model(
    model, loss_fn, optimizer,
    test_loader,
    evaluate=True
)
print(f"Test loss: {test_loss:.3f} | acc: {test_acc:.2f}%")

# 6) Save the trained model and vectorizer
torch.save(model.state_dict(), "saved_model.pth")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset
from models.model import initialize_model, train_model
from utils.preprocess import load_and_preprocess_imdb
from utils.predict import predict_sentiment
from datasets import load_dataset
from collections import Counter

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
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True  # shuffle training data to improve generalization
)
test_loader  = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=batch_size,
    shuffle=False
)

# 4) Initialize and train the model
input_size  = X_train.shape[1]  # number of TF-IDF features
hidden_size = 128
output_size = 2  # IMDB: positive vs. negative
model, loss_fn, optimizer = initialize_model(input_size, hidden_size, output_size)
train_model(model, loss_fn, optimizer, train_loader, num_epochs=25)

# 5) Prepare for inference on the test set
imdb       = load_dataset("imdb")
test_texts = imdb["test"]["text"]
device     = next(model.parameters()).device
model.eval()

# 6) Gather predictions and labels
all_preds, all_labels = [], []
with torch.no_grad():
    for raw, x_vec, actual in zip(test_texts, X_test, y_test):
        x = x_vec.unsqueeze(0).to(device)
        logits = model(x)
        pred   = logits.argmax(dim=1).item()
        all_preds.append(pred)
        all_labels.append(actual.item())
        # print(f"Text: {raw}")  # uncomment to see each sentence
        print(f"→ Predicted: {pred}    |    Actual: {actual.item()}")

# 7) Display class distributions
print("Predicted distribution:", Counter(all_preds))
print("True label distribution:   ", Counter(all_labels))
# 7) Compute and display accuracy
correct = sum(p == a for p, a in zip(all_preds, all_labels))
total = len(all_labels)
accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")

# 8) Save the trained model and vectorizer
torch.save(model.state_dict(), "saved_model.pth")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

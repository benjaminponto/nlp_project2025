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

# Track training loss
import matplotlib.pyplot as plt
all_losses = []
num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    all_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the loss curve
plt.figure(figsize=(8, 6))
plt.plot(all_losses, marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
print("Saved training curve as 'loss_curve.png'")

# 5) Evaluate on the test split
model.eval()
total_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for X, y in test_loader:
        outputs = model(X)
        loss = loss_fn(outputs, y)
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

avg_test_loss = total_loss / len(test_loader)
test_accuracy = 100 * correct / total
print(f"Test loss: {avg_test_loss:.3f} | acc: {test_accuracy:.2f}%")

# 6) Save the trained model and vectorizer
torch.save(model.state_dict(), "saved_model.pth")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

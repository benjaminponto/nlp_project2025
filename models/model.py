#Define the nural network model here
import torch # touch is a deep learning libary where we will be accessing functions 
import torch.nn as nn 
import torch.optim as optim

#STEP 1: Outline and define the Neural Network Model. In our case, we will have three layers. 2 fully connected layers, and 1 hidden layer 
class SentimentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SentimentClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size) #first layer, connects all the input features to the hidden layer.

        self.relu = nn.ReLU() # hidden layer, adds non-linearity with the ReLU activation function. This allows the model to learn complex patterns in the data. 

        self.fc2 = nn.Linear(hidden_size, output_size) # second layer, takes hidden layers output and produces the final output

    def forward(self, x):
        x = self.fc1(x) # pass input through first layer
        x = self.relu(x) # ReLU with hidden layer
        x = self.fc2(x) #pass through last layer which in the next epoch, will be the first layers input

        return x



#STEP 2: Intialize model, create the loss function, and create the optimizer
def initialize_model(input_size, hidden_size, output_size):
    model = SentimentClassifier(input_size, hidden_size, output_size)

    loss_fn = nn.CrossEntropyLoss() # calculates the difference between the predicted class probabilities and the actual class labels. it shows how  good our model is.
    #still dont fully understand this 

    #Adaptive Moment Estimation (ADAM), does a bunch of cool math that adjusts how much our models weights are adjusted during each interation
    #Our lr (learning rate) is set to the default value of 0.001. Changing this will affect the rate at which our weights are adjusted
    #The lr will NEED to be changed and tested with to find the most effective value.
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    return model, loss_fn, optimizer

#recall, an epoch is a pass of the entire training dataset through our algorithm
def train_model(model, loss_fn, optimizer, data_loader, num_epochs=25, evaluate=False):
    # pick up whatever device the model is on
    device = next(model.parameters()).device

    if evaluate:
        model.eval()  # no weight updates
        total_loss, total_correct = 0.0, 0
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                total_loss   += loss_fn(out, y).item() * X.size(0)
                total_correct += (out.argmax(1) == y).sum().item()
        N = len(data_loader.dataset)
        return total_loss / N, total_correct / N * 100

    model.train()
    for epoch in range(1, num_epochs+1):
        epoch_loss = 0.0
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)  # send batch to device
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss/len(data_loader):.4f}")

    return model


                                                   

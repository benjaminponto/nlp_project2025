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
def train_model(model, loss_fn, optimizer, train_loader, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            
            #Zeros the gradients before we backpropagate. if we dont,  PyTorch will accumulate the gradients on each pass, which can disrupt the values
            optimizer.zero_grad()

           
            outputs = model(inputs) #forward pass

            loss = loss_fn(outputs, labels) #Compute loss
            loss.backward() #backpropogate
            optimizer.step() # update the weights

            running_loss += loss.item() #Accumulate loss

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
        
    return model

                                                   

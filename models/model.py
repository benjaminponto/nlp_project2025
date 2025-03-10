#Define the nural network model here
import torch # touch is a deep learning libary where we will be accessing functions 
import torch.nn as nn 
import torch.optim as optim

#STEP 1: Outline and define the Neural Network Model. In our case, we will have three layers. 2 fully connected layers, and 1 hidden layer 
class SentimentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SentimentClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size) #first layer, connects all the input features to the hidden layer.

        self.relu = nn.ReLU() # hidden layer, adds non-linearity, we need this because this allows us to model non-linear relationships. this sort of acts like a sigmoid function 

        self.fc2 = nn.Linear(hidden_size, output_size) # second layer, takes hidden layers output and produces the final output

    def forward(self, x):
        x = self.fc1(x) # pass input through first layer
        x = self.relu(x) # ReLU with hidden layer
        x = self.fc2(x) #pass through second layer, this gets output

        return x




def initialize_model(input_size, hidden_size, output_size):
    model = SentimentAnalyzer(input_size, hidden_size, output_size)

    loss_fn = nn.CrossEntropyLoss() # calculates the difference between the predicted class probabilities and the actual class labels. I.E shows how  good our model is.

    #Adaptive Moment Estimation (ADAM), does a bunch of cool math that adjusts how much our models weights are adjusted during each interation
    #Our lr (learning rate) is set to the default value of 0.001. Chaning this will affect the rate at which our weights are adjusted
    #The lr will NEED to be changed and tested with to find the most effective value.
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    return model, loss_fn, optimizer

#recall, an epoch is a pass of the entire training dataset through our algorithm
def train_model(model, loss_fn, optimizer, train_loader, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            
            #Zeros the gradients before we backpropagate. if we dont,  PyTorch will accumulate the gradients on each pass
            optimizer.zero_grad()

            #forward pass
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

                                                   

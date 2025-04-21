#predict functionality here 
import torch
import torch.nn.functional as F
from models.model import SentimentClassifier
from utils.preprocess import preprocess_text
import pickle



def load_model(model_path, input_size, hidden_size, output_size):
    model = SentimentClassifier(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def prepare_input(text, vectorizer):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Vectorize the text using the same vectorizer used during training
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    
    # Convert to tensor
    input_tensor = torch.tensor(vectorized_text, dtype=torch.float32)
    
    return input_tensor

def predict_sentiment(model, text, vectorizer):
    input_tensor = prepare_input(text, vectorizer)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()  # Get the predicted class index

    labels = ["Positive", "Neutral", "Negative"]
    return labels[predicted_class], probabilities[0][predicted_class].item()  # Return class label and confidence score
    
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
from collections import deque
from timesformer.models.vit import TimeSformer  # Ensure this is accessible

# Define the model architecture (must match training)
class TimeSformerLSTMModel(nn.Module):
    def __init__(self, img_size, num_classes, num_frames, lstm_hidden_size=512):
        super(TimeSformerLSTMModel, self).__init__()
        self.timesformer = TimeSformer(
            img_size=img_size, 
            num_classes=0,  # To get feature embeddings
            num_frames=num_frames, 
            attention_type='divided_space_time',
             # Set to False as we'll load your own pretrained weights
        )
        self.lstm = nn.LSTM(
            input_size=768, 
            hidden_size=lstm_hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
    
    def forward(self, x):
        x = self.timesformer(x)
        
        # Ensure correct shape for LSTM
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        elif len(x.shape) == 3 and x.shape[2] != 768:
            x = x.view(x.size(0), -1, 768)  # (batch_size, num_frames, 768)
    
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Use the last LSTM output
        x = self.fc(x)
        return x

def load_class_names(class_names_path):
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def load_model(checkpoint_path, class_names, device):
    num_classes = len(class_names)
    num_frames = 8
    img_size = 224
    
    model = TimeSformerLSTMModel(
        img_size=img_size,
        num_classes=num_classes,
        num_frames=num_frames
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_frame(frame, transform):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = transform(frame)
    return frame

def main():
    # Paths
    model_folder_path = r'C:\Users\malle\Desktop\asl_timesformer\modeltest'
    checkpoint_path = os.path.join(model_folder_path, 'best_asl_alphabet_timesformer_lstm.pth')
    class_names_path = os.path.join(model_folder_path, 'class_names.json')
    
    # Load class names
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"Class names file not found at {class_names_path}.")
    class_names = load_class_names(class_names_path)
    print(f"Loaded class names: {class_names}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    model = load_model(checkpoint_path, class_names, device)
    print("Model loaded successfully.")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize if applied during training
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])
    
    # Initialize video capture
    cap = cv2.VideoCapture(1)  # 0 is the default camera
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize a buffer to hold the last num_frames frames
    num_frames = 8
    frame_buffer = deque(maxlen=num_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Preprocess the frame and add to buffer
        processed_frame = preprocess_frame(frame, transform)
        frame_buffer.append(processed_frame)
        
        # Only perform prediction if buffer is full
        if len(frame_buffer) == num_frames:
            # Stack frames: (num_frames, C, H, W)
            input_tensor = torch.stack(list(frame_buffer))
            # Add batch dimension and permute to (1, C, T, H, W)
            input_tensor = input_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                predicted_class = class_names[predicted_idx.item()]
                confidence = confidence.item()  # Keep as a decimal (0.0 - 1.0)
            
            # Apply confidence threshold
            if confidence >= 0.1:
                # Convert confidence to percentage for display
                confidence_pct = confidence * 100
                # Overlay prediction on the frame
                cv2.putText(frame, f"Prediction: {predicted_class} ({confidence_pct:.2f}%)", 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 255, 0), 
                            2, 
                            cv2.LINE_AA)
            else:
                # Indicate low confidence
                cv2.putText(frame, "Prediction: Uncertain", 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 0, 255),  # Red color for uncertainty
                            2, 
                            cv2.LINE_AA)
            
        # Display the frame
        cv2.imshow('ASL Alphabet Recognition', frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from torch.nn.functional import softmax
import time
import warnings

# Suppress warnings from torchvision and other libraries
warnings.filterwarnings("ignore", category=UserWarning)

# Path to the saved model file on your PC
model_file_path = r'C:\Users\malle\Desktop\asl_timesformer\gesture_recognition_model.pth'

# Updated ASL class labels
class_labels = [
    "1", "10", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", 
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "excuse me", "friend", "help", "no", 
    "rest position", "sorry", "stop", "thank you"
]

# Confidence threshold
confidence_threshold = 0.2  # Only display predictions with confidence >= 40%
sequence_length = 30  # Number of frames in each sequence

# Define Frame Feature Extractor
class FrameFeatureExtractor(nn.Module):
    def __init__(self):
        super(FrameFeatureExtractor, self).__init__()
        cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(cnn.children())[:-1])  # Remove the final classification layer

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)  # Flatten batch and time dimensions
        features = self.feature_extractor(x)
        features = features.view(b, t, -1)  # Reshape back to (batch, time, features)
        return features

# Define Temporal Transformer-based Gesture Recognition Model
class GestureRecognitionModel(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(GestureRecognitionModel, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.fc = nn.Linear(feature_dim, num_classes)  # Final classification layer

    def forward(self, x):
        x = self.transformer_encoder(x)  # Process the frame features sequentially
        x = x.mean(dim=1)  # Pooling across the sequence
        return self.fc(x)

# Full Model that combines the feature extractor and temporal model
class FullGestureRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FullGestureRecognitionModel, self).__init__()
        self.feature_extractor = FrameFeatureExtractor()
        self.gesture_recognizer = GestureRecognitionModel(feature_dim=512, num_classes=num_classes)

    def forward(self, x):
        frame_features = self.feature_extractor(x)  # Extract frame-level features
        output = self.gesture_recognizer(frame_features)  # Recognize gesture based on temporal patterns
        return output

# Load the model and set it to evaluation mode
num_classes = 44  # Updated based on the number of ASL classes
model = FullGestureRecognitionModel(num_classes=num_classes)
model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
model.eval()  # Evaluation mode disables dropout, etc.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define transformation for preprocessing frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize video capture
cap = cv2.VideoCapture(1)  
if not cap.isOpened():
    cap = cv2.VideoCapture(0)  # Fall back to primary camera if the second camera fails

# Frame buffer to store 30 frames
frame_buffer = []
pause = False  # Control to pause/unpause the camera feed

# Variables to store the last prediction and confidence
last_prediction = "Waiting for Prediction"
last_confidence = 0.0

print("Starting ASL detection. Press 'p' to pause/unpause, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    if not pause:
        # Convert the OpenCV BGR image to RGB, then to PIL Image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # Convert from NumPy array to PIL Image

        # Apply transformations and add to frame buffer
        img_tensor = transform(img)  # Shape: (channels, height, width)
        frame_buffer.append(img_tensor)

        # Check if we have a full sequence
        if len(frame_buffer) == sequence_length:
            # Convert list of frames to a single tensor of shape (batch, channels, time, height, width)
            input_tensor = torch.stack(frame_buffer, dim=1).unsqueeze(0).to(device)  # Shape: (1, 3, 30, 224, 224)

            # Run the model for prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = softmax(output, dim=1)  # Convert to probabilities
                confidence, predicted = torch.max(probabilities, 1)  # Get the top prediction

                # Retrieve label and confidence score
                label_index = predicted.item()
                confidence = confidence.item()  # Confidence as a decimal

                if confidence >= confidence_threshold:
                    last_prediction = f"Predicted: {class_labels[label_index]} ({confidence * 100:.2f}%)"
                else:
                    last_prediction = "Low Confidence"

                # Clear the buffer for the next sequence
                frame_buffer = []

    # Display the last prediction on the frame
    x, y = 10, 80
    cv2.putText(frame, last_prediction, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)  # Black shadow
    cv2.putText(frame, last_prediction, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4, cv2.LINE_AA)     # Red text

    # Show the frame with the current or last prediction
    cv2.imshow('ASL Detection with Debugging', frame)

    # Key control for testing
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        print("Quitting...")
        break
    elif key == ord('p'):  # Pause/unpause
        pause = not pause
        if pause:
            print("Paused video feed.")
        else:
            print("Resuming video feed.")

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

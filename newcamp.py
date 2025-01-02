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
import tkinter as tk
import matplotlib.pyplot as plt
from collections import deque

# Suppress warnings from torchvision and other libraries
warnings.filterwarnings("ignore", category=UserWarning)

# Path to the saved model file on your PC
model_file_path = r'C:\Users\malle\Desktop\asl_timesformer\gesture_recognition_model.pth'

# Define classes for each mode
alphabet_classes = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]
number_classes = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
phrase_classes = ["excuse me", "friend", "help", "no", "rest position", "sorry", "stop", "thank you"]

# Combine all classes for the model and create indices for each mode
all_classes = alphabet_classes + number_classes + phrase_classes
alphabet_indices = list(range(len(alphabet_classes)))
number_indices = list(range(len(alphabet_classes), len(alphabet_classes) + len(number_classes)))
phrase_indices = list(range(len(alphabet_classes) + len(number_classes), len(all_classes)))

# Confidence threshold
confidence_threshold = 0.2  # Only display predictions with confidence >= 40%
sequence_length = 30  # Number of frames in each sequence
skin_threshold = 0.5  # Minimum percentage of skin pixels to trigger prediction

# Mode variables
modes = ["Alphabet", "Number", "Phrase"]
mode_indices = [alphabet_indices, number_indices, phrase_indices]
current_mode = 0  # Default to Alphabet mode

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
num_classes = len(all_classes)
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

# Setup for confidence plot
confidence_deque = deque(maxlen=50)  # Rolling deque to store the last 50 confidence scores
plt.ion()  # Interactive mode on for real-time plotting

def update_confidence_plot():
    plt.clf()
    plt.plot(list(confidence_deque), label="Confidence")
    plt.axhline(y=confidence_threshold, color='r', linestyle='--', label="Threshold")
    plt.ylim(0, 1)
    plt.xlabel("Prediction Step")
    plt.ylabel("Confidence")
    plt.legend()
    plt.draw()
    plt.pause(0.001)

def detect_skin(frame):
    """Detect skin in the frame and return the percentage of skin pixels."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_pixels = np.sum(skin_mask > 0)
    total_pixels = skin_mask.size
    skin_ratio = skin_pixels / total_pixels
    return skin_ratio

def set_mode(selected_mode):
    """Set the current mode based on GUI selection."""
    global current_mode
    current_mode = modes.index(selected_mode)
    print(f"Mode changed to {selected_mode}.")

# Create the GUI for mode selection
root = tk.Tk()
root.title("ASL Mode Selection")

# Add buttons for each mode
tk.Label(root, text="Select Mode:", font=("Arial", 14)).pack(pady=10)
for mode in modes:
    button = tk.Button(root, text=mode, font=("Arial", 12), command=lambda m=mode: set_mode(m))
    button.pack(pady=5)

print("Starting ASL detection. Press 'p' to pause/unpause, 'q' to quit.")
# Initialize frame buffer at the top level
frame_buffer = []

def main_loop():
    global last_prediction, pause, frame_buffer  # Declare frame_buffer and pause as global

    pause = False  # Initialize pause here
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        if not pause:
            # Detect skin and calculate the skin ratio
            skin_ratio = detect_skin(frame)

            # Only proceed with prediction if skin_ratio exceeds the threshold
            if skin_ratio >= skin_threshold:
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
                        
                        # Filter probabilities based on the current mode
                        relevant_indices = mode_indices[current_mode]
                        filtered_probs = probabilities[0, relevant_indices]
                        
                        confidence, idx = torch.max(filtered_probs, 0)
                        label_index = relevant_indices[idx.item()]
                        confidence = confidence.item()  # Confidence as a decimal

                        if confidence >= confidence_threshold:
                            last_prediction = f"Mode: {modes[current_mode]} - Predicted: {all_classes[label_index]} ({confidence * 100:.2f}%)"
                        else:
                            last_prediction = "Low Confidence"

                        # Add the confidence score to the rolling deque
                        confidence_deque.append(confidence)
                        update_confidence_plot()  # Update the confidence plot

                        # Clear the buffer for the next sequence
                        frame_buffer = []
            else:
                last_prediction = "No skin detected"

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

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()  # Show plot window if needed after quitting
    root.quit()  # Close the GUI window if the main loop is terminated


main_loop()

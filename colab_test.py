import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import time

# Load class labels
dataset_path = "C:\\Users\\malle\\Desktop\\asl_timesformer\\dataset"
classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

# Frame Feature Extractor and Gesture Recognition Models
class FrameFeatureExtractor(nn.Module):
    def __init__(self, backbone="mobilenet_v2", train_backbone=False):
        super(FrameFeatureExtractor, self).__init__()
        if backbone == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT
            cnn = models.mobilenet_v2(weights=weights)
            self.feature_extractor = nn.Sequential(*list(cnn.features.children()))
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            if not train_backbone:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                self.feature_extractor.eval()

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(b, t, -1)
        return x

class GestureRecognitionModel(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(GestureRecognitionModel, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=3)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

class FullGestureRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FullGestureRecognitionModel, self).__init__()
        self.feature_extractor = FrameFeatureExtractor(train_backbone=False)
        self.gesture_recognizer = GestureRecognitionModel(feature_dim=1280, num_classes=num_classes)

    def forward(self, x):
        frame_features = self.feature_extractor(x)
        output = self.gesture_recognizer(frame_features)
        return output

# Load trained model
def load_model(model_path, num_classes, device):
    model = FullGestureRecognitionModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Capture and preprocess frame
def capture_frame(cap, transform):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        return None, None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    processed_frame = transform(pil_image)
    return frame, processed_frame

# Predict gesture from sequence of frames
def predict_gesture(model, frames, device, confidence_threshold=0.7):
    input_sequence = torch.stack(frames, dim=1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_sequence)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        confidence_pct = confidence.item() * 100
        predicted_label = classes[predicted_class.item()]

    return predicted_label, confidence_pct if confidence.item() >= confidence_threshold else None

# Display prediction on the frame
def display_prediction(frame, label, confidence_pct):
    if confidence_pct:
        cv2.putText(frame, f"Sign: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence_pct:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Uncertain", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("ASL Recognition", frame)

# Main function
def main():
    model_path = r'C:\Users\malle\Desktop\asl_timesformer\colab_model\gesture_recognition_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, len(classes), device)

    # Initialize camera and set properties
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Define transform, sequence parameters, and initialize frame collection
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    sequence_length = 30
    frames = []
    last_display_time = 0
    display_interval = 0.1

    while True:
        frame, processed_frame = capture_frame(cap, transform)
        if frame is None:
            break

        frames.append(processed_frame)
        if len(frames) < sequence_length:
            continue

        label, confidence_pct = predict_gesture(model, frames, device)

        if (time.time() - last_display_time) > display_interval:
            last_display_time = time.time()
            display_prediction(frame, label, confidence_pct)

        frames = []  # Clear frames after each prediction

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()

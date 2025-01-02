import cv2
import torch
import numpy as np
from torchvision.transforms import functional as F
from timesformer.models.vit import TimeSformer

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Parameters
num_frames = 8  # Number of frames per sequence
process_interval = 1  # Process every 3rd frame
confidence_threshold = 0.2  # Confidence threshold for displaying predictions

# Initialize and load model
model = TimeSformer(img_size=224, num_classes=27, num_frames=num_frames, attention_type='divided_space_time')
model.load_state_dict(torch.load(r"C:\Users\malle\Desktop\asl_timesformer\models\best_asl_alphabet_timesformer.pth", map_location=device))
model.to(device)
model.eval()

# Initialize video capture
cap = cv2.VideoCapture(1)
frame_sequence = []
predicted_label = "No gesture"  # Default label when confidence is low
confidence_display = "0.00%"    # Initialize confidence display
frame_counter = 0  # Frame counter

# Class names (A-Z for 27 classes)
class_names = [chr(i) for i in range(65, 65 + 27)]  # A-Z + 1 extra

# Real-time camera loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1  # Increment frame counter

    # Process only every `process_interval` frame
    if frame_counter % process_interval == 0:
        # Resize only the frame for model input, but keep display frame at original resolution
        model_input_frame = cv2.resize(frame, (224, 224))
        frame_rgb = cv2.cvtColor(model_input_frame, cv2.COLOR_BGR2RGB)
        frame_tensor = F.to_tensor(frame_rgb).float()
        frame_sequence.append(frame_tensor)

        # Check if enough frames have been collected
        if len(frame_sequence) == num_frames:
            # Stack frames and adjust tensor dimensions to [batch_size, channels, num_frames, height, width]
            video_tensor = torch.stack(frame_sequence).permute(1, 0, 2, 3).unsqueeze(0).to(device)

            # Perform model inference
            with torch.no_grad():
                output = model(video_tensor)
                probabilities = torch.softmax(output, dim=1)  # Get probabilities for each class
                confidence, predicted = torch.max(probabilities, 1)  # Get max confidence and its index

                # Check if confidence exceeds threshold and if the predicted class is valid
                if confidence.item() >= confidence_threshold and 0 <= predicted.item() < len(class_names):
                    predicted_label = class_names[predicted.item()]
                    confidence_display = f"{confidence.item() * 100:.2f}%"
                else:
                    predicted_label = "No gesture"  # Default label if prediction is invalid
                    confidence_display = "0.00%"

            # Clear the frame sequence for the next set of frames
            frame_sequence = []

    # Display the live video feed with prediction and confidence without resizing
    cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Confidence: {confidence_display}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Live ASL Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

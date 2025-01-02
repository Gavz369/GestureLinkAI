import os
import torch
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchvision.io import read_video
import torchvision.transforms as transforms
import sys
sys.path.append("C:/Users/malle/Desktop/asl_timesformer/TimeSformer")  # Updated path

from timesformer.models.vit import TimeSformer

# Set memory configuration to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Paths
dataset_path = r'C:\Users\malle\Desktop\asl_timesformer\dataset'  # Updated path
model_folder_path = r'C:\Users\malle\Desktop\asl_timesformer\all classess'  # Updated path
model_file_path = os.path.join(model_folder_path, 'asl_alphabet_timesformer.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class with dynamic class counting
class ASLAlphabetDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=8):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.num_classes = len(self.classes)  # Automatically count classes
        self.data = []
        
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for video_file in os.listdir(class_path):
                video_path = os.path.join(class_path, video_file)
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    self.data.append((video_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        video, _, _ = read_video(video_path, pts_unit='sec')
        
        if video.shape[0] > self.num_frames:
            start_frame = torch.randint(0, video.shape[0] - self.num_frames, (1,)).item()
            video = video[start_frame:start_frame + self.num_frames]
        elif video.shape[0] < self.num_frames:
            padding = torch.zeros((self.num_frames - video.shape[0], video.shape[1], video.shape[2], video.shape[3]))
            video = torch.cat((video, padding), dim=0)

        video = video.float() / 255.0
        video = video.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

        if self.transform:
            video = torch.stack([self.transform(frame) for frame in video])
        
        video = video.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
        
        return video, label

# Instantiate the dataset once to determine the number of classes
transform = transforms.Compose([transforms.Resize((224, 224))])
full_dataset = ASLAlphabetDataset(root_dir=dataset_path, transform=transform, num_frames=8)
num_classes = full_dataset.num_classes  # Use this throughout the script

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Optuna objective function with memory management
def objective(trial):
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    batch_size = trial.suggest_categorical('batch_size', [4, 8])  # Reduced batch size
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    accumulation_steps = 2  # Number of batches to accumulate gradients

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model = TimeSformer(img_size=224, num_classes=num_classes, num_frames=30, attention_type='divided_space_time').to(device)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=10, delta=0.001)
    
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (videos, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            videos, labels = videos.to(device), labels.to(device)
            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, labels) / accumulation_steps  # Scale loss for accumulation
                
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()  # Clear cache after optimizer step
            
            train_loss += loss.item() * accumulation_steps  # Undo scaling for actual loss

            # Clear large variables and cache
            del outputs, loss
            torch.cuda.empty_cache()

        model.eval()
        val_loss = 0.0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                with autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Clear large variables and cache
                del outputs, loss
                torch.cuda.empty_cache()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        early_stopping(val_loss)
        
        torch.cuda.empty_cache()  # Clear cache after validation

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

    return accuracy

# Run Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Display best trial
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print(f"  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Load or save the best model
best_model = TimeSformer(img_size=224, num_classes=num_classes, num_frames=8, attention_type='divided_space_time').to(device)
if os.path.exists(model_file_path):
    best_model.load_state_dict(torch.load(model_file_path))
    print("Loaded pretrained model.")
else:
    optimizer = getattr(torch.optim, study.best_trial.params['optimizer'])(best_model.parameters(), lr=study.best_trial.params['learning_rate'])
    os.makedirs(model_folder_path, exist_ok=True)
    torch.save(best_model.state_dict(), model_file_path)
    print("Saved new model.")

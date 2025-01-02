import os
import numpy as np
import torch
from torchvision.io import read_video
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from timesformer.models.vit import TimeSformer
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedShuffleSplit

# Define dataset and model paths
dataset_path = r'C:\Users\malle\Desktop\asl_timesformer\dataset'
model_folder_path = r'C:\Users\malle\Desktop\asl_timesformer\modeltest'
checkpoint_path = os.path.join(model_folder_path, 'best_asl_alphabet_timesformer_lstm.pth')

# Ensure the model folder exists
os.makedirs(model_folder_path, exist_ok=True)

# Custom dataset for loading ASL alphabet videos
class ASLAlphabetDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=8):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        self.data = []
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            video_files = [f for f in os.listdir(class_path) if f.endswith(('.mp4', '.avi', '.mov'))]
            for video_file in video_files:
                video_path = os.path.join(class_path, video_file)
                self.data.append((video_path, label))
        
        print(f"Detected {len(self.classes)} classes:")
        for class_name, idx in self.class_to_idx.items():
            print(f"{idx}: {class_name}")
        print(f"Total number of samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        video, _, _ = read_video(video_path, pts_unit='sec')
        
        # Truncate or pad video to ensure consistent frame count
        if video.shape[0] > self.num_frames:
            start_frame = torch.randint(0, video.shape[0] - self.num_frames, (1,)).item()
            video = video[start_frame:start_frame + self.num_frames]
        elif video.shape[0] < self.num_frames:
            padding = torch.zeros((self.num_frames - video.shape[0], video.shape[1], video.shape[2], video.shape[3]))
            video = torch.cat((video, padding), dim=0)
        
        video = video.float() / 255.0
        video = video.permute(0, 3, 1, 2)

        if self.transform:
            video = torch.stack([self.transform(frame) for frame in video])

        return video, label

# Define transformation for resizing frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

# Custom model with TimeSformer and LSTM
class TimeSformerLSTMModel(nn.Module):
    def __init__(self, img_size, num_classes, num_frames, lstm_hidden_size=512):
        super(TimeSformerLSTMModel, self).__init__()
        self.timesformer = TimeSformer(
            img_size=img_size, 
            num_classes=0,  # Set to 0 to get feature embeddings
            num_frames=num_frames, 
            attention_type='divided_space_time'
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

def train(model, train_loader, optimizer, criterion, scaler, device):
    model.train()
    correct, total, epoch_loss = 0, 0, 0
    with tqdm(total=len(train_loader), desc="Training", unit='batch') as pbar:
        for batch_idx, (videos, labels) in enumerate(train_loader):
            try:
                videos = videos.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Calculate current accuracy
                current_acc = 100 * correct / total
                # Update tqdm progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.2f}%'})
                pbar.update(1)
            except torch.cuda.OutOfMemoryError:
                print("CUDA OutOfMemoryError encountered during training. Clearing cache and skipping this batch.")
                torch.cuda.empty_cache()
                continue
    train_accuracy = 100 * correct / total
    avg_epoch_loss = epoch_loss / len(train_loader)
    return avg_epoch_loss, train_accuracy

def test(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []
    y_score = []
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Testing", unit='batch') as pbar:
            for videos, labels in test_loader:
                try:
                    videos = videos.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast():
                        outputs = model(videos)
                        probabilities = torch.softmax(outputs, dim=1)
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(predicted.cpu().tolist())
                    y_score.extend(probabilities.cpu().numpy())
                    
                except torch.cuda.OutOfMemoryError:
                    print("CUDA OutOfMemoryError encountered during testing. Clearing cache and skipping this batch.")
                    torch.cuda.empty_cache()
                    continue
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Calculate current accuracy
                current_acc = 100 * correct / total
                # Update tqdm progress bar
                pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
                pbar.update(1)
    test_accuracy = 100 * correct / total
    return test_accuracy, y_true, y_pred, y_score

def plot_and_save(train_accuracies, test_accuracy, y_true, y_pred, y_score, dataset, model_folder_path, num_classes):
    # Plot Training and Testing Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Training Accuracy')
    plt.plot([len(train_accuracies)], [test_accuracy], 'ro', label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    accuracy_plot_path = os.path.join(model_folder_path, 'accuracy_plot.png')
    plt.savefig(accuracy_plot_path)
    print(f"Training and Testing Accuracy plot saved to {accuracy_plot_path}")
    plt.close()
    
    # Confusion Matrix with all classes
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig_cm, ax_cm = plt.subplots(figsize=(12, 12))  # Increased size for better readability
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
    plt.xticks(rotation=90)
    cm_plot_path = os.path.join(model_folder_path, 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    print(f"Confusion Matrix plot saved to {cm_plot_path}")
    plt.close(fig_cm)
    
    # ROC Curve
    # Binarize the output labels for ROC computation
    y_true_binarized = label_binarize(y_true, classes=list(range(num_classes)))
    y_score = np.array(y_score)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {dataset.idx_to_class[i]} (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right', fontsize='small', ncol=2)  # Adjust legend for readability
    roc_plot_path = os.path.join(model_folder_path, 'roc_curve.png')
    plt.savefig(roc_plot_path)
    print(f"ROC Curve plot saved to {roc_plot_path}")
    plt.close()


def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = ASLAlphabetDataset(root_dir=dataset_path, transform=transform, num_frames=8)
    num_classes = len(dataset.classes)
    # Instantiate dataset, split into train and test sets
    labels = [label for _, label in dataset.data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(sss.split(np.zeros(len(labels)), labels))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # Create DataLoader for train and test sets
    # Reduce batch size to alleviate memory issues
    batch_size = 4  # Start with 2, adjust as needed (reduce to 1 if OOM persists)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    # Instantiate model, optimizer, and loss function
    model = TimeSformerLSTMModel(img_size=224, num_classes=num_classes, num_frames=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Early stopping and checkpointing variables
    patience = 3
    best_loss = np.inf
    epochs_no_improve = 0
    
    # Lists to store accuracy for plotting
    train_accuracies = []
    test_accuracies = []
    
    # Training loop with progress update using tqdm and mixed precision
    print("Starting training...")
    num_epochs = 50
    for epoch in range(num_epochs):
        avg_epoch_loss, train_accuracy = train(model, train_loader, optimizer, criterion, scaler, device)
        train_accuracies.append(train_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        
        # Early stopping and checkpointing
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch+1} with loss {best_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
        
        if epochs_no_improve == patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break
    
    # Testing and accuracy calculation with tqdm and mixed precision
    test_accuracy, y_true, y_pred, y_score = test(model, test_loader, device)
    test_accuracies.append(test_accuracy)
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # Plot and save all visualizations
    plot_and_save(train_accuracies, test_accuracy, y_true, y_pred, y_score, dataset, model_folder_path, num_classes)

if __name__ == '__main__':
    main()

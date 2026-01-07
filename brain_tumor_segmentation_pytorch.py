"""
Brain Tumor Segmentation using U-Net - PyTorch Implementation
Dataset: BraTS 2021
Author: GitHub Copilot Assistant
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================================================
# Configuration
# ============================================================
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues on Windows

# Paths
DATA_PATH = os.path.join(os.getcwd(), 'BraTS2021_Training_Data')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'models_pytorch')
RESULTS_PATH = os.path.join(os.getcwd(), 'results_pytorch')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Data Loading Functions
# ============================================================

def load_nifti_file(filepath):
    """Load a NIfTI file and return the data array"""
    try:
        nifti = nib.load(filepath)
        return nifti.get_fdata()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def normalize_volume(volume):
    """Normalize volume to [0, 1] range"""
    min_val = np.min(volume)
    max_val = np.max(volume)
    if max_val - min_val > 0:
        volume = (volume - min_val) / (max_val - min_val)
    return volume

def process_patient_data(patient_path, modality='flair'):
    """Process a single patient's data"""
    patient_id = os.path.basename(patient_path)
    
    image_filename = f"{patient_id}_{modality}.nii.gz"
    mask_filename = f"{patient_id}_seg.nii.gz"
    
    image_path = os.path.join(patient_path, image_filename)
    mask_path = os.path.join(patient_path, mask_filename)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    image_volume = load_nifti_file(image_path)
    mask_volume = load_nifti_file(mask_path)
    
    if image_volume is None or mask_volume is None:
        raise ValueError(f"Failed to load data from {patient_path}")
    
    image_volume = normalize_volume(image_volume)
    mask_volume = (mask_volume > 0).astype(np.float32)
    
    return image_volume, mask_volume

def extract_2d_slices(image_volume, mask_volume, axis=2, min_tumor_ratio=0.01):
    """Extract 2D slices from 3D volume"""
    images = []
    masks = []
    
    num_slices = image_volume.shape[axis]
    
    for i in range(num_slices):
        if axis == 2:
            img_slice = image_volume[:, :, i]
            mask_slice = mask_volume[:, :, i]
        
        tumor_ratio = np.sum(mask_slice) / mask_slice.size
        if tumor_ratio >= min_tumor_ratio:
            images.append(img_slice)
            masks.append(mask_slice)
    
    return images, masks

def resize_slice(img, size=(IMG_SIZE, IMG_SIZE)):
    """Resize a 2D slice"""
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

def prepare_dataset(data_path, num_patients=None, modality='flair'):
    """Prepare the complete dataset"""
    patient_dirs = sorted([d for d in glob(os.path.join(data_path, '*')) if os.path.isdir(d)])
    
    if num_patients:
        patient_dirs = patient_dirs[:num_patients]
    
    all_images = []
    all_masks = []
    
    print(f"Processing {len(patient_dirs)} patients...")
    
    for patient_path in tqdm(patient_dirs, desc="Processing patients"):
        try:
            image_vol, mask_vol = process_patient_data(patient_path, modality)
            images, masks = extract_2d_slices(image_vol, mask_vol)
            
            for img, msk in zip(images, masks):
                img_resized = resize_slice(img)
                msk_resized = resize_slice(msk)
                all_images.append(img_resized)
                all_masks.append(msk_resized)
        
        except Exception as e:
            continue
    
    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_masks, dtype=np.float32)
    
    print(f"\nDataset prepared: {len(X)} slices")
    return X, y

# ============================================================
# Dataset Class
# ============================================================

class BraTSDataset(Dataset):
    def __init__(self, images, masks):
        self.images = torch.FloatTensor(images).unsqueeze(1)  # Add channel dimension
        self.masks = torch.FloatTensor(masks).unsqueeze(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

# ============================================================
# U-Net Model
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return skip, x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        
        # Output
        x = self.out(x)
        x = self.sigmoid(x)
        return x

# ============================================================
# Loss Functions and Metrics
# ============================================================

def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    """Calculate IoU score"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, pred, target):
        return 1 - dice_coefficient(pred, target)

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)

# ============================================================
# Training Functions
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_dice += dice_coefficient(outputs, masks).item()
        running_iou += iou_score(outputs, masks).item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice_coefficient(outputs, masks).item():.4f}'
        })
    
    avg_loss = running_loss / len(dataloader)
    avg_dice = running_dice / len(dataloader)
    avg_iou = running_iou / len(dataloader)
    
    return avg_loss, avg_dice, avg_iou

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            running_dice += dice_coefficient(outputs, masks).item()
            running_iou += iou_score(outputs, masks).item()
    
    avg_loss = running_loss / len(dataloader)
    avg_dice = running_dice / len(dataloader)
    avg_iou = running_iou / len(dataloader)
    
    return avg_loss, avg_dice, avg_iou

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    """Complete training loop"""
    history = {
        'train_loss': [], 'train_dice': [], 'train_iou': [],
        'val_loss': [], 'val_dice': [], 'val_iou': []
    }
    
    best_dice = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_dice, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        
        print(f"Train Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'unet_best.pth'))
            print(f"✓ Saved best model (Dice: {best_dice:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    return history

# ============================================================
# Visualization and Results Functions
# ============================================================

def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice Coefficient
    axes[0, 1].plot(epochs, history['train_dice'], 'b-', label='Train Dice')
    axes[0, 1].plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    axes[0, 1].set_title('Dice Coefficient')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU
    axes[1, 0].plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    axes[1, 0].plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    axes[1, 0].set_title('IoU Metric')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Combined metrics
    axes[1, 1].plot(epochs, history['train_dice'], 'b-', label='Train Dice', alpha=0.7)
    axes[1, 1].plot(epochs, history['val_dice'], 'r-', label='Val Dice', alpha=0.7)
    axes[1, 1].plot(epochs, history['train_iou'], 'b--', label='Train IoU', alpha=0.7)
    axes[1, 1].plot(epochs, history['val_iou'], 'r--', label='Val IoU', alpha=0.7)
    axes[1, 1].set_title('All Metrics')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved to {save_path}")

def evaluate_model(model, dataloader, device):
    """Evaluate model on test set"""
    model.eval()
    all_dice = []
    all_iou = []
    
    print("\nEvaluating model on test set...")
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # Calculate metrics for each sample
            for i in range(outputs.size(0)):
                dice = dice_coefficient(outputs[i:i+1], masks[i:i+1]).item()
                iou = iou_score(outputs[i:i+1], masks[i:i+1]).item()
                all_dice.append(dice)
                all_iou.append(iou)
    
    all_dice = np.array(all_dice)
    all_iou = np.array(all_iou)
    
    print("\n" + "="*60)
    print("Test Set Results:")
    print("="*60)
    print(f"Dice - Mean: {all_dice.mean():.4f}, Std: {all_dice.std():.4f}")
    print(f"     - Min: {all_dice.min():.4f}, Max: {all_dice.max():.4f}")
    print(f"IoU  - Mean: {all_iou.mean():.4f}, Std: {all_iou.std():.4f}")
    print(f"     - Min: {all_iou.min():.4f}, Max: {all_iou.max():.4f}")
    print("="*60)
    
    return all_dice, all_iou

def plot_metrics_distribution(dice_scores, iou_scores, save_path):
    """Plot distribution of metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Dice distribution
    axes[0].hist(dice_scores, bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[0].axvline(dice_scores.mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {dice_scores.mean():.3f}')
    axes[0].set_title('Dice Score Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Dice Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # IoU distribution
    axes[1].hist(iou_scores, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(iou_scores.mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {iou_scores.mean():.3f}')
    axes[1].set_title('IoU Score Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('IoU Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics distribution saved to {save_path}")

def visualize_predictions(model, dataloader, device, save_path, num_samples=10):
    """Visualize predictions on random samples"""
    model.eval()
    
    # Get random samples
    dataset = dataloader.dataset
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices):
            image, mask = dataset[sample_idx]
            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            
            output = model(image)
            prediction = (output > 0.5).float()
            
            # Move to CPU for visualization
            image_np = image.cpu().squeeze().numpy()
            mask_np = mask.cpu().squeeze().numpy()
            pred_np = prediction.cpu().squeeze().numpy()
            
            # Calculate metrics
            dice = dice_coefficient(output, mask).item()
            iou = iou_score(output, mask).item()
            
            # Original image
            axes[idx, 0].imshow(image_np, cmap='gray')
            axes[idx, 0].set_title('MRI Slice', fontsize=12)
            axes[idx, 0].axis('off')
            
            # Ground truth
            axes[idx, 1].imshow(mask_np, cmap='jet')
            axes[idx, 1].set_title('Ground Truth', fontsize=12)
            axes[idx, 1].axis('off')
            
            # Prediction
            axes[idx, 2].imshow(pred_np, cmap='jet')
            axes[idx, 2].set_title('Prediction', fontsize=12)
            axes[idx, 2].axis('off')
            
            # Overlay
            axes[idx, 3].imshow(image_np, cmap='gray')
            axes[idx, 3].imshow(pred_np, cmap='jet', alpha=0.4)
            axes[idx, 3].set_title(f'Overlay\nDice: {dice:.3f} | IoU: {iou:.3f}', fontsize=11)
            axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Predictions saved to {save_path}")

def save_results_summary(history, dice_scores, iou_scores, save_path):
    """Save text summary of results"""
    summary_path = os.path.join(save_path, 'results_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Brain Tumor Segmentation - PyTorch Results Summary\n")
        f.write("="*60 + "\n\n")
        
        f.write("Configuration:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Model: U-Net\n")
        f.write(f"Framework: PyTorch {torch.__version__}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Image Size: {IMG_SIZE}x{IMG_SIZE}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Epochs Trained: {len(history['train_loss'])}\n\n")
        
        f.write("Training History:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Best Train Dice: {max(history['train_dice']):.4f}\n")
        f.write(f"Best Val Dice: {max(history['val_dice']):.4f}\n")
        f.write(f"Best Train IoU: {max(history['train_iou']):.4f}\n")
        f.write(f"Best Val IoU: {max(history['val_iou']):.4f}\n")
        f.write(f"Final Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Val Loss: {history['val_loss'][-1]:.4f}\n\n")
        
        f.write("Test Set Results:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Dice Coefficient:\n")
        f.write(f"  Mean: {dice_scores.mean():.4f}\n")
        f.write(f"  Std:  {dice_scores.std():.4f}\n")
        f.write(f"  Min:  {dice_scores.min():.4f}\n")
        f.write(f"  Max:  {dice_scores.max():.4f}\n\n")
        
        f.write(f"IoU Score:\n")
        f.write(f"  Mean: {iou_scores.mean():.4f}\n")
        f.write(f"  Std:  {iou_scores.std():.4f}\n")
        f.write(f"  Min:  {iou_scores.min():.4f}\n")
        f.write(f"  Max:  {iou_scores.max():.4f}\n\n")
        
        f.write("="*60 + "\n")
    
    print(f"✓ Results summary saved to {summary_path}")

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    # Create directories
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device:", torch.cuda.get_device_name(0))
    print("\nDevice:", DEVICE)
    print("Model save path:", MODEL_SAVE_PATH)
    print("Results path:", RESULTS_PATH)
    
    print("="*60)
    print("Brain Tumor Segmentation - PyTorch Implementation")
    print("="*60)
    
    # 1. Prepare dataset
    X, y = prepare_dataset(DATA_PATH, num_patients=10, modality='flair')
    
    # 2. Split dataset
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 3. Create datasets and dataloaders
    train_dataset = BraTSDataset(X_train, y_train)
    val_dataset = BraTSDataset(X_val, y_val)
    test_dataset = BraTSDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # 4. Initialize model
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 5. Train model
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, DEVICE)
    
    # 6. Save final model
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'unet_final.pth'))
    print("\n✓ Training complete!")
    
    # 7. Plot training history
    print("\n" + "="*60)
    print("Generating Results...")
    print("="*60)
    plot_training_history(history, RESULTS_PATH)
    
    # 8. Evaluate on test set
    dice_scores, iou_scores = evaluate_model(model, test_loader, DEVICE)
    
    # 9. Plot metrics distribution
    plot_metrics_distribution(dice_scores, iou_scores, RESULTS_PATH)
    
    # 10. Visualize predictions
    visualize_predictions(model, test_loader, DEVICE, RESULTS_PATH, num_samples=10)
    
    # 11. Save results summary
    save_results_summary(history, dice_scores, iou_scores, RESULTS_PATH)
    
    print("\n" + "="*60)
    print("All results saved to:", RESULTS_PATH)
    print("="*60)
    print("\nGenerated files:")
    print("  - training_history.png")
    print("  - metrics_distribution.png")
    print("  - predictions.png")
    print("  - results_summary.txt")
    print("\nModel saved to:", MODEL_SAVE_PATH)
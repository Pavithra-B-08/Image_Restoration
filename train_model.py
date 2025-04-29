import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import os
import traceback
import sys
from datetime import datetime

# First, let's check if the unet_model.py exists
if not os.path.exists('unet_model.py'):
    print("ERROR: unet_model.py does not exist in the current directory.")
    sys.exit(1)

try:
    from unet_model import UNet
    print("Successfully imported UNet model")
except Exception as e:
    print(f"ERROR importing UNet model: {e}")
    traceback.print_exc()
    sys.exit(1)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset class with error handling
class ImageDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        # Check if directories exist
        if not os.path.exists(low_res_dir):
            raise FileNotFoundError(f"Low resolution directory not found: {low_res_dir}")
        if not os.path.exists(high_res_dir):
            raise FileNotFoundError(f"High resolution directory not found: {high_res_dir}")
            
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        
        # Get image lists and check if they're not empty
        self.low_res_images = sorted([f for f in os.listdir(low_res_dir) 
                                    if os.path.isfile(os.path.join(low_res_dir, f)) and 
                                    f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
        
        self.high_res_images = sorted([f for f in os.listdir(high_res_dir) 
                                     if os.path.isfile(os.path.join(high_res_dir, f)) and 
                                     f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
        
        if len(self.low_res_images) == 0:
            raise ValueError(f"No valid images found in {low_res_dir}")
        if len(self.high_res_images) == 0:
            raise ValueError(f"No valid images found in {high_res_dir}")
            
        # Check if the number of images match
        if len(self.low_res_images) != len(self.high_res_images):
            raise ValueError(f"Number of low resolution images ({len(self.low_res_images)}) " 
                           f"does not match high resolution images ({len(self.high_res_images)})")
            
        self.transform = transform
        print(f"Dataset initialized with {len(self.low_res_images)} image pairs")

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        if idx >= len(self.low_res_images):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.low_res_images)} elements")
            
        low_res_path = os.path.join(self.low_res_dir, self.low_res_images[idx])
        high_res_path = os.path.join(self.high_res_dir, self.high_res_images[idx])

        try:
            low_res = Image.open(low_res_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error loading low resolution image {low_res_path}: {e}")
            
        try:
            high_res = Image.open(high_res_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error loading high resolution image {high_res_path}: {e}")

        if self.transform:
            try:
                low_res = self.transform(low_res)
                high_res = self.transform(high_res)
            except Exception as e:
                raise RuntimeError(f"Error applying transforms to images: {e}")

        return low_res, high_res

def main():
    try:
        # Create timestamp for logs and saved models
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up logging
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
        
        def log_message(message):
            print(message)
            with open(log_file, 'a') as f:
                f.write(message + '\n')
        
        log_message(f"Training started at {timestamp}")
        
        # Hyperparameters and settings
        batch_size = 8
        num_epochs = 25
        learning_rate = 1e-4
        save_freq = 5  # Save model every 5 epochs
        
        # Transform
        print("Setting up transformations...")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        log_message("Transformations set up successfully")
        
        # Paths
        train_low_res_dir = 'dataset/train/low_res'
        train_high_res_dir = 'dataset/train/high_res'
        val_low_res_dir = 'dataset/val/low_res'
        val_high_res_dir = 'dataset/val/high_res'
        
        # Test dataset directories
        for dir_path in [train_low_res_dir, train_high_res_dir, val_low_res_dir, val_high_res_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory does not exist: {dir_path}")
            else:
                log_message(f"Directory exists: {dir_path}")
        
        # Datasets
        log_message("Creating training dataset...")
        train_dataset = ImageDataset(train_low_res_dir, train_high_res_dir, transform=transform)
        log_message(f"Training dataset created with {len(train_dataset)} samples")
        
        log_message("Creating validation dataset...")
        val_dataset = ImageDataset(val_low_res_dir, val_high_res_dir, transform=transform)
        log_message(f"Validation dataset created with {len(val_dataset)} samples")
        
        # Test a sample from dataset
        log_message("Testing a sample from the dataset...")
        sample_input, sample_target = train_dataset[0]
        log_message(f"Sample input shape: {sample_input.shape}")
        log_message(f"Sample target shape: {sample_target.shape}")
        
        # DataLoaders
        log_message("Creating data loaders...")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        log_message("Data loaders created successfully")
        
        # Initialize model
        log_message("Initializing UNet model...")
        try:
            # Try to initialize the model and print its structure
            model = UNet(in_channels=3, out_channels=3)  # Adjust parameters based on your UNet implementation
            model = model.to(device)
            log_message(f"Model initialized successfully and moved to {device}")
            log_message(f"Model structure: {model}")
        except TypeError as e:
            # If there's a type error, it might be due to incorrect parameters
            log_message(f"TypeError when initializing model: {e}")
            log_message("Trying to initialize model without parameters...")
            try:
                model = UNet()
                model = model.to(device)
                log_message("Model initialized without parameters successfully")
            except Exception as inner_e:
                raise RuntimeError(f"Failed to initialize model: {inner_e}")
        
        # Loss and optimizer
        log_message("Setting up loss function and optimizer...")
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        log_message("Loss function and optimizer set up successfully")
        
        # Create directory for saving models
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        log_message(f"Model save directory created: {models_dir}")
        
        # Create directory for sample outputs
        samples_dir = 'samples'
        os.makedirs(samples_dir, exist_ok=True)
        log_message(f"Sample output directory created: {samples_dir}")
        
        # Training Loop
        log_message(f"Starting training for {num_epochs} epochs...")
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            
            log_message(f"Epoch {epoch+1}/{num_epochs} - Training...")
            for i, (inputs, targets) in enumerate(train_loader):
                try:
                    # Move data to device
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    # Print progress every 10 batches
                    if (i + 1) % 10 == 0:
                        log_message(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], "
                                   f"Loss: {loss.item():.4f}")
                
                except Exception as e:
                    log_message(f"Error during training batch {i}: {e}")
                    traceback.print_exc()
                    continue
            
            epoch_loss = running_loss / len(train_loader)
            log_message(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            log_message(f"Epoch {epoch+1}/{num_epochs} - Validation...")
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(val_loader):
                    try:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                    
                    except Exception as e:
                        log_message(f"Error during validation batch {i}: {e}")
                        traceback.print_exc()
                        continue
            
            val_loss /= len(val_loader)
            log_message(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
            
            # Save sample outputs
            try:
                if i > 0:  # Make sure we have at least one batch
                    sample_idx = 0  # Use first image in batch
                    sample_input = inputs[sample_idx].cpu()
                    sample_target = targets[sample_idx].cpu()
                    sample_output = outputs[sample_idx].cpu()
                    
                    # Create grid of images
                    samples = torch.stack([sample_input, sample_output, sample_target])
                    save_image(samples, os.path.join(samples_dir, f'epoch_{epoch+1}.png'), 
                              nrow=3, normalize=True)
                    log_message(f"Saved sample outputs for epoch {epoch+1}")
            except Exception as e:
                log_message(f"Error saving sample outputs: {e}")
            
            # Save model checkpoint
            if (epoch + 1) % save_freq == 0 or epoch == num_epochs - 1:
                model_path = os.path.join(models_dir, f"model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), model_path)
                log_message(f"Saved model checkpoint at {model_path}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(models_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                log_message(f"New best model saved with validation loss: {val_loss:.4f}")
        
        log_message("Training completed successfully!")
        return True
    
    except Exception as e:
        print(f"ERROR in training process: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("Training script executed successfully.")
    else:
        print("Training script encountered errors. Check the logs for details.")
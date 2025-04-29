import torch
import argparse
import os
import sys
import traceback
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Check if the unet_model.py exists
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

def load_model(model_path, device):
    """Load the trained restoration model"""
    print(f"Loading model from {model_path}...")
    
    try:
        # Try with parameters first
        model = UNet(in_channels=3, out_channels=3).to(device)
        print("Model structure initialized with parameters")
    except TypeError as e:
        print(f"TypeError when initializing model with parameters: {e}")
        # Try without parameters as fallback
        try:
            model = UNet().to(device)
            print("Model structure initialized without parameters")
        except Exception as inner_e:
            print(f"Failed to initialize model structure: {inner_e}")
            traceback.print_exc()
            sys.exit(1)
    
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            sys.exit(1)
        
        # Load the model weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if it's a state_dict or full checkpoint
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded model state_dict from checkpoint")
            if 'epoch' in checkpoint:
                print(f"Model was trained for {checkpoint['epoch']} epochs")
            if 'val_loss' in checkpoint:
                print(f"Validation loss: {checkpoint['val_loss']:.6f}")
        else:
            # Assume it's just the state_dict
            model.load_state_dict(checkpoint)
            print(f"Loaded model state_dict directly")
        
        # Set model to evaluation mode
        model.eval()
        print(f"Model loaded successfully and set to evaluation mode")
        return model
    
    except Exception as e:
        print(f"ERROR loading model weights: {e}")
        traceback.print_exc()
        sys.exit(1)

def process_and_display_image(model, input_path, device, save_output=True):
    """Restore a degraded image using the trained model, display the results, and save to Testing_outputs folder"""
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"ERROR: Input image not found: {input_path}")
            return False
        
        print(f"Processing image: {input_path}")
        
        # Load the image
        try:
            input_image = Image.open(input_path).convert("RGB")
            print(f"Image loaded successfully: {input_path}")
            print(f"Image size: {input_image.size}")
        except Exception as e:
            print(f"ERROR loading input image: {e}")
            return False
        
        # Store original size for later resizing
        original_size = input_image.size
        
        # Transform the image
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Apply transformation
        input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(device)
        print(f"Image transformed to tensor of shape: {input_tensor.shape}")
        
        # Process with model
        print("Running image through model...")
        with torch.no_grad():
            try:
                output_tensor = model(input_tensor)
                print("Model processing completed successfully")
            except Exception as e:
                print(f"ERROR during model inference: {e}")
                traceback.print_exc()
                return False
        
        # Convert output tensor to image
        print("Converting output tensor to image...")
        try:
            # Move to CPU and remove batch dimension
            output_tensor = output_tensor.cpu().squeeze(0)
            
            # Convert tensor to numpy array and transpose from CxHxW to HxWxC
            output_array = output_tensor.numpy().transpose(1, 2, 0)
            
            # Clip values to [0, 1] range
            output_array = np.clip(output_array, 0, 1)
            
            # Convert to uint8 range [0, 255]
            output_array = (output_array * 255).astype(np.uint8)
            
            # Create PIL image
            output_image = Image.fromarray(output_array)
            
            # Resize back to original dimensions
            output_image = output_image.resize(original_size, Image.BICUBIC)
            
            # Save the output image if requested
            if save_output:
                # Create Testing_outputs directory if it doesn't exist
                output_dir = "Testing_outputs"
                os.makedirs(output_dir, exist_ok=True)
                
                # Extract filename from input path
                input_filename = os.path.basename(input_path)
                base_name, ext = os.path.splitext(input_filename)
                
                # Create output filename
                output_filename = f"{base_name}_restored{ext}"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save the image
                output_image.save(output_path)
                print(f"Restored image saved to: {output_path}")
            
            # Display the images side by side
            plt.figure(figsize=(14, 7))
            
            # Set figure title
            plt.suptitle("Image Restoration Results", fontsize=16)
            
            # Display input image
            plt.subplot(1, 2, 1)
            plt.imshow(np.array(input_image))
            plt.title("Input (Degraded) Image", fontsize=14)
            plt.axis('off')
            
            # Display restored image
            plt.subplot(1, 2, 2)
            plt.imshow(np.array(output_image))
            plt.title("Restored Image", fontsize=14)
            plt.axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for main title
            
            # Show plot
            plt.show()
            
            print("Images displayed successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR creating or displaying output image: {e}")
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"ERROR in process_and_display_image function: {e}")
        traceback.print_exc()
        return False

def process_batch_images(model, input_dir, device):
    """Process all images in a directory and save outputs to Testing_outputs folder"""
    print(f"\nBatch processing images from: {input_dir}")
    
    # Create Testing_outputs directory if it doesn't exist
    output_dir = "Testing_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created/verified: {output_dir}")
    
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(input_dir, file))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return False
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    success_count = 0
    for image_path in image_files:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        if process_and_display_image(model, image_path, device, save_output=True):
            success_count += 1
    
    print(f"\nBatch processing complete. Successfully processed {success_count} out of {len(image_files)} images.")
    print(f"Restored images saved to '{output_dir}' directory")
    
    return success_count > 0

def get_user_input_path():
    """Prompt user for input image path and validate it exists"""
    while True:
        input_path = input("\nEnter the path to the degraded image you want to restore: ").strip()
        
        # Handle empty input
        if not input_path:
            print("Please enter a valid path.")
            continue
            
        # Check if file exists
        if not os.path.exists(input_path):
            print(f"Error: File '{input_path}' not found. Please enter a valid path.")
            continue
            
        # Check if it's an image file
        try:
            with Image.open(input_path) as img:
                img.verify()  # Verify it's an image file
            return input_path
        except Exception as e:
            print(f"Error: '{input_path}' is not a valid image file. Please try again.")
            continue

def get_batch_input_directory():
    """Prompt user for a directory containing images to process"""
    while True:
        input_dir = input("\nEnter the path to the directory containing images to process: ").strip()
        
        # Handle empty input
        if not input_dir:
            print("Please enter a valid directory path.")
            continue
            
        # Check if directory exists
        if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
            print(f"Error: Directory '{input_dir}' not found. Please enter a valid path.")
            continue
            
        return input_dir

def get_model_path(default_path="models/best_model.pth"):
    """Prompt user for model path or use default"""
    while True:
        model_path = input(f"\nEnter the path to the trained model [default: {default_path}]: ").strip()
        
        # Use default if empty
        if not model_path:
            model_path = default_path
            
        # Check if file exists
        if not os.path.exists(model_path):
            print(f"Warning: Model file '{model_path}' not found.")
            confirm = input("Continue anyway? (y/n): ").lower()
            if confirm == 'y' or confirm == 'yes':
                return model_path
            else:
                continue
                
        return model_path

def main():
    # Display welcome message
    print("\n" + "="*60)
    print("IMAGE RESTORATION VISUALIZATION TOOL".center(60))
    print("="*60)
    print("This tool restores and displays degraded images using a trained UNet model.\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Optional: Parse arguments for non-interactive mode
    parser = argparse.ArgumentParser(description='Restore and display degraded images using trained model')
    parser.add_argument('--input', '-i', type=str, help='Path to input degraded image or directory (optional)')
    parser.add_argument('--model', '-m', type=str, help='Path to trained model (optional)')
    parser.add_argument('--batch', '-b', action='store_true', help='Process all images in the input directory')
    parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode')
    
    # Parse arguments
    try:
        args = parser.parse_args()
    except Exception as e:
        print(f"ERROR parsing arguments: {e}")
        return False
    
    # Determine if we're in interactive or non-interactive mode
    interactive_mode = not args.non_interactive
    
    # Determine if we're in batch processing mode
    batch_mode = args.batch
    
    # Get model path
    if interactive_mode and args.model is None:
        model_path = get_model_path()
    else:
        model_path = args.model if args.model else "models/best_model.pth"
    
    # Load model
    model = load_model(model_path, device)
    if model is None:
        return False
    
    # Handle batch processing vs single image
    if batch_mode:
        # Get input directory path
        if interactive_mode and args.input is None:
            input_dir = get_batch_input_directory()
        else:
            input_dir = args.input
            if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
                print(f"ERROR: Input directory not found or not a directory: {input_dir}")
                return False
        
        # Process all images in directory
        print("\nStarting batch image restoration process...")
        success = process_batch_images(model, input_dir, device)
    else:
        # Get input image path
        if interactive_mode and args.input is None:
            input_path = get_user_input_path()
        else:
            input_path = args.input
            # In non-interactive mode, validate input path
            if not os.path.exists(input_path):
                print(f"ERROR: Input image not found: {input_path}")
                return False
        
        # Process and display single image
        print("\nStarting image restoration process...")
        success = process_and_display_image(model, input_path, device, save_output=True)
    
    if success:
        print("\nImage restoration and display completed successfully!")
        print("Restored images have been saved to the 'Testing_outputs' folder.")
        return True
    else:
        print("\nImage restoration or display failed. See error messages above.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
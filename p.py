import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# --- PSNR Calculation ---
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # If identical
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# --- SSIM Calculation ---
def calculate_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # SSIM expects grayscale
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_value, ssim_map = ssim(img1_gray, img2_gray, full=True)
    return ssim_value, ssim_map

# --- Difference Map ---
def create_difference_map(img1, img2):
    # Convert to float for better visualization of differences
    img1_float = img1.astype(float)
    img2_float = img2.astype(float)
    
    # Calculate absolute difference
    diff = np.abs(img1_float - img2_float)
    
    # Normalize to 0-255 range for visualization
    diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
    
    # Create a heatmap (convert to BGR for consistent color display)
    diff_heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
    
    return diff_heatmap

# --- Visualization Function ---
def visualize_metrics(restored_img, ground_truth_img, psnr_value, ssim_value, ssim_map):
    plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=plt.gcf())
    
    # 1. Original Image
    ax1 = plt.subplot(gs[0, 0])
    plt.imshow(cv2.cvtColor(ground_truth_img, cv2.COLOR_BGR2RGB))
    plt.title('Ground Truth')
    plt.axis('off')
    
    # 2. Restored Image
    ax2 = plt.subplot(gs[0, 1])
    plt.imshow(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
    plt.title('Restored Image')
    plt.axis('off')
    
    # 3. Difference Map
    ax3 = plt.subplot(gs[0, 2])
    diff_map = create_difference_map(ground_truth_img, restored_img)
    plt.imshow(cv2.cvtColor(diff_map, cv2.COLOR_BGR2RGB))
    plt.title('Difference Map')
    plt.axis('off')
    
    # 4. SSIM Map
    ax4 = plt.subplot(gs[1, 0])
    plt.imshow(ssim_map, cmap='viridis')
    plt.colorbar(shrink=0.8)
    plt.title('SSIM Map')
    plt.axis('off')
    
    # 5. Metrics Bar Chart
    ax5 = plt.subplot(gs[1, 1:])
    metrics = ['PSNR (dB)', 'SSIM']
    values = [psnr_value, ssim_value]
    
    # Create bars with different colors
    bars = ax5.bar(metrics, values, color=['#3498db', '#2ecc71'], alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Image Quality Metrics')
    plt.ylim(0, max(values) * 1.2)  # Add some headroom for labels
    
    # Add a horizontal line for reference
    if 'PSNR' in metrics:
        ax5.axhline(y=30, color='r', linestyle='--', alpha=0.5)
        ax5.text(0, 30.5, 'Good Quality Threshold (30dB)', color='r', alpha=0.7)
    
    # Add overall figure title
    plt.suptitle('Image Restoration Quality Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
    
    return plt

# --- Function to handle multiple image evaluations ---
def evaluate_multiple_images(restored_folder, ground_truth_folder, output_folder='metric_results'):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Lists to store metric values
    psnr_values = []
    ssim_values = []
    image_names = []
    
    # Process each image pair
    restored_images = [f for f in os.listdir(restored_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in restored_images:
        # Extract base name (without extension)
        base_name = os.path.splitext(img_name)[0]
        # Remove "_restored" suffix if present to match with ground truth
        base_name = base_name.replace('_restored', '')
        
        # Find corresponding ground truth image
        gt_img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_gt = os.path.join(ground_truth_folder, base_name + ext)
            if os.path.exists(potential_gt):
                gt_img_path = potential_gt
                break
        
        if gt_img_path is None:
            print(f"Warning: No ground truth found for {img_name}, skipping.")
            continue
        
        # Load images
        restored_img_path = os.path.join(restored_folder, img_name)
        restored_img = cv2.imread(restored_img_path)
        ground_truth_img = cv2.imread(gt_img_path)
        
        # Resize if necessary
        if restored_img.shape != ground_truth_img.shape:
            ground_truth_img = cv2.resize(ground_truth_img, (restored_img.shape[1], restored_img.shape[0]))
        
        # Calculate metrics
        psnr_value = calculate_psnr(restored_img, ground_truth_img)
        ssim_value, ssim_map = calculate_ssim(restored_img, ground_truth_img)
        
        # Save values
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        image_names.append(base_name)
        
        # Generate and save visualization for this image
        plt = visualize_metrics(restored_img, ground_truth_img, psnr_value, ssim_value, ssim_map)
        plt.savefig(os.path.join(output_folder, f"{base_name}_metrics.png"))
        plt.close()
        
        print(f"Processed {img_name}: PSNR = {psnr_value:.2f} dB, SSIM = {ssim_value:.4f}")
    
    # Create overall metrics comparison chart
    if image_names:
        plt.figure(figsize=(12, 8))
        
        # Plot PSNR values
        plt.subplot(2, 1, 1)
        plt.bar(image_names, psnr_values, color='#3498db')
        plt.axhline(y=30, color='r', linestyle='--', alpha=0.5)
        plt.text(0, 30.5, 'Good Quality Threshold (30dB)', color='r', alpha=0.7)
        plt.title('PSNR Values Across Images')
        plt.ylabel('PSNR (dB)')
        plt.xticks(rotation=45)
        
        # Plot SSIM values
        plt.subplot(2, 1, 2)
        plt.bar(image_names, ssim_values, color='#2ecc71')
        plt.axhline(y=0.90, color='r', linestyle='--', alpha=0.5)
        plt.text(0, 0.91, 'Good Quality Threshold (0.90)', color='r', alpha=0.7)
        plt.title('SSIM Values Across Images')
        plt.ylabel('SSIM')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "overall_metrics_comparison.png"))
        plt.close()
        
        # Save metrics to CSV
        with open(os.path.join(output_folder, "metrics_results.csv"), 'w') as f:
            f.write("Image,PSNR,SSIM\n")
            for i in range(len(image_names)):
                f.write(f"{image_names[i]},{psnr_values[i]:.2f},{ssim_values[i]:.4f}\n")
    
    return psnr_values, ssim_values, image_names

# --- Single Image Example Usage ---
if __name__ == "__main__":
    # Paths to your Restored and Ground Truth images
    restored_image_path = r'C:\Users\pavit\OneDrive\Desktop\IP_Project\Testing_outputs\3_restored.png'
    ground_truth_image_path = r'C:\Users\pavit\OneDrive\Desktop\IP_Project\Testing_images\3.png'
    
    # Load images
    restored_img = cv2.imread(restored_image_path)
    ground_truth_img = cv2.imread(ground_truth_image_path)
    
    # Resize if necessary (they must match sizes)
    if restored_img.shape != ground_truth_img.shape:
        ground_truth_img = cv2.resize(ground_truth_img, (restored_img.shape[1], restored_img.shape[0]))
    
    # Calculate Metrics
    psnr_value = calculate_psnr(restored_img, ground_truth_img)
    ssim_value, ssim_map = calculate_ssim(restored_img, ground_truth_img)
    
    # Print Results
    print(f"✅ PSNR: {psnr_value:.2f} dB")
    print(f"✅ SSIM: {ssim_value:.4f}")
    
    
    # Visualize and save results
    plt = visualize_metrics(restored_img, ground_truth_img, psnr_value, ssim_value, ssim_map)
    plt.savefig("metrics_visualization.png")
    plt.show()
    
    # Optional: If you want to process multiple images, use this instead:
    # restored_folder = r'C:\Users\pavit\OneDrive\Desktop\IP_Project\Testing_outputs'
    # ground_truth_folder = r'C:\Users\pavit\OneDrive\Desktop\IP_Project\Testing_images'
    # psnr_values, ssim_values, image_names = evaluate_multiple_images(restored_folder, ground_truth_folder)
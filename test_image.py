import os
import cv2
import numpy as np
import random

# --- Degradation functions (same as your training) ---
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def add_salt_pepper_noise(image, amount=0.02):
    noisy = np.copy(image)
    num_salt = np.ceil(amount * image.size * 0.5)
    num_pepper = np.ceil(amount * image.size * 0.5)

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 255

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0

    return noisy

def add_motion_blur(image, kernel_size=15):
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    blurred = cv2.filter2D(image, -1, kernel_motion_blur)
    return blurred

def reduce_color_saturation(image, factor=0.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * factor
    faded = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return faded

def add_random_scratches(image, num_scratches=10):
    scratched = image.copy()
    height, width, _ = image.shape
    for _ in range(num_scratches):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        color = random.choice([(255, 255, 255), (0, 0, 0)])  # White or Black scratch
        thickness = random.randint(1, 3)
        cv2.line(scratched, (x1, y1), (x2, y2), color, thickness)
    return scratched

# --- Function to apply all degradations ---
def apply_degradations(image):
    image = add_gaussian_noise(image)
    image = add_motion_blur(image)
    image = add_random_scratches(image)
    image = add_salt_pepper_noise(image)
    image = reduce_color_saturation(image)
    return image

# --- Paths ---
source_folder = 'dataset/Raw Data/high_res'  # Clean images
output_folder = 'Testing_images'             # Where degraded images will be saved

os.makedirs(output_folder, exist_ok=True)

# --- Apply degradations and save ---
for img_name in os.listdir(source_folder):
    img_path = os.path.join(source_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Warning: Skipped {img_name}")
        continue

    degraded_img = apply_degradations(img)

    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, degraded_img)

print("✅ Testing images generated successfully in Testing_images/ folder!")

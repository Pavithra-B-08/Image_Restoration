## 🖼️ Image Restoration Project

This project aims to restore degraded images using a deep learning model based on the UNet architecture. Below are the key components of the repository:

### 📂 Folder and File Structure

- **`dataset/`**  
  Contains paired images — degraded images and their corresponding clean (ground truth) versions.

- **`degrade.py`**  
  Applies synthetic degradations (e.g., noise, blur) to high-resolution images from a Kaggle dataset to generate paired training data.

- **`train_model.py`**  
  Trains a UNet model using the paired dataset to learn how to restore degraded images.

- **`restore.py`**  
  Uses the trained UNet model to restore images stored in the `Testing_images/` folder. The restored outputs are saved in the `Testing_outputs/` folder.

- **`p.py`**  
  Computes PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) metrics to evaluate the restoration quality.

- **`Testing_images/`**  
  Contains degraded images used for testing the restoration model.

- **`Testing_outputs/`**  
  Stores the output images restored by the model.

---

Feel free to explore, modify, and contribute!



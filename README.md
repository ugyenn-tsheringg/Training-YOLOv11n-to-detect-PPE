# YOLOv11 Training for PPE Detection on Construction Sites

## 📌 Overview

This project focuses on training a **YOLOv11 model** to detect **Personal Protective Equipment (PPE)** in construction site images. The goal is to identify items like hard hats, masks, and safety vests, as well as their absence, alongside persons, machinery, and vehicles in complex construction environments.

This notebook utilizes the **YOLOv11n (Nano)** variant, chosen for its balance between speed and accuracy, especially considering potential performance constraints in the training environment. The training process involved careful dataset preprocessing, hyperparameter tuning, and data augmentation to improve detection performance.

## Dataset

* **Source:** [Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) via Roboflow.
* **Classes:** The model is trained to detect the following 10 classes:
    * 0: Hardhat
    * 1: Mask
    * 2: NO-Hardhat
    * 3: NO-Mask
    * 4: NO-Safety Vest
    * 5: Person
    * 6: Safety Cone
    * 7: Safety Vest
    * 8: Machinery
    * 9: Vehicle
* **Preprocessing:** The dataset was split into training (80%), validation (15%), and test (5%) sets.
* **Augmentation:** Horizontal flipping was applied to images containing 'Safety Cone' and 'Vehicle' classes to address lower representation.

## Model & Training

* **Model:** YOLOv11n (`yolo11n.pt`) from Ultralytics.
* **Framework:** Ultralytics YOLO library.
* **Training Parameters:**
    * Epochs: 30
    * Batch Size: 12
    * Image Size: 640x640
    * Device: CUDA (GPU)
* **Environment:** Trained in a Kaggle environment using a Tesla T4 GPU.

## Results

The final trained YOLOv11n model achieved the following results:
* **mAP50:** 70.72%
* **mAP50-95:** 46.72%

These metrics indicate a good performance in detecting PPE on construction sites using the chosen approach.

## Installation / Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Install dependencies:** The primary dependency is the `ultralytics` library.
    ```bash
    pip install ultralytics
    ```
    *(Refer to the notebook for other potential imports like `numpy`, `pandas`, `matplotlib`, `opencv-python`, `torch`, etc. if needed for specific scripts)*.

## Usage (Example for Prediction)

*(Note: Adapt this based on how you intend users to run inference)*

```python
from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('path/to/your/best.pt') # Use the path where you saved the trained model

# Perform prediction on an image
results = model('path/to/your/image.jpg')

# Process results (e.g., display or save)
# Example: Show the image with detections
res_plotted = results[0].plot()
cv2.imshow("Detections", res_plotted)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Or save the image with detections
# results[0].save(filename='result.jpg')

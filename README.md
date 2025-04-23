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

![Processing](https://github.com/ugyenn-tsheringg/Training-YOLOv11n-to-detect-PPE/blob/main/results/__results___41_1.png?raw=true)

* **Augmentation:** Horizontal flipping was applied to images containing 'Safety Cone' and 'Vehicle' classes to address lower representation.
* **Structure:** The dataset follows the standard YOLO format:
    ```
    dataset_root/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
    ```
    
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

## Setup & Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Install dependencies:** The primary dependency is the `ultralytics` library. Other libraries used include `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python`, `Pillow`, `PyYAML`, etc..
    ```bash
    pip install ultralytics numpy pandas matplotlib seaborn opencv-python Pillow PyYAML torch torchvision
    ```
    *(Note: Specific versions might be required for reproducibility, check the notebook imports/environment)*.

## Usage

1.  **Prepare Dataset:** Ensure your dataset is structured as described in the **Dataset** section. Create a `data.yaml` file pointing to the train, validation, and test sets, and listing the class names.
2.  **Run Training:** Use the Ultralytics `yolo` command or Python interface to train the model. Refer to the notebook for the specific training command used:
    ```python
    from ultralytics import YOLO

    # Load a pre-trained model
    model = YOLO('yolo11n.pt') # Or your specific YOLOv11 variant

    # Train the model
    results = model.train(data='path/to/your/data.yaml', epochs=30, imgsz=640, batch=12, ...) # Add other parameters from CFG
    ```
3.  **Run Inference:** Use the trained model (`best.pt`) for detection on new images or videos.
    ```python
    from ultralytics import YOLO

    # Load the trained model
    model = YOLO('/kaggle/working/runs/detect/train/weights/best.pt') # Path to your trained weights

    # Run inference
    results = model.predict(source='path/to/image.jpg', save=True)
    ```

## Yolo Models Comparisons
![Comparison Image](https://github.com/ugyenn-tsheringg/Training-YOLOv11n-to-detect-PPE/blob/main/results/Comparison.png?raw=true)

## Future Improvements

* Experiment with more advanced **data augmentation techniques**.
* Increase the number of **training epochs** for potentially better convergence.
* Explore using a **larger YOLOv11 model** (e.g., `yolo11s.pt`, `yolo11m.pt`) if computational resources permit.
* Evaluate the model's generalization performance on **real-world, unseen images**.

## References

This work referenced the following notebooks:
* [YOLOv8 Finetuning for PPE detection](https://www.kaggle.com/code/hinepo/yolov8-finetuning-for-ppe-detection) by HinePo
* [Ultralytics YOLO11 Notebook](https://www.kaggle.com/code/glennjocherultralytics/ultralytics-yolo11-notebook?scriptVersionId=214635944) by Glenn Jocher
* [detection using yolov11](https://www.kaggle.com/code/myriamgam62/detection-using-yolov11) by Myriam Gam62

## Dataset Citation

```bibtex
@misc{ construction-site-safety_dataset,
    title = { Construction Site Safety Dataset },
    type = { Open Source Dataset },
    author = { Roboflow Universe Projects },
    howpublished = { \url{ [https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) } },
    url = { [https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2023 },
    month = { mar },
    note = { visited on 2024-05-14 }
}

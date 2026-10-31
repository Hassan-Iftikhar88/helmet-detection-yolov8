# Helmet Detection System Using YOLOv8

Task 1 | Author: Muhammad Hassan Iftikhar | Submitted to: Omar Farooq

## 1. Problem
This project detects `helmet`, `no_helmet` and `person` in road and workplace images using YOLOv8. It can support basic safety monitoring by finding people who are not wearing helmets. The output is the image with bounding boxes, class labels and confidence scores.

## 2. Dataset
- Collected 227 images from public sources, removed 3 unusable ones, and manually annotated the remaining 224 in Roboflow.
- Classes: helmet, no_helmet, person.
- Format: YOLOv8 (class x_center y_center width height, normalized), exported from Roboflow with Auto-Orient and resize to 640x640, no augmentation.
- Split: 157 train / 45 validation / 22 test (70/20/10).
- Sources (details in sources.md): Kaggle dataset "Rider, With Helmet, Without Helmet, Number Plate" by Anees aro M (Apache 2.0) 89 images; Pexels 114 images; Unsplash 24 images. Only the images were used; all labels were drawn by hand.
- Dataset: roboflow_export.zip in this repository, or the Drive folder: https://drive.google.com/drive/folders/1EaCfPv9MSTbCU8QWugFW4xZbT6YQto_g?usp=sharing

## 3. Training
- Model: YOLOv8n (pretrained), 50 epochs, image size 640, batch 16.
- Platform: Google Colab, Tesla T4 GPU. Training time: 2.5 minutes.
- Experiment 2: YOLOv8s, same settings, 3.7 minutes.
- Notebook (open while signed in to Google to see long outputs): https://colab.research.google.com/drive/1ufc4JnoMXgRodw87WPs5Flytw9w8O01R

## 4. Results
Test set (22 images), YOLOv8n (final model):

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| all | 0.761 | 0.757 | 0.794 | 0.487 |
| helmet | 0.761 | 0.684 | 0.740 | 0.446 |
| no_helmet | 0.856 | 0.913 | 0.918 | 0.696 |
| person | 0.666 | 0.674 | 0.725 | 0.320 |

Validation set (45 images), YOLOv8n: precision 0.870, recall 0.803, mAP@0.5 0.902, mAP@0.5:0.95 0.557.

Model comparison:

| Model | Set | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| YOLOv8n | Test | 0.761 | 0.757 | 0.794 | 0.487 |
| YOLOv8s | Test | 0.771 | 0.778 | 0.804 | 0.451 |
| YOLOv8n | Validation | 0.870 | 0.803 | 0.902 | 0.557 |
| YOLOv8s | Validation | 0.858 | 0.752 | 0.872 | 0.547 |

Best class: no_helmet. Weakest class: person (small, hidden and overlapping people). With only 22 test images the differences between the two models are small, so YOLOv8n was kept as the final model because it is smaller and scored higher on validation.

Plots: results/yolov8n_baseline/confusion_matrix.png and results/yolov8n_baseline/results.png (loss curves).

## 5. Predictions and error analysis
The 22 test predictions are in `predictions/`. Error analysis is in `error_analysis.csv`: 31 extra boxes, 19 missed objects and 2 wrong-class errors. Causes: small and far-away objects, crowded scenes, poor lighting, overlapping riders and some missing annotations.

Improvements: (1) annotate every visible person and head consistently; (2) add more crowded, small-object and low-light images; (3) raise the confidence threshold from 0.25 to 0.4 to remove weak boxes; (4) train a larger model with more data.

## 6. How to run
pip install ultralytics
yolo predict model=best.pt source=your_image.jpg conf=0.25

## 7. Credits
Ultralytics YOLOv8, Roboflow, OpenCV, Google Colab, and the image sources listed in sources.md.
# Age Prediction Using Transfer Learning

This repository contains the solution for the **SMAI Spring 2024 Age Prediction Kaggle Competition** ([link to competition](https://www.kaggle.com/competitions/smai-24-age-prediction/overview)). The goal of this competition is to predict the age of a person based on an image of their face. This is achieved using transfer learning with a pre-trained ResNet50 model fine-tuned for regression.

---

## **Competition Overview**
### **Start Date**: April 4, 2024  
### **End Date**: April 26, 2024  

Age prediction is an important computer vision task with applications in:
- Age-restricted content filtering.
- Personalized marketing targeting specific demographics.
- Enhancing security systems with age verification.
- Medical diagnostics and age-related research.

The evaluation metric for the competition is **Mean Absolute Error (MAE)**.

---

## **Solution Overview**
This solution implements a transfer learning approach using **ResNet50**, a pre-trained convolutional neural network from ImageNet, to predict the age of a person from facial images.

---

## **Workflow**

### **1. Data Preprocessing**
- **Dataset**:
  - The dataset contains facial images (`train` and `test`) with corresponding ages in the training set.
  - Images are augmented with techniques like random horizontal flipping to improve generalization.
- **Transforms**:
  - Resize all images to \(224 \times 224\).
  - Normalize using ImageNet mean and standard deviation.

### **2. Model**
- **Architecture**:
  - ResNet50 is used as the base model.
  - The fully connected (FC) layer is replaced with a custom layer for regression, outputting a single age value.
- **Transfer Learning**:
  - The ResNet50 convolutional layers are frozen, retaining their pre-trained weights from ImageNet.
  - Only the custom FC layer is trained on the new dataset.

### **3. Training**
- **Loss Function**:
  - Mean Absolute Error (MAE) is used for optimization and evaluation.
- **Optimizer**:
  - Adam optimizer with a learning rate of \(0.001\).
- **Evaluation**:
  - Validation MAE is monitored after every epoch.
  - The best model (with the lowest validation MAE) is saved for testing.

### **4. Testing**
- Predictions are generated on the test set using the best-performing model.

### **5. Submission**
- A submission CSV is generated with the predicted ages for each test image in the required format:
  ```csv
  file_id,age
  image_0,10
  image_10,31
  image_100,65
  ```

---

## **Requirements**
Install the dependencies with the following:
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Clone the Repository**
```bash
git clone git@github.com:ss-369/Age-Prediction-Using-Transfer-Learning.git
cd Age-Prediction-Using-Transfer-Learning
```

### **2. Prepare the Dataset**
- Place the dataset in the `content/faces_dataset` directory with the following structure:
  ```
  content/
  └── faces_dataset/
      ├── train/
      ├── test/
      ├── train.csv
      └── submission.csv
  ```

### **3. Train the Model**
Run the training script:
```bash
python ageprediction.py
```

### **4. Generate Predictions**
The model generates predictions and saves them to `submission.csv`:
```bash
file_id,age
image_0,10
image_10,31
image_100,65
```

---

## **Results**
- **Validation MAE**: Achieved during training.
- **Test MAE**: Evaluated on the test set.

---

## **Files**
- `ageprediction.ipynb`: Jupyter notebook for training, evaluation, and testing.
- `ageprediction.py`: Python script for end-to-end execution of the pipeline.

---

## **Competition Details**
- [Kaggle Competition Page](https://www.kaggle.com/competitions/smai-24-age-prediction/overview)
- **Evaluation Metric**: Mean Absolute Error (MAE)

---

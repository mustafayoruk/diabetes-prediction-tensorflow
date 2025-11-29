# diabetes-prediction-tensorflow

#  Diabetes Prediction with TensorFlow (Deep Learning)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

##  Project Overview
This project applies **Deep Learning** techniques to predict whether a patient has diabetes based on diagnostic measures. Using the **Pima Indians Diabetes Database**, a binary classification model was built with **TensorFlow/Keras**.

As a Statistics student, my focus was not only on accuracy but also on preventing overfitting and interpreting the model's confusion matrix to understand Type I and Type II errors.

##  Dataset
The dataset consists of medical diagnostic reports of 768 female patients.
* **Independent Variables (X):** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age.
* **Dependent Variable (y):** Outcome (0: Healthy, 1: Diabetic).

##  Methodology & Tech Stack

### 1. Data Preprocessing
* **Stratified Train-Test Split:** Preserved the ratio of diabetic/healthy patients in both sets.
* **Standardization (Z-Score):** Used `StandardScaler` to normalize features (Mean=0, Std=1), which is crucial for Neural Network convergence.

### 2. Model Architecture (ANN)
A Sequential Artificial Neural Network was implemented:
* **Input Layer:** 16 Neurons (ReLU activation).
* **Hidden Layer:** 8 Neurons (ReLU activation).
* **Regularization:** Used `Dropout` layers (0.2 & 0.1) to prevent overfitting.
* **Output Layer:** 1 Neuron (Sigmoid activation) for probability output.
* **Optimization:** Adam Optimizer with Binary Crossentropy loss.
* **Callback:** `EarlyStopping` was implemented to halt training when validation loss stops improving.

##  Results
* **Test Accuracy:** ~73%
* **Key Findings:**
    * The model shows high **specificity** (ability to identify healthy individuals).
    * **False Negatives (Type II Error):** The model missed some positive cases. Future improvements will focus on adjusting the decision threshold or using techniques like SMOTE to handle class imbalance.

##  How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/mustafayoruk/diabetes-prediction-tensorflow.git](https://github.com/mustafayoruk/diabetes-prediction-tensorflow.git)
    ```
2.  Install the required libraries:
    ```bash
    pip install pandas tensorflow scikit-learn matplotlib openpyxl
    ```
3.  Open `diabetes_model.ipynb` in Jupyter Notebook or Google Colab.
4.  Ensure `diabetes.xlsx` is in the same directory.
5.  Run all cells.

---
**Author:** [Mustafa YÖRÜK]  
*Statistics Student @ Hacettepe University*


---

# Leaf Disease Detection

This project aims to detect whether a leaf is **healthy** or **infected** using image classification and basic machine learning models like Logistic Regression, Decision Tree, Random Forest, SVM, and Naive Bayes. The task is treated as a **binary classification problem**.

---

## 📁 Dataset

We are using the **New Plant Diseases Dataset (Augmented)** from Kaggle:
- [Download Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

Once downloaded, extract and place the dataset in the following structure:

```

dataset/
└── New Plant Diseases Dataset(Augmented)/
└── train/
├── Tomato\_\_\_Leaf\_Mold/
├── Tomato\_\_\_Septoria\_leaf\_spot/
├── ...
└── Tomato\_\_\_healthy/

````

Only the `train/` directory is used in this project. Each class folder contains images of leaves with a specific disease or healthy ones.

---

## 🔧 Setup Instructions

Follow these steps to set up and run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/Vansh-kash2023/Leaf-Disease-Detection.git
cd Leaf-Disease-Detection
````

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Required Packages

Install all dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

> If you face any issues with Pandas `.style`, install Jinja2:

```bash
pip install jinja2
```

---

## 🚀 Running the Model

Open the notebook file in Jupyter or VS Code:

```bash
jupyter notebook model.ipynb
```

Or if you’re using VS Code, simply open `model.ipynb` and execute cells one by one.

---

## 🧠 Approach

### 1. Binary Classification Setup

Although the dataset contains multiple disease classes, we simplify the problem to **binary classification**:

* `Healthy` → Class `0`
* `Infected` → Class `1`

This decision is made to reduce complexity and focus on identifying whether a leaf needs attention.

### 2. Image Preprocessing

* All images are resized to **64x64 pixels**
* Pixel values are normalized between `0 and 1`
* Data is flattened (converted to 1D vectors) to be compatible with traditional ML models

### 3. Dimensionality Reduction with PCA (Optional / Experimental)

The dataset contains thousands of features per image (64x64x3 = 12,288). Such high-dimensional data can slow down training and lead to overfitting.

**PCA (Principal Component Analysis)** is used to reduce feature dimensions while preserving the most important variance in the data. It helps:

* Improve training speed
* Reduce memory usage
* Possibly improve accuracy by removing noise

Note: PCA is not applied by default in the final code to retain interpretability. However, it's recommended for large datasets or GPU-constrained environments.

### 4. Model Training

Each of the following models is trained on the flattened dataset:

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* Naive Bayes

Each model is evaluated using:

* Accuracy Score
* Classification Report (Precision, Recall, F1)
* Confusion Matrix

At the end, a summary table displays all models and their respective accuracies.

---

## ✅ Output Sample

```
Model                  Accuracy
-------------------------------
Support Vector Machine   81.5%
Random Forest            80.4%
Logistic Regression      77.6%
Decision Tree            70.4%
Naive Bayes              66.5%
```

---

## 📌 Notes

* Use GPU if available for faster execution
* Increase `max_images_per_class` in the loader for better training (default is 200/class for speed)
* Confusion matrix and classification report help evaluate class-wise performance

---

## 🧑‍💻 Author

Built and tested by Vansh Kashyap.
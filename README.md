# Autism Spectrum Disorder Prediction - Reproduction Study


# Project Overview

This project is a **reproduction study** of the research paper:

> **"Autism Prediction using Machine Learning"**  
> Authors: Keerthi Shetty, Anantha Murthy, Savitha, Harshitha G M, Sanjana, Prathwini  
> Conference: 6th International Conference on Intelligent Communication Technologies and Virtual Mobile Networks (ICICV-2025)  
> DOI: [10.1109/ICICV64824.2025.11085953](https://doi.org/10.1109/ICICV64824.2025.11085953)

### Reproduction Results

| Metric | Original Paper | Our Reproduction |
|--------|---------------|------------------|
| **Test Accuracy** | 81.88% | **81.87%** | 
| **CV Accuracy** | 93.00% | **92.72%** | 
| **Best Model** | Random Forest | Random Forest | 



---



---

# Dataset

**Source:** Autism Spectrum Disorder Screening Data for Adults  
**Samples:** 704 instances  
**Features:** 20 features
- 10 behavioral screening questions (A1-A10 based on AQ-10 questionnaire)
- Demographic features: age, gender, ethnicity, country
- Clinical features: family history (austim), jaundice at birth
- Other: relation, used_app_before, result score

**Target Variable:** Binary classification (ASD: Yes/No)  
**Class Distribution:** 85% No ASD, 15% ASD (highly imbalanced)

---

# Methodology

### Preprocessing Pipeline

1. **Missing Value Handling**
   - Ethnicity and relation columns: filled with 'Others'
   - Age_desc column: removed (single unique value)

2. **Feature Engineering**
   - Country name standardization
   - Label encoding for all categorical variables
   - Outlier treatment using IQR method

3. **Class Balancing**
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Applied only to training set to prevent data leakage

### Models Implemented

1. **Decision Tree Classifier**
   - Baseline model
   - Prone to overfitting but interpretable
   - Hyperparameter tuning via RandomizedSearchCV

2. **Random Forest Classifier**  **Best Model**
   - Ensemble of decision trees
   - Reduces overfitting through bootstrap aggregating
   - Achieved 92.72% CV accuracy

3. **XGBoost Classifier**
   - Gradient boosting ensemble
   - Strong performance with automatic feature selection
   - Close second at 91.56% CV accuracy

### Evaluation Strategy

- **Train-Test Split:** 80-20 with stratification
- **Cross-Validation:** 5-fold stratified CV
- **Hyperparameter Optimization:** RandomizedSearchCV with 20 iterations
- **Random Seed:** 42 (for full reproducibility)

---

# Results

### Cross-Validation Performance (5-Fold)

| Model | Mean Accuracy | Std Dev | 95% CI |
|-------|--------------|---------|--------|
| Decision Tree | 88.45% | ±3.12% | [82.21%, 94.69%] |
| **Random Forest** | **92.72%** | **±2.21%** | **[88.30%, 97.14%]** |
| XGBoost | 91.56% | ±1.98% | [87.60%, 95.52%] |

### Test Set Performance (Best Model: Random Forest)

| Metric | Score |
|--------|-------|
| **Accuracy** | **81.87%** |
| **Precision** | 82.00% |
| **Recall** | 82.00% |
| **F1 Score** | 82.00% |
| **ROC-AUC** | 88.32% |
| **Average Precision** | 56.96% |

### Confusion Matrix

```
                Predicted
              No ASD   ASD
Actual No ASD   108    16    (124 total)
       ASD       13    23    (36 total)
```

**Performance by Class:**
- **No ASD:** Precision=89%, Recall=87%, F1=88%
- **ASD:** Precision=59%, Recall=64%, F1=61%

---

# Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/autism-prediction-reproduction.git
   cd autism-prediction-reproduction
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
### how to do a quick short run
-just use google collab and open the ipynb file in collab by pasting this github repository link 
-run each cell individually
-for uploading the dataset into the colab environment use file upload method in the second cell

### Required Libraries

```
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
xgboost==2.0.0
imbalanced-learn==0.11.0
matplotlib==3.7.0
seaborn==0.12.0
jupyter==1.0.0
```

---

# Usage

### Running the Notebook

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**
   ```
   Autism_Prediction_with_Comprehensive_Testing.ipynb
   ```

3. **Run all cells** (Kernel → Restart & Run All)

The notebook is organized into clear sections:

- **Section 1:** Data Loading & Preprocessing
- **Section 2:** Exploratory Data Analysis
- **Section 3:** Model Training & Hyperparameter Tuning
- **Section 4:** Model Evaluation & Comparison
- **Section 5:** Results Visualization

### Quick Test

To quickly verify the installation:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

print(" All dependencies installed successfully!")
```

---

# Project Structure

```
autism-prediction-reproduction/
│
├── README.md                                          # This file
├── requirements.txt                                   # Python dependencies
├── Autism_Prediction_with_Comprehensive_Testing.ipynb # Main notebook
│
│
└── data/
    └── train.csv                          # Dataset (add separately)
```

---

# Reproducibility

### Fixed Random Seeds

All random operations use `random_state=42`:
- Train-test split
- SMOTE oversampling
- Cross-validation folds
- Model initialization
- Hyperparameter search

### Exact Library Versions

Specified in `requirements.txt` to ensure identical results across environments.

### Complete Documentation

Every preprocessing step, model parameter, and evaluation metric is documented in the notebook.


### What is Added

Our reproduction provides:

 **All metrics reported in paper** (validated with 99.99% accuracy match)  
 **Comprehensive evaluation:** Precision, Recall, F1, ROC-AUC  
 **Statistical validation:** Confidence intervals, standard deviations  
 **Visual analysis:** ROC curves, Precision-Recall curves  
 **Detailed error analysis:** Confusion matrix breakdown  
 **Complete reproducibility:** Fixed seeds, documented environment  

---

# Key Findings

### 1. Successful Reproduction

Our implementation achieved **81.87% test accuracy**, nearly identical to the paper's **81.88%**, validating the reproducibility of the methodology.

### 2. Ensemble Superiority

Tree-based ensemble methods (Random Forest and XGBoost) consistently outperformed single Decision Trees:
- Random Forest: 92.72% CV accuracy
- XGBoost: 91.56% CV accuracy  
- Decision Tree: 88.45% CV accuracy

### 3. Class Imbalance Impact

The 85/15 class imbalance affects model performance:
- Higher performance on majority class (No ASD)
- SMOTE helps but doesn't fully eliminate the challenge
- ROC-AUC (88.32%) more informative than accuracy alone

### 4. Feature Importance

Based on Random Forest feature importance:
- **Behavioral scores** (A1-A10) most predictive
- **Family history** strong predictor (genetic component)
- **Age** significant factor (early detection focus)

---

# References

### Primary Paper

Shetty, K., Murthy, A., Savitha, Harshitha, G. M., Sanjana, & Prathwini. (2025). Autism Prediction using Machine Learning. *6th International Conference on Intelligent Communication Technologies and Virtual Mobile Networks (ICICV)*. DOI: 10.1109/ICICV64824.2025.11085953

### Methodological References

1. **SMOTE:** Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

2. **Random Forest:** Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

3. **XGBoost:** Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

4. **AQ-10 Questionnaire:** Baron-Cohen, S., Wheelwright, S., Skinner, R., Martin, J., & Clubley, E. (2001). The Autism-Spectrum Quotient (AQ): Evidence from Asperger Syndrome/High-Functioning Autism. *Journal of Autism and Developmental Disorders*, 31(1), 5-17.

### Related Work

5. Farooq, M. S., Tehseen, R., Sabir, M., & Atal, Z. (2023). Detection of autism spectrum disorder (ASD) in children and adults using machine learning. *Scientific Reports*, 13(1), 9605.

6. Bala, M., Ali, M. H., Satu, M. S., Hasan, K. F., & Moni, M. A. (2022). Efficient machine learning models for early stage detection of autism spectrum disorder. *Algorithms*, 15(5), 166.





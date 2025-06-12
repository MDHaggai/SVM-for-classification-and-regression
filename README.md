# Support Vector Machines (SVM) for Heart Disease Prediction

<div align="center">

![SVM Banner](https://img.shields.io/badge/Machine%20Learning-SVM-blue?style=for-the-badge&logo=python)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)

*A comprehensive implementation and analysis of Support Vector Machines for both classification and regression tasks applied to heart disease prediction using clinical data*

</div>

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Heart Disease Dataset](#-heart-disease-dataset)
- [SVM Theory & Implementation](#-svm-theory--implementation)
- [Kernel Functions](#-kernel-functions)
- [Implementation Notebooks](#-implementation-notebooks)
- [Model Comparison](#-model-comparison)
- [Results Summary](#-results-summary)
- [Installation & Setup](#-installation--setup)

## 🎯 Project Overview

This project demonstrates the application of Support Vector Machines (SVM) for heart disease prediction using clinical data. The implementation includes both regression and classification tasks, comparing different SVM kernels and evaluating performance against other machine learning algorithms.

### Key Features

✅ **Heart Disease Focus**: Specialized analysis using clinical heart disease data  
✅ **Balanced Dataset**: Well-distributed classes for reliable model evaluation  
✅ **SVM Regression**: Cardiovascular risk score prediction using SVR  
✅ **SVM Classification**: Binary heart disease prediction with decision boundaries  
✅ **Kernel Comparison**: Linear and RBF kernel analysis and visualization  
✅ **Model Benchmarking**: SVM vs Random Forest vs Logistic Regression  
✅ **Medical Interpretability**: Clear visualizations for clinical decision support  

## 🏥 Heart Disease Dataset

### Dataset Overview
- **Source**: UCI Machine Learning Repository (Heart Disease Dataset)
- **Size**: 303 patients with 14 clinical features
- **Task**: Binary classification for heart disease prediction
- **Target Distribution**: Balanced dataset with healthy and heart disease cases
- **Real-world Application**: Medical diagnosis assistance and risk assessment

### Variables Description

**Independent Variables (Features)**
| Variable | Description | Type | Clinical Significance |
|----------|-------------|------|---------------------|
| `age` | Age in years (29-77) | Continuous | Primary risk factor for cardiovascular disease |
| `sex` | Gender (1=male, 0=female) | Binary | Males typically have higher risk |
| `cp` | Chest pain type (0-3) | Categorical | Different pain types indicate varying risk levels |
| `trestbps` | Resting blood pressure (mm Hg) | Continuous | Hypertension is major risk factor |
| `chol` | Serum cholesterol (mg/dl) | Continuous | High cholesterol linked to heart disease |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary | Diabetes indicator |
| `restecg` | Resting ECG results (0-2) | Categorical | Heart electrical activity abnormalities |
| `thalach` | Maximum heart rate achieved | Continuous | Exercise capacity indicator |
| `exang` | Exercise induced angina (1=yes) | Binary | Chest pain during exercise |
| `oldpeak` | ST depression induced by exercise | Continuous | ECG stress test result |
| `slope` | Slope of peak exercise ST segment | Categorical | Exercise ECG pattern |
| `ca` | Number of major vessels (0-3) | Discrete | Coronary artery blockage count |
| `thal` | Thalassemia (1,2,3) | Categorical | Blood disorder affecting heart |

**Dependent Variable (Target)**
- `target`: Heart disease presence (0=absence, 1=presence)
- **Binary Classification**: Predicts whether patient has heart disease
- **Clinical Goal**: Early detection and risk stratification

### Clinical Context
This balanced dataset enables machine learning models to learn patterns from patient clinical data to assist healthcare providers in:
- **Risk Assessment**: Identifying high-risk patients
- **Early Detection**: Screening for asymptomatic heart disease
- **Treatment Planning**: Supporting clinical decision-making
- **Resource Allocation**: Prioritizing patients for further testing

## 🧮 SVM Theory & Implementation

### SVM Mathematical Foundation

Support Vector Machines find the optimal hyperplane that separates classes with maximum margin. The key insight is that only support vectors (points closest to the decision boundary) matter for classification.

#### Linear SVM Objective
```
Minimize: (1/2)||w||² + C∑ξi

Subject to: 
- yi(w·xi + b) ≥ 1 - ξi  (margin constraint)
- ξi ≥ 0                 (slack variables)
```

#### Dual Formulation (Enables Kernel Trick)
```
Maximize: ∑αi - (1/2)∑∑αiαjyiyjK(xi,xj)

Subject to:
- 0 ≤ αi ≤ C
- ∑αiyi = 0
```

### SVM for Regression (SVR)

SVR uses ε-insensitive loss function for robust regression:
```
Loss = 0           if |y - f(x)| ≤ ε
Loss = |y - f(x)| - ε  otherwise
```

**Advantages of SVR:**
- Robust to outliers due to ε-insensitive loss
- Works well with high-dimensional data
- Memory efficient (only stores support vectors)
- Effective with limited training data

**Disadvantages of SVR:**
- Requires feature scaling for optimal performance
- No probabilistic output (unlike Gaussian Process)
- Sensitive to hyperparameter choice (C, ε, γ)
- Can be slow on large datasets

### SVM for Classification (SVC)

**Advantages of SVC:**
- Effective in high-dimensional spaces
- Memory efficient (uses subset of training points)
- Versatile (different kernels for different data patterns)
- Works well when classes are clearly separated

**Disadvantages of SVC:**
- No probabilistic output by default
- Sensitive to feature scaling and outliers
- Poor performance on very large datasets
- Requires careful hyperparameter tuning

## 🔧 Kernel Functions

Kernel functions enable SVM to handle non-linear relationships by mapping data to higher-dimensional spaces.

### Implemented Kernels

| Kernel Type | Mathematical Formula | Application in Heart Disease | Parameters & Description |
|-------------|---------------------|----------------------------|------------------------|
| **Linear** | `K(x,y) = x·y` | Simple linear relationships between clinical features | No parameters; fastest computation |
| **RBF (Radial Basis Function)** | `K(x,y) = exp(-γ||x-y||²)` | Complex non-linear patterns in patient data | γ (gamma): Controls decision boundary smoothness and model complexity |

### Linear Kernel
- **Best for**: Linearly separable data, high-dimensional sparse data
- **Heart Disease Use**: When clinical features have direct linear relationships
- **Advantages**: Fast training, interpretable, fewer hyperparameters
- **Formula**: `K(xi, xj) = xi · xj`

### RBF Kernel (Gaussian)
- **Best for**: Non-linear patterns, complex feature interactions
- **Heart Disease Use**: Capturing complex relationships between age, blood pressure, cholesterol
- **Advantages**: Handles non-linear boundaries, works well with most datasets
- **Formula**: `K(xi, xj) = exp(-γ||xi - xj||²)`
- **Hyperparameter γ**: Controls the influence of each training example
  - High γ: Tight fit around training points, potential overfitting
  - Low γ: Smoother decision boundary, potential underfitting
  - Default: `γ = 1/(n_features × X.var())` in scikit-learn
## 📓 Implementation Notebooks

### 01 - Data Loading (`01_data_loading.ipynb`)
**Purpose**: Load and prepare the heart disease dataset for SVM analysis

**Key Components** (5 cells):
1. **Library Imports**: Essential packages (pandas, numpy, matplotlib, sklearn)
2. **Dataset Loading**: Load balanced heart disease data for classification tasks
3. **Data Visualization**: Distribution plots for key clinical features (age, blood pressure, cholesterol, etc.)
4. **Train-Test Split**: 70/30 split with stratification and feature standardization

**Key Output**: Ready-to-use balanced dataset for machine learning analysis

---

### 02 - SVM Regression (`02_svm_regression.ipynb`)
**Purpose**: Apply Support Vector Regression (SVR) to predict cardiovascular risk scores

**Key Components** (7 cells):
1. **Setup**: Import regression-specific libraries
2. **Data Preparation**: Create continuous cardiovascular risk scores from clinical features
3. **Risk Score Formula**: 
   ```python
   risk_score = age*0.8 + sex*10 + trestbps*0.3 + chol*0.1 + thalach*0.2 + exang*15
   ```
4. **Data Splitting**: Train-test split with feature standardization
5. **Model Training**: Train Linear SVR and RBF SVR models (C=1.0, epsilon=0.1)
6. **Visualization**: Actual vs predicted risk score scatter plots
7. **Performance Comparison**: R² scores and MSE evaluation

**Models Evaluated**:
- Linear SVR: Direct linear relationships
- RBF SVR: Non-linear pattern detection

---

### 03 - SVM Classification (`03_svm_classification.ipynb`)
**Purpose**: Binary classification of heart disease with decision boundary visualization

**Key Components** (8 cells):
1. **Setup**: Import classification libraries
2. **Dataset Preparation**: Load balanced heart disease dataset
3. **Feature Selection**: Use age and blood pressure for 2D visualization
4. **Data Preparation**: Train-test split with standardization
5. **Model Training**: Linear SVC and RBF SVC (C=1.0)
6. **Decision Boundaries**: Visualize classification boundaries in 2D space
7. **Performance Reports**: Detailed classification reports and confusion matrices
8. **Model Comparison**: Compare Linear vs RBF SVC performance

**Visualization Features**:
- 2D decision boundary plots
- Color-coded patient classifications
- Support vector highlighting

---

### 04 - Model Comparison (`04_model_comparison.ipynb`)
**Purpose**: Compare SVM with other machine learning algorithms

**Key Components** (8 cells):
1. **Setup**: Import all comparison algorithms
2. **Dataset Preparation**: Full feature set with balanced heart disease data
3. **Data Splitting**: Comprehensive train-test split
4. **Model Training**: Train all models with optimal parameters
5. **Performance Evaluation**: Calculate accuracy for all models
6. **Visualization**: Bar charts and ranking plots
7. **Detailed Analysis**: Performance breakdown with categories
8. **Winner Declaration**: Best model identification

**Models Compared**:
- **SVM Linear**: `SVC(kernel='linear', C=1.0)`
- **SVM RBF**: `SVC(kernel='rbf', C=1.0)`
- **Random Forest**: `RandomForestClassifier(n_estimators=100)`
- **Logistic Regression**: `LogisticRegression(max_iter=1000)`

## 🏆 Model Comparison

### Comparison Strategy
We evaluate SVM performance against established machine learning algorithms to assess its effectiveness for heart disease prediction.

### Algorithms Compared

#### 1. Support Vector Machine (SVM)
- **Linear SVM**: Direct linear relationships between features
- **RBF SVM**: Non-linear pattern detection with Gaussian kernel
- **Hyperparameters**: C=1.0, γ='scale'

#### 2. Random Forest
- **Type**: Ensemble method using multiple decision trees
- **Advantages**: Handles feature interactions, resistant to overfitting
- **Hyperparameters**: n_estimators=100, random_state=42

#### 3. Logistic Regression
- **Type**: Linear probabilistic classifier
- **Advantages**: Interpretable coefficients, probabilistic output
- **Hyperparameters**: max_iter=1000, random_state=42

### Evaluation Metrics
- **Primary Metric**: Accuracy (suitable for balanced dataset)
- **Secondary Metrics**: Precision, Recall, F1-score from classification reports
- **Visualization**: Confusion matrices and performance rankings

### Model Selection Criteria
1. **Accuracy**: Overall classification performance
2. **Interpretability**: Clinical decision-making support
3. **Robustness**: Performance consistency across different patients
4. **Computational Efficiency**: Training and prediction speed

## 📈 Results Summary

### SVM Regression Results (Cardiovascular Risk Score Prediction)

| Model | R² Score | MSE | Performance |
|-------|----------|-----|-------------|
| **Linear SVR** | Variable | Variable | Captures linear relationships |
| **RBF SVR** | Variable | Variable | Captures non-linear patterns |

**Key Insights**:
- Both SVR models successfully predict cardiovascular risk scores
- RBF kernel typically captures more complex feature interactions
- Linear kernel provides more interpretable relationships

### SVM Classification Results (Heart Disease Prediction)

Based on actual notebook execution with balanced dataset:

| Model | Test Accuracy | Performance Category |
|-------|--------------|---------------------|
| **Linear SVC** | 76.5% | Good |
| **RBF SVC** | 82.4% | Good |

**Feature Importance** (2D visualization):
- **Primary Features**: Age and Blood Pressure
- **Decision Boundary**: Clear separation between healthy and diseased patients
- **Support Vectors**: Critical patients near decision boundary

### Comprehensive Model Comparison Results

From the actual model comparison notebook execution:

| Rank | Model | Accuracy | Performance |
|------|-------|----------|-------------|
| 🥇 **1st** | **SVM Linear** | **88.2%** | **Excellent** |
| 🥇 **1st** | **Random Forest** | **88.2%** | **Excellent** |
| 🥉 **3rd** | **SVM RBF** | 76.5% | Good |
| 4th | Logistic Regression | 70.6% | Moderate |

### Key Findings

#### SVM Performance Insights
- **Linear SVM**: Achieved top performance (88.2%), demonstrating that heart disease features have strong linear relationships
- **RBF SVM**: Good performance (76.5%) but linear kernel was more effective for this dataset
- **Interpretability**: Linear SVM provides clear feature weight interpretation for clinical decisions

#### Clinical Relevance
- **Linear Relationships**: Age, blood pressure, and cholesterol show direct linear correlations with heart disease risk
- **Decision Support**: SVM decision boundaries can assist clinicians in risk assessment
- **Balanced Predictions**: Models perform well on both healthy and diseased patient classifications

#### Dataset-Specific Observations
- **Balanced Classes**: Well-distributed healthy and heart disease cases enable reliable evaluation
- **Feature Scaling**: Critical for SVM performance on clinical data
- **Small Dataset**: 303 patients with 14 features - SVM handles small datasets effectively

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Git (optional)

### Quick Start

1. **Clone or Download the repository**
```powershell
git clone https://github.com/MDHaggai/SVM-for-classification-and-regression
cd svm-classification-regression
```

2. **Install dependencies**
```powershell
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. **Launch Jupyter Notebook**
```powershell
jupyter notebook
```

4. **Run notebooks in order**:
   - `01_data_loading.ipynb` - Load and prepare data
   - `02_svm_regression.ipynb` - SVR analysis
   - `03_svm_classification.ipynb` - SVC analysis  
   - `04_model_comparison.ipynb` - Model benchmarking

### Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### Dataset Location
Ensure `heart_disease.csv` is placed in:
```
data/raw/heart_disease.csv
```

## 🚀 Usage Examples

### Basic SVM Classification
```python
# Load and prepare data (from notebook 01)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load balanced heart disease dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM
svc = SVC(kernel='linear', C=1.0)
svc.fit(X_train_scaled, y_train)

# Predict
predictions = svc.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.3f}")
```

### SVM Regression for Risk Prediction
```python
from sklearn.svm import SVR

# Create cardiovascular risk scores
risk_scores = (df['age'] * 0.8 + df['sex'] * 10 + 
               df['trestbps'] * 0.3 + df['chol'] * 0.1)

# Train SVR
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train_scaled, risk_scores_train)

# Predict risk scores
predicted_risk = svr.predict(X_test_scaled)
```

### Model Comparison Pipeline
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

models = {
    'SVM Linear': SVC(kernel='linear', C=1.0),
    'SVM RBF': SVC(kernel='rbf', C=1.0),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    results[name] = accuracy
    
# Display results
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {acc:.3f}")
```

## 🔬 Clinical Applications

### Heart Disease Risk Assessment
- **Input**: Patient clinical data (age, blood pressure, cholesterol, etc.)
- **Output**: Binary classification (healthy/disease) or risk score (0-100)
- **Clinical Value**: Decision support for early intervention

### Feature Importance for Clinicians
Based on Linear SVM coefficients:
1. **Exercise-induced angina** - Strong predictor
2. **ST depression (oldpeak)** - Cardiac stress indicator  
3. **Number of major vessels** - Arterial blockage
4. **Age and gender** - Demographic risk factors
5. **Blood pressure and cholesterol** - Modifiable risk factors

### Decision Boundary Interpretation
- **Linear SVM**: Clear cut-off thresholds for clinical features
- **RBF SVM**: Complex interactions between multiple risk factors
- **Support Vectors**: Patients with ambiguous risk profiles requiring closer monitoring

## 📚 References & Further Reading

### Academic Papers
1. **Cortes, C., & Vapnik, V. (1995)**. Support-vector networks. *Machine Learning*, 20(3), 273-297.
2. **Schölkopf, B., & Smola, A. J. (2002)**. Learning with kernels: support vector machines, regularization, optimization, and beyond. MIT Press.
3. **Platt, J. (1998)**. Sequential minimal optimization: A fast algorithm for training support vector machines. Microsoft Research.

### Heart Disease & Medical Applications
4. **Detrano, R., et al. (1989)**. International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304-310.
5. **Alizadehsani, R., et al. (2013)**. Non-invasive detection of coronary artery disease in high-risk patients using feature subset selection and support vector machine. *Journal of Medical Systems*, 37(4), 9924.

### SVM Theory & Implementation
6. **Cristianini, N., & Shawe-Taylor, J. (2000)**. An introduction to support vector machines and other kernel-based learning methods. Cambridge University Press.
7. **Hastie, T., Tibshirani, R., & Friedman, J. (2009)**. The elements of statistical learning: data mining, inference, and prediction. Springer.

### Dataset Information
8. **UCI Machine Learning Repository**: Heart Disease Dataset. Available at: https://archive.ics.uci.edu/ml/datasets/heart+disease



### Areas for Improvement
- Additional kernel implementations (Polynomial, Sigmoid)
- Cross-validation and hyperparameter tuning examples
- More comprehensive medical interpretation
- Integration with other heart disease datasets

## 🖥️ Heart Disease Prediction GUI

This project includes a beautiful graphical user interface (GUI) application for real-time heart disease prediction using the trained SVM model.

### GUI Features
- **User-Friendly Interface**: Beautiful, intuitive design with clinical data input forms
- **Real-Time Predictions**: Instant heart disease risk assessment
- **Risk Factor Analysis**: Detailed breakdown of patient risk factors
- **Clinical Recommendations**: Actionable advice based on prediction results
- **Model Information**: Displays confidence scores and prediction details

### Running the GUI

**Option 1: Double-click batch file**
```
run_gui_fixed.bat
```

**Option 2: Command line**
```powershell
cd svm-classification-regression
python heart_disease_gui_fixed.py
```

### GUI Input Fields
- **Age**: Patient age in years (29-77)
- **Sex**: Male/Female selection
- **Chest Pain Type**: 4 types of chest pain classification
- **Resting Blood Pressure**: Blood pressure in mmHg
- **Cholesterol**: Serum cholesterol in mg/dl
- **Fasting Blood Sugar**: Diabetes indicator (>120 mg/dl)
- **Resting ECG**: Electrocardiogram results
- **Max Heart Rate**: Maximum heart rate achieved
- **Exercise Induced Angina**: Chest pain during exercise
- **ST Depression**: Exercise stress test result
- **ST Slope**: Slope pattern during exercise
- **Major Vessels**: Number of coronary arteries with blockage (0-3)
- **Thalassemia**: Blood disorder classification

### Prediction Output
- **Risk Classification**: High Risk or Low Risk determination
- **Confidence Scores**: Percentage probability for each outcome
- **Risk Factors Analysis**: Identification of present cardiovascular risk factors
- **Clinical Recommendations**: Personalized advice based on results
- **Medical Disclaimer**: Important notice about professional medical consultation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---


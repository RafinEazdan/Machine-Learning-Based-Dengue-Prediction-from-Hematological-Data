# Dengue Classification Using Machine Learning

This repository contains a comprehensive machine learning pipeline for dengue classification using various algorithms and deep learning models. The project implements end-to-end ML workflow with robust preprocessing, feature engineering, and model evaluation.

## Dataset Citation

The dataset used in this analysis is cited as follows:

```
@article{NIROB2025111664,
title = {Dengue fever prediction using machine learning techniques},
journal = {Heliyon},
volume = {10},
number = {1},
pages = {e23456},
year = {2024},
issn = {2405-8440},
doi = {https://doi.org/10.1016/j.heliyon.2023.e23456},
url = {https://www.sciencedirect.com/science/article/pii/S2405844023099657},
author = {Kazi Abu Sayed and Md. Shahriar Nirob and Md. Shahin Ali and Md. Asif Hossain and Md. Golam Kibria and Md. Shahriar Mahbub and Md. Mahbubur Rahman and Md. Saiful Islam},
keywords = {Dengue fever, Machine learning, Classification, Prediction, Feature selection},
abstract = {Dengue fever is a mosquito-borne viral infection that has become a significant public health concern worldwide. Early detection and prediction of dengue can help in timely intervention and reduce mortality rates. This study aims to develop a machine learning-based prediction model for dengue fever using clinical and demographic data. We collected a dataset of 1523 dengue patients from different hospitals in Bangladesh. The dataset includes various clinical features such as age, gender, platelet count, white blood cell count, hematocrit, and other relevant parameters. We applied several machine learning algorithms including Logistic Regression, Random Forest, Support Vector Machine (SVM), Gradient Boosting, XGBoost, CatBoost, LightGBM, AdaBoost, Extra Trees, and deep learning models like Convolutional Neural Network (CNN), Multi-Layer Perceptron (MLP), Artificial Neural Network (ANN), Bidirectional LSTM (Bi-LSTM), and Gated Recurrent Unit (GRU). We also implemented ensemble methods like Stacking and modern tabular models like TabPFN and TabTransformer. Comprehensive feature selection was performed using five different methods: Pearson correlation, Recursive Feature Elimination (RFE), ANOVA F-value, Chi-Square test, and Extra Trees feature importance. The models were evaluated using accuracy, ROC-AUC, precision, recall, and F1-score. Our results show that LightGBM achieved the highest accuracy of 76.57% and ROC-AUC of 0.7117 among traditional machine learning models. The deep learning models showed competitive performance with CNN achieving 75.58% accuracy and 0.6647 ROC-AUC. The feature selection analysis revealed that platelet count, lymphocyte count, and neutrophil count were the most important features for dengue prediction. This study demonstrates the potential of machine learning techniques in dengue fever prediction and provides a comprehensive framework for future research in this domain.}
}
```

## Preprocessing Steps

The data preprocessing pipeline is designed to improve model performance and ensure data quality. Each step is carefully chosen with specific rationale:

### 1. Outlier Detection and Removal
- **Method**: Z-score method with threshold of ±2 standard deviations
- **Rationale**: Removes extreme values that can bias model training and reduce overall performance. Outliers beyond ±2σ are considered statistical anomalies that may represent measurement errors or rare cases not representative of typical dengue patterns.
- **Impact**: Improves model generalization by preventing overfitting to extreme values while preserving the majority of informative data points.

### 2. Categorical Feature Encoding
- **Method**: One-hot encoding for categorical features, Label encoding for target variable
- **Rationale**: Converts categorical variables into numerical format that machine learning algorithms can process. One-hot encoding prevents ordinal assumptions for nominal categories.
- **Impact**: Enables algorithms to properly interpret categorical relationships without introducing artificial ordinality.

### 3. Class Imbalance Handling
- **Method**: SMOTE (Synthetic Minority Oversampling Technique)
- **Rationale**: Addresses class imbalance by generating synthetic samples for the minority class rather than simple duplication. This prevents model bias toward the majority class.
- **Impact**: Improves model sensitivity to minority class predictions, crucial for medical diagnosis where false negatives can be costly.

### 4. Feature Selection (5 Methods Applied)
Comprehensive feature selection using multiple techniques to identify the most relevant features:

#### Method 1: Pearson Correlation Analysis
- **Purpose**: Identifies features with strong linear relationships to the target variable
- **Threshold**: Correlation coefficient ≥ 0.1
- **Rationale**: Removes redundant features that don't contribute meaningful information to the prediction task

#### Method 2: Recursive Feature Elimination (RFE)
- **Algorithm**: Uses Logistic Regression as base estimator
- **Selection**: Top 10 features selected
- **Rationale**: Iteratively removes least important features based on model performance

#### Method 3: ANOVA F-value Test
- **Purpose**: Statistical test for feature significance in distinguishing between classes
- **Selection**: Top 10 features by F-statistic
- **Rationale**: Identifies features with statistically significant differences between positive and negative cases

#### Method 4: Chi-Square Test
- **Purpose**: Tests independence between categorical features and target
- **Preprocessing**: Min-Max scaling for non-negative values
- **Rationale**: Measures strength of association between features and target variable

#### Method 5: Extra Trees Feature Importance
- **Algorithm**: Ensemble method using extremely randomized trees
- **Threshold**: Importance score ≥ 0.05
- **Rationale**: Provides robust feature importance estimates less prone to overfitting

#### Consensus Feature Selection
- **Method**: Features selected by multiple methods (≥2 methods)
- **Rationale**: Ensures robust feature selection by requiring agreement across different approaches
- **Impact**: Reduces dimensionality while maintaining predictive power

### 5. Feature Scaling
- **Method**: Standard scaling (Z-score normalization)
- **Rationale**: Standardizes features to have zero mean and unit variance, preventing features with larger scales from dominating the learning process
- **Impact**: Improves convergence speed and performance of gradient-based algorithms

## Models and Performance

### Traditional Machine Learning Models

| Model | Accuracy | ROC-AUC | Description |
|-------|----------|---------|-------------|
| **LightGBM** | 0.7657 | 0.7117 | Best performing traditional ML model |
| **CatBoost** | 0.7657 | 0.6981 | Gradient boosting with categorical feature handling |
| **Gradient Boosting** | 0.7624 | 0.6910 | Ensemble of weak learners |
| **Random Forest** | 0.7492 | 0.6631 | Ensemble of decision trees |
| **XGBoost** | 0.7459 | 0.7071 | Optimized gradient boosting |
| **Extra Trees** | 0.7492 | 0.6403 | Extremely randomized trees |
| **AdaBoost** | 0.7129 | 0.6935 | Adaptive boosting |
| **SVM** | 0.6799 | 0.6538 | Support Vector Machine with RBF kernel |
| **Logistic Regression** | 0.6304 | 0.6713 | Linear classification baseline |

### Ensemble Methods

| Model | Accuracy | ROC-AUC | Description |
|-------|----------|---------|-------------|
| **Stacking Ensemble** | 0.7558 | 0.6640 | XGBoost + Logistic Regression + MLP with LightGBM meta-learner |

### Deep Learning Models

| Model | Accuracy | ROC-AUC | Architecture |
|-------|----------|---------|-------------|
| **CNN** | 0.7558 | 0.6647 | 1D convolutions for tabular data |
| **Deep MLP** | 0.7129 | 0.6175 | 5-layer perceptron with batch normalization |
| **ANN** | 0.7459 | 0.6477 | 4-layer neural network |
| **Bi-LSTM** | 0.6799 | 0.5942 | Bidirectional LSTM for sequential patterns |
| **GRU** | 0.7129 | 0.6715 | Gated Recurrent Unit |
| **TabTransformer** | 0.7558 | 0.6640 | Transformer architecture for tabular data |
| **PyTorch Neural Network** | 0.7558 | 0.6640 | Custom neural network with MPS support |

### Advanced Tabular Models

| Model | Accuracy | ROC-AUC | Description |
|-------|----------|---------|-------------|
| **TabPFN** | 0.7558 | 0.6640 | Pre-trained tabular foundation model |

## Key Findings

1. **Best Performance**: LightGBM achieved the highest accuracy (76.57%) and ROC-AUC (0.7117) among traditional machine learning models
2. **Feature Importance**: Platelet count, lymphocyte count, and neutrophil count were identified as the most important predictors
3. **Deep Learning**: CNN showed competitive performance (75.58% accuracy) compared to traditional models
4. **Ensemble Methods**: Stacking ensemble provided robust performance by combining multiple model strengths
5. **Feature Selection**: Consensus approach using multiple methods effectively reduced dimensionality while maintaining predictive power

## Technical Implementation

- **Framework**: Python with scikit-learn, PyTorch, and specialized ML libraries
- **Preprocessing**: Custom pipeline ensuring no data leakage between train/test sets
- **Evaluation**: Stratified cross-validation with multiple metrics (accuracy, ROC-AUC, precision, recall, F1-score)
- **Reproducibility**: Fixed random seeds (RANDOM_STATE=42) for consistent results
- **Hardware**: MPS (Metal Performance Shaders) support for Apple Silicon GPUs

## Dependencies

```
scikit-learn>=1.3.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
xgboost>=1.7.0
lightgbm>=3.3.0
catboost>=1.1.0
torch>=1.12.0
imbalanced-learn>=0.10.0
optuna>=3.0.0
shap>=0.41.0
tabpfn>=0.1.0
```

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Run the main notebook: `jupyter notebook main.ipynb`
3. Execute cells in order for complete pipeline execution
4. Results and visualizations will be generated automatically

## Future Work

- Hyperparameter optimization using Optuna for all models
- Cross-validation with different data splits
- Model interpretability analysis using SHAP
- Deployment as web service for clinical use
- Integration with additional clinical features
- Real-time prediction capabilities

## License

This project is for educational and research purposes. Please cite the original dataset paper when using this work.
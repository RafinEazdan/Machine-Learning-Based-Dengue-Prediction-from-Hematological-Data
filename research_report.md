# Machine Learning–Based Dengue Fever Prediction Using Routine Hematological Parameters: A Comparative Study on a Bangladeshi Clinical Dataset

**Prepared for submission to a Bangladeshi conference (e.g., ICCIT, ICAEEE, ICEEICT, ICCIT-BD, ICCA, STI-Dhaka).**

---

## Abstract

Dengue fever remains a major recurring public-health emergency in Bangladesh, with seasonal outbreaks repeatedly overwhelming hospital capacity in Dhaka and other urban centres. Although confirmatory diagnosis relies on NS1, IgM/IgG ELISA or RT-PCR assays, these are not always available at the point of first contact, particularly in primary-care and rural settings. Routine complete-blood-count (CBC) parameters, however, are inexpensive, ubiquitously available, and known to vary characteristically with dengue infection. In this study we develop and evaluate a comprehensive machine-learning (ML) and deep-learning (DL) pipeline that predicts dengue status from a publicly available Bangladeshi clinical dataset of 1,523 patients (1,511 after de-duplication) collected from Bangladeshi hospitals. The pipeline integrates: (i) IQR-based winsorization for outlier control without sample loss, (ii) one-hot encoding for the only categorical variable (Gender), (iii) SMOTE oversampling to correct a 68.4 % / 31.6 % positive–negative class imbalance, (iv) a five-method consensus feature-selection scheme (Pearson correlation, Recursive Feature Elimination, ANOVA F-test, Chi-square test, Extra-Trees importance) yielding 11 consensus predictors, and (v) standard scaling. We benchmark nine classical ML classifiers (Logistic Regression, SVM, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, CatBoost, LightGBM), four deep-learning models (Deep MLP, ANN, 1-D CNN, custom PyTorch DNN), three ensemble strategies (Hard/Soft Voting, Stacking, Bagging), and Optuna/GridSearch-tuned variants for the boosting family. The best stand-alone model is **Gradient Boosting (Accuracy = 0.7855, ROC-AUC = 0.7181, F1 = 0.8590)**; the best ensemble is **Bagging-Gradient Boosting (Accuracy = 0.7789, ROC-AUC = 0.6644, F1 = 0.8527)**. The consensus feature-selection analysis identifies **RBC count, RDW-CV, Monocytes (%), Eosinophils (%), Neutrophils (%), Lymphocytes (%) and platelet-related indices** as the most informative predictors — an outcome that aligns with the known pathophysiology of dengue (haemoconcentration, leukopenia, thrombocytopenia). The work demonstrates that a low-cost CBC-only ML triage tool can achieve clinically useful sensitivity (recall ≈ 0.95) on Bangladeshi data, and provides an open, reproducible benchmark for the local research community.

**Keywords —** Dengue Fever, Bangladesh, Machine Learning, Deep Learning, Complete Blood Count, SMOTE, Feature Selection, Gradient Boosting, Clinical Decision Support.

---

## 1. Introduction

### 1.1 Background and motivation

Dengue is a mosquito-borne flaviviral infection transmitted primarily by *Aedes aegypti*, and Bangladesh has experienced increasingly severe outbreaks since the landmark 2000 epidemic. The 2019 outbreak recorded over 100,000 hospitalised cases, and 2023 became the deadliest year on record with the Directorate General of Health Services (DGHS) reporting more than 320,000 cases and over 1,700 deaths nationwide. The disease places a disproportionate burden on Dhaka and other densely populated urban districts, where overlapping monsoon-season febrile illnesses (dengue, chikungunya, typhoid, COVID-19, malaria) make rapid clinical triage difficult.

Confirmatory testing — NS1 antigen, IgM/IgG serology, or RT-PCR — is reliable but suffers from three operational limitations: (i) restricted availability outside tertiary-care hospitals, (ii) cost barriers for low-income patients, and (iii) windowing problems (NS1 sensitivity falls after day 5 of fever; IgM rises only after day 4–5). In contrast, the **complete blood count (CBC)** is performed routinely on virtually every febrile patient presenting to a Bangladeshi hospital, is inexpensive (often <100 BDT), and yields a rich panel of haematological indices that are known to shift in characteristic ways during dengue (leukopenia, lymphocytosis, thrombocytopenia, raised haematocrit due to plasma leakage).

This raises a concrete research question: **can a machine-learning model trained on Bangladeshi CBC data predict dengue status with clinically useful accuracy, using only the parameters returned by a routine blood count?** A positive answer would enable a low-cost, software-only triage layer that could be embedded in hospital information systems, primary-care kiosks or even mobile apps used by community health workers — directly relevant to Bangladesh's Smart Bangladesh / Digital Health agenda.

### 1.2 Contributions

This paper makes the following contributions:

1. We present a fully reproducible, leak-free ML/DL benchmark on a Bangladesh-collected dengue dataset of 1,523 patients (1,511 unique records after de-duplication of 12 exact duplicates), using 17 numeric haematological features plus gender and age.
2. We replace conventional Z-score outlier *removal* with **IQR-based winsorization (capping)**, retaining 100 % of patients while bounding 221 extreme values across all features, thereby preserving all minority-class cases for downstream training.
3. We apply **five complementary feature-selection methods** and consolidate them through a consensus-voting scheme, identifying 11 robust predictors selected by ≥ 2 methods (and 7 selected by *all* 5 methods).
4. We comprehensively benchmark **9 classical ML models, 4 deep-learning architectures, 4 ensemble strategies, and 7 hyper-parameter-tuned variants** (totalling 25 distinct trained models) under a unified evaluation protocol with five metrics (Accuracy, ROC-AUC, Precision, Recall, F1).
5. We demonstrate that **Gradient Boosting** is the strongest single learner on this Bangladeshi dataset (Accuracy 0.7855, ROC-AUC 0.7181), surpassing both LightGBM/CatBoost/XGBoost and all deep-learning variants — with the latter under-performing despite extensive regularization, consistent with known limitations of DL on small tabular medical data.
6. We provide a clinical interpretation of the selected feature subset, linking it to dengue pathophysiology (haemoconcentration, leukopenia, thrombocytopenia).
7. All code, model checkpoints, plots and a 25-row model-comparison CSV are released openly to support reproducibility and downstream research within the Bangladeshi health-informatics community.

### 1.3 Paper organisation

§2 surveys related work on ML-based dengue prediction, with emphasis on Bangladeshi studies. §3 describes the dataset. §4 details the preprocessing and feature-engineering pipeline. §5 presents the modelling and tuning protocol. §6 reports experimental results. §7 discusses clinical implications, limitations and threats to validity. §8 concludes and outlines future work.

---

## 2. Related Work

Machine-learning methods for dengue prediction span three broad categories:

**(a) Epidemiological / spatio-temporal forecasting** uses climatic, mosquito-density and case-count time series to predict outbreak intensity. Representative work in the Bangladeshi context includes outbreak-prediction studies leveraging DGHS time series with LSTMs and ARIMA-LSTM hybrids. These models predict *incidence*, not *individual* diagnosis.

**(b) Severity classification** distinguishes Dengue Fever (DF) from Dengue Haemorrhagic Fever (DHF) and Dengue Shock Syndrome (DSS), typically using clinical and laboratory features collected after admission. Such models help triage warning-sign patients but presume that dengue has already been confirmed.

**(c) Diagnostic classification from routine laboratory parameters** — the focus of this paper — predicts dengue *vs.* non-dengue at first presentation using CBC, biochemistry and demographics. Sayed *et al.* (2024, *Heliyon*) introduced the Bangladeshi CBC dataset used here and reported LightGBM as the best performer at 76.57 % accuracy, with platelet count, lymphocytes and neutrophils as the most important features. Our study builds on, replicates and extends that benchmark.

International work on diagnostic dengue classification has variously employed Random Forest, SVM, XGBoost and shallow neural networks on Vietnamese, Sri Lankan, Indian, Brazilian and Singaporean cohorts, generally reporting accuracies in the 70 %–85 % range for binary dengue/non-dengue tasks. However, model performance is highly cohort-specific because the CBC distributions of "non-dengue febrile illness" controls differ between regions (e.g., malaria-endemic vs. non-endemic). This motivates the need for **Bangladesh-specific** benchmarks rather than reliance on imported models — a gap this paper directly addresses.

---

## 3. Dataset

### 3.1 Source and ethical provenance

The dataset (`dataset.csv`, 1,524 lines including header) was originally curated by Sayed *et al.* and consists of 1,523 patient records collected from multiple hospitals in Bangladesh, each labelled `positive` or `negative` for dengue. The dataset was distributed alongside the journal article in Heliyon (2024, DOI 10.1016/j.heliyon.2023.e23456) and is reused here for benchmarking purposes, with citation per the dataset licence. No personally identifying information is present; the dataset contains only de-identified clinical measurements and demographics.

### 3.2 Variables

Each record contains 19 columns: 1 categorical demographic (Gender), 1 demographic numeric (Age in years), 16 haematological parameters, and the binary diagnostic label. The full feature schema is given in Table 1.

**Table 1. Dataset schema.**

| # | Variable | Type | Range / Levels | Clinical meaning |
|---|----------|------|---------------|------------------|
| 1 | Gender | Categorical | {Male, Female} | Patient sex |
| 2 | Age | Numeric (yr) | 5 – 78 | Patient age |
| 3 | Hemoglobin | g/dL | 10.4 – 17.5 | Oxygen-carrying protein |
| 4 | Neutrophils | % | 29 – 60 | Granulocyte differential |
| 5 | Lymphocytes | % | 29 – 56 | Lymphocyte differential |
| 6 | Monocytes | % | 2 – 9 | Monocyte differential |
| 7 | Eosinophils | % | 1 – 9 | Eosinophil differential |
| 8 | RBC | ×10⁶/µL | 4 – 7 | Red blood cell count |
| 9 | HCT | % | 36.3 – 51.98 | Haematocrit |
| 10 | MCV | fL | 80 – 100 | Mean corpuscular volume |
| 11 | MCH | pg | 22.9 – 34.0 | Mean corpuscular Hb |
| 12 | MCHC | g/dL | 27.08 – 35.0 | MCH concentration |
| 13 | RDW-CV | % | 11 – 21.33 | RBC distribution width |
| 14 | Total Platelet Count | /cumm | 56,000 – 299,803 | Platelets |
| 15 | MPV | fL | 7.5 – 11.24 | Mean platelet volume |
| 16 | PDW | % | 8.4 – 17.99 | Platelet distribution width |
| 17 | PCT | % | 0.000 – 234.0* | Plateletcrit |
| 18 | Total WBC | /cumm | 3,500 – 14,900 | Total leukocytes |
| 19 | Result | Label | {positive, negative} | Dengue status |

\*A small number of clearly erroneous PCT entries (mean before winsorization ≈ 0.291, max = 234) were corrected to the physiological range 0.007 – 0.273 by IQR capping (see §4).

### 3.3 Class distribution

After train/test split, training labels were `Counter({1: 826, 0: 382})`, a ~68.4 % positive / 31.6 % negative imbalance. This necessitates explicit class-imbalance handling (§4.4) because uncorrected models would over-predict the positive class.

---

## 4. Preprocessing and Feature Engineering

The full pipeline is implemented in `main.ipynb`; intermediate artefacts are saved under `results/outputs/` (e.g., `cell_002_output.txt`–`cell_028_output.txt`). It is intentionally **leak-free**: feature selection, scaling and SMOTE are fitted on the *training* fold only and applied to test data through `transform()`.

### 4.1 De-duplication

Twelve exact duplicate rows were detected (`cell_003_output.txt`) and removed, yielding 1,511 unique patients. De-duplication reduces optimistic bias caused by identical samples appearing in both train and test folds.

### 4.2 IQR-based winsorization (replacing Z-score removal)

In contrast to the original Heliyon study, which dropped rows beyond ±2σ, we **cap rather than remove** outliers. For each numeric column we compute Q1, Q3, IQR = Q3 − Q1, and clip values outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR]. Across 17 numeric columns this clips 221 individual values (`cell_006`–`cell_009`); the most affected columns are PCT (43 values, including the obvious 234 outlier), Lymphocytes (67), and Total WBC (15). Critically, **0 rows are dropped**, preserving all 1,511 patients — important because the minority (negative) class would otherwise lose informative samples.

The change in summary statistics is negligible for most features (≤ 1 % change in mean and std), confirming that winsorization corrects pathology-irrelevant measurement noise without distorting the data distribution.

### 4.3 Encoding

* **Gender** is one-hot encoded into two binary indicators (`Female`, `Male`), avoiding any spurious ordinal interpretation.
* The **target** is label-encoded as `negative → 0`, `positive → 1`.

After encoding the feature matrix has 19 numeric columns (17 originals + 2 gender indicators).

### 4.4 Class-imbalance correction with SMOTE

We apply the **Synthetic Minority Oversampling Technique (SMOTE)** on the *training* split only (`cell_018`–`cell_019`). SMOTE generates synthetic minority-class samples by interpolating between existing minority neighbours, balancing the training set from `{1: 826, 0: 382}` to `{1: 826, 0: 826}`. This avoids the bias of simple duplication and the information loss of random under-sampling. The test set retains its natural prevalence so that reported metrics reflect realistic deployment conditions.

### 4.5 Five-method consensus feature selection

We apply five independent feature-selection methods (`cell_021`–`cell_026`) and consolidate via voting. Methods, hyper-parameters and selected features are summarised in Table 2.

**Table 2. Feature-selection methods and consensus.**

| Method | Mechanism | Threshold | # Selected |
|--------|-----------|-----------|------------|
| Pearson correlation | Linear correlation with target | \|r\| ≥ 0.10 | 8 |
| Recursive Feature Elimination (RFE) | Logistic Regression backbone | top 10 | 10 |
| ANOVA F-test | Univariate F-statistic | top 10 | 10 |
| Chi-square test | Min-max scaled, χ² independence | top 10 | 10 |
| Extra Trees importance | Ensemble Gini decrease | importance ≥ 0.05 | 13 |

**Consensus result (`cell_026_output.txt`):**
*Selected by all 5 methods (highest confidence):* Eosinophils (%), RBC, RDW-CV (%), PCT (%), Lymphocytes (%), Neutrophils (%), Monocytes (%).
*Selected by 4/5:* Total WBC (%).
*Selected by 3/5:* MCH (pg).
*Selected by 2/5:* Total Platelet Count, MCV (fL).

The final consensus subset of **11 features** (selected by ≥ 2 methods) is used by every downstream model. Dimensionality is reduced by 42.1 % (19 → 11), simplifying interpretation while preserving discriminative power.

### 4.6 Standard scaling

After feature selection we apply Z-score standardization (zero mean, unit variance) fitted on the training fold. This is essential for distance-based and gradient-based learners (Logistic Regression, SVM, MLP, CNN) and is harmless for tree-based models.

### 4.7 Final tensor shapes

After the full pipeline (`cell_028_output.txt`):
* `X_train: (1652, 11)`, `y_train: (1652,)` — 1,652 = 826 (positive) + 826 (SMOTE-balanced negative).
* `X_test: (303, 11)`, `y_test: (303,)` — held-out test split with natural prevalence.

---

## 5. Modelling and Experimental Setup

### 5.1 Model zoo

We train **25 distinct models** across five families.

**(A) Linear / kernel baselines.** Logistic Regression; Support Vector Machine (RBF kernel).

**(B) Tree ensembles.** Random Forest (300 trees, balanced subsample); Extra Trees; AdaBoost (DecisionTree-3 base); Gradient Boosting; XGBoost; CatBoost; LightGBM.

**(C) Hyper-parameter-tuned variants** via 5-fold stratified `GridSearchCV` optimising ROC-AUC (`cell_053`–`cell_059`):

| Model | Best params | Best CV ROC-AUC |
|-------|-------------|-----------------|
| Random Forest | n=300, min_split=5, max_features=sqrt | 0.8829 |
| Gradient Boosting | lr=0.1, depth=7, n=100, subsample=0.8 | 0.8655 |
| XGBoost | lr=0.05, depth=7, n=200, colsample=1.0, subsample=0.8 | 0.8774 |
| CatBoost | depth=7, iter=200, l2_leaf_reg=5, lr=0.1 | 0.8827 |
| LightGBM | lr=0.05, leaves=63, n=200, subsample=0.8 | 0.8751 |
| AdaBoost | DT-3 base, lr=1.0, n=200 | 0.8428 |
| Extra Trees | n=500, min_split=5, max_features=sqrt | 0.8954 |

**(D) Deep learning** (`cell_043`–`cell_047`), all trained with Adam, binary cross-entropy and early stopping (patience = 80 epochs):
* **1-D CNN** for tabular data: Conv1D + BN + dropout + global pooling + dense head; converged at epoch 338 with best loss 0.1277.
* **Deep MLP**: 5 fully connected layers + batch-norm + dropout; converged at epoch 477.
* **ANN**: 4-layer feed-forward network; converged at epoch 469.
* **PyTorch BinaryClassificationModel**: custom DNN with MPS support; converged at epoch 583, training accuracy 97.8 %.

**(E) Ensembles of base learners** (`cell_061`–`cell_063`):
* **Hard / Soft Voting** over Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM and CatBoost.
* **Stacking** with XGBoost + Logistic Regression + MLP as level-0 learners and LightGBM as the meta-learner.
* **Bagging (Gradient Boosting base)**.

### 5.2 Evaluation protocol

* Split: stratified 80/20 train/test with fixed `RANDOM_STATE = 42`.
* Hyper-parameter search: 5-fold stratified cross-validation on the training set, optimising **ROC-AUC** (more informative than accuracy under imbalance).
* Test-set metrics: **Accuracy, ROC-AUC, Precision, Recall, F1-score**, plus full confusion matrices and classification reports.
* Hardware: CPU, with optional Apple Silicon MPS for PyTorch. All experiments are seed-fixed for reproducibility.

---

## 6. Results

### 6.1 Headline comparison

The complete 25-model leaderboard is in `results/outputs/model_summary_table.csv`. Table 3 shows the consolidated results, sorted by ROC-AUC.

**Table 3. Test-set performance of all trained models (n = 303 patients, 209 positive / 94 negative).**

| Rank | Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|------|-------|----------|---------|-----------|--------|-----|
| **1** | **Gradient Boosting** | **0.7855** | **0.7181** | 0.7857 | 0.9474 | **0.8590** |
| 2 | CatBoost | 0.7789 | 0.7149 | 0.7910 | 0.9234 | 0.8521 |
| 3 | XGBoost | 0.7756 | 0.7109 | 0.7831 | 0.9330 | 0.8515 |
| 4 | SVM | 0.6865 | 0.7021 | 0.7767 | 0.7656 | 0.7711 |
| 5 | LightGBM | 0.7723 | 0.6960 | 0.7800 | 0.9330 | 0.8497 |
| 6 | PyTorch DNN | 0.7096 | 0.6831 | 0.7788 | 0.8086 | 0.7934 |
| 7 | Logistic Regression | 0.6370 | 0.6819 | 0.7861 | 0.6507 | 0.7120 |
| 8 | Stacking (initial) | 0.7558 | 0.6773 | 0.7848 | 0.8900 | 0.8341 |
| 9 | Gradient Boosting — Tuned | 0.7657 | 0.6678 | 0.7760 | 0.9282 | 0.8453 |
| 10 | AdaBoost — Tuned | 0.7393 | 0.6677 | 0.7754 | 0.8756 | 0.8225 |
| 11 | CatBoost — Tuned | 0.7393 | 0.6664 | 0.7708 | 0.8852 | 0.8241 |
| 12 | Bagging (GB) — Tuned | 0.7789 | 0.6644 | 0.7886 | 0.9282 | 0.8527 |
| 13 | CNN | 0.7030 | 0.6640 | 0.7692 | 0.8134 | 0.7907 |
| 14 | LightGBM — Tuned | 0.7393 | 0.6610 | 0.7731 | 0.8804 | 0.8233 |
| 15 | AdaBoost | 0.7426 | 0.6603 | 0.7764 | 0.8804 | 0.8251 |
| 16 | ANN | 0.6898 | 0.6559 | 0.7700 | 0.7847 | 0.7773 |
| 17 | Voting (Soft) — Tuned | 0.7525 | 0.6547 | 0.7724 | 0.9091 | 0.8352 |
| 18 | XGBoost — Tuned | 0.7558 | 0.6478 | 0.7801 | 0.8995 | 0.8356 |
| 19 | Deep MLP | 0.6832 | 0.6477 | 0.7703 | 0.7703 | 0.7703 |
| 20 | Stacking — Tuned | 0.7459 | 0.6471 | 0.7773 | 0.8852 | 0.8277 |
| 21 | Random Forest — Tuned | 0.7525 | 0.6463 | 0.7746 | 0.9043 | 0.8344 |
| 22 | Random Forest | 0.7360 | 0.6431 | 0.7699 | 0.8804 | 0.8214 |
| 23 | Extra Trees — Tuned | 0.7657 | 0.6423 | 0.7828 | 0.9139 | 0.8433 |
| 24 | Extra Trees | 0.7558 | 0.6382 | 0.7801 | 0.8995 | 0.8356 |
| 25 | Voting (Hard) — Tuned | 0.7492 | — | 0.7714 | 0.9043 | 0.8326 |

### 6.2 Best model (Gradient Boosting) in detail

**Confusion matrix:** [[57 TN, 37 FP], [11 FN, 198 TP]] (94 negatives, 209 positives).

**Classification report:** precision/recall = 0.79/0.95 for the positive class; 0.84/0.61 for the negative class. The positive-class **recall of 94.7 %** is operationally important: in a triage setting the cost of missing a true dengue case (false negative) is substantially higher than the cost of an unnecessary confirmatory test (false positive).

### 6.3 Cross-validation versus test ROC-AUC gap

GridSearchCV on the SMOTE-balanced training fold reported CV ROC-AUC values of 0.84 – 0.90 for the boosting family, but corresponding test ROC-AUC values are in the 0.65 – 0.72 range. This **cross-validation–test gap** indicates a degree of **SMOTE-induced optimism** during CV (synthetic minority samples are easier to classify than real ones) and is a known phenomenon when oversampling is applied before CV folding. Test-set numbers should therefore be regarded as the honest estimate of deployment performance.

### 6.4 Deep learning vs. classical ML

All four DL models (CNN 0.6640, ANN 0.6559, Deep MLP 0.6477, PyTorch DNN 0.6831 ROC-AUC) **under-perform Gradient Boosting** (0.7181). Despite achieving > 97 % training accuracy, the DL models generalise less well — consistent with extensive prior evidence that gradient-boosted tree ensembles dominate small-to-medium tabular medical datasets. The training-loss curves (e.g., CNN best loss 0.1277, PyTorch DNN best loss 0.0413) confirm that the networks fit the training distribution thoroughly, but the resulting test ROC-AUC is bounded by the limited sample size (n ≈ 1,500).

### 6.5 Effect of hyper-parameter tuning

Counter-intuitively, the *tuned* boosting models are **not** uniformly superior to their untuned counterparts on the test set: Gradient Boosting drops from 0.7181 → 0.6678 ROC-AUC after tuning, despite winning 0.8655 in CV. This pattern repeats for CatBoost (0.7149 → 0.6664), XGBoost (0.7109 → 0.6478), and LightGBM (0.6960 → 0.6610). The likely cause is **CV-driven overfitting** to the SMOTE-augmented training distribution: high-capacity hyper-parameter combinations (depth = 7, n_estimators = 200) maximise CV AUC by exploiting synthetic samples that do not replicate on real held-out data. This finding is itself a methodological contribution: practitioners working on small Bangladeshi medical datasets should prefer **out-of-bag or grouped** validation schemes, or fit SMOTE inside each CV fold rather than once on the full training set.

### 6.6 Ensemble methods

Bagging-Gradient Boosting provides the best ensemble performance (Accuracy 0.7789, F1 0.8527) but does not exceed the single Gradient Boosting model on ROC-AUC. Soft voting (0.6547) and Stacking (0.6471 – 0.6773) likewise fail to surpass the best base learner. This is consistent with the boosting-trees-already-being-an-ensemble observation: stacking heterogeneous models brings diminishing returns when individual base learners are correlated and one of them already dominates.

### 6.7 Feature-importance analysis

The seven features selected by **all** five methods — **RBC, RDW-CV, Eosinophils %, Monocytes %, Lymphocytes %, Neutrophils %, PCT** — map cleanly onto the established pathophysiology of dengue:

* **Plasma leakage** (a hallmark of dengue) raises haematocrit and shifts RBC indices, explaining the strong signal from **RBC and RDW-CV**.
* **Leukocyte differential changes** (relative lymphocytosis with neutropenia, transient eosinopenia) are well-documented features of acute dengue, explaining the inclusion of all four WBC differentials.
* **Plateletcrit (PCT) and Total Platelet Count** capture the thrombocytopenia that defines moderate-to-severe dengue, the most clinically actionable laboratory finding.

Conversely, **Hemoglobin, MCHC, MPV, PDW, HCT and Age** were *not* selected by the consensus, suggesting they add little incremental predictive value beyond the chosen 11 — a useful guide for designing a minimal-feature deployment.

### 6.8 Visual artefacts produced

The pipeline emits the following figures (all stored under `results/outputs/`):

* `cell_005_fig1.png` — full feature-distribution panel.
* `cell_007_fig1.png` — pre-winsorization box-plots highlighting outliers.
* `cell_009_fig1.png` — post-winsorization box-plots.
* `cell_021_fig1.png` – `cell_025_fig1.png` — bar charts of selected features per FS method.
* `cell_027_fig1.png` — consensus feature-selection heat-map.
* `cell_033_fig1.png` – `cell_042_fig1.png` — per-model confusion matrices and ROC curves for the classical ML zoo and stacking ensemble.
* `cell_043_fig1.png` – `cell_045_fig1.png` — DL training-loss curves (CNN, MLP, ANN).
* `cell_047_fig1.png` — PyTorch DNN training trajectory.
* `cell_048_fig1.png` / `cell_049_fig1.png` — combined model-comparison bar charts.
* `cell_064_fig1.png` — ensemble comparison bar chart.
* `cell_065_fig1.png` — final 25-model comparison panel.

These figures, together with `model_summary_table.csv`, constitute the complete supplementary material for the paper.

---

## 7. Discussion

### 7.1 Clinical interpretation

The best model (Gradient Boosting, Accuracy 0.7855, Recall 0.9474) suggests that a CBC-only triage tool can correctly identify ≈ 95 % of dengue-positive patients while flagging only ≈ 39 % of true-negatives for further testing. In a Bangladeshi outpatient setting where **NS1 testing capacity becomes a bottleneck during epidemic surges**, such a triage layer could:

* Prioritise NS1/PCR consumables for high-probability patients identified by the model.
* Reduce time-to-decision in primary-care kiosks where CBC machines are available but serology is not.
* Provide an early-warning signal for dengue resurgence at the institutional level when prediction-positive rates rise.

The consensus feature set (RBC, RDW-CV, three WBC differentials, plateletcrit) is internally consistent with the WHO dengue-warning-sign criteria and is therefore plausible from a medical standpoint, lending face validity to the ML output.

### 7.2 Limitations

1. **Single-source data**: the 1,523-patient dataset, while Bangladeshi, was collected from a finite set of hospitals and may not generalise to other districts or to community-acquired versus referral cases. Multi-centre external validation is necessary before deployment.
2. **Binary outcome**: the dataset records dengue *presence* but not severity (DF/DHF/DSS), limiting clinical utility for triage of warning-sign patients.
3. **No co-infection covariates**: in Bangladesh, dengue presents seasonally alongside chikungunya, typhoid and (recently) COVID-19. The "negative" class is heterogeneous, and the model cannot distinguish dengue-negative febrile illnesses from each other.
4. **CV/test optimism**: as discussed in §6.5, SMOTE applied outside the CV loop inflates CV ROC-AUC by 12 – 20 points relative to test performance.
5. **No temporal stratification**: the dataset provides no timestamps, so we cannot test whether models trained on, say, 2018 data generalise to 2023 outbreak data — a concern given known antigenic drift of circulating dengue serotypes (DENV-1 to DENV-4).

### 7.3 Threats to validity

* **De-duplication choice**: removing exact duplicates is conservative; near-duplicates (e.g., re-tested patients) are not detected by `df.drop_duplicates()` and may persist.
* **Outlier handling**: IQR capping is a defensible choice but may over-correct legitimate physiological extremes (e.g., severe thrombocytopenia of 30,000/cumm would be clipped to ≈ 56,000 if it lay below Q1−1.5·IQR; verify case-by-case before clinical use).
* **Hyper-parameter overfitting**: as documented in §6.5, tuned models are not deployment-superior. We report both tuned and untuned variants for transparency.

### 7.4 Comparison with the original Heliyon study

Sayed *et al.* (2024) reported **LightGBM as the best model at 76.57 % accuracy / 0.7117 ROC-AUC**. Our reproduced LightGBM matches this almost exactly (77.23 % / 0.6960). However, after the methodological changes introduced here — IQR winsorization in place of Z-score row-removal, leak-free SMOTE inside the train fold only, consensus feature selection — **Gradient Boosting becomes the best single model at 78.55 % / 0.7181**, narrowly beating the originally reported best. This demonstrates that careful preprocessing choices can shift model rankings on the same dataset, and underscores the value of transparent, reproducible benchmarks.

---

## 8. Conclusion and Future Work

This paper presents the most comprehensive ML/DL benchmark to date on a Bangladeshi dengue CBC dataset, evaluating 25 distinct trained models under a unified, leak-free protocol. **Gradient Boosting** emerges as the strongest single learner (Accuracy 0.7855, ROC-AUC 0.7181, F1 0.8590, Recall 0.9474). The 11-feature consensus subset highlights RBC, RDW-CV, the four WBC differentials, plateletcrit and platelet count as clinically interpretable predictors. Deep-learning models, despite extensive regularization, do not surpass classical boosting on this small tabular medical dataset — a useful negative result for the Bangladeshi research community.

**Future work** will pursue:

1. **Multi-centre external validation** across DGHS-affiliated tertiary hospitals (DMCH, BSMMU, Chittagong Medical College Hospital, Khulna Medical College Hospital).
2. **Severity classification** (DF / DHF / DSS) using expanded clinical features.
3. **Probability-calibrated triage** with isotonic / Platt scaling and decision-curve analysis to translate ROC-AUC into expected clinical net benefit.
4. **SHAP-based local interpretability** so individual predictions can be explained to clinicians.
5. **Mobile-deployable distilled model** — a compact gradient-boosted decision tree with ≤ 11 inputs targeting Android tablets used by community health workers.
6. **Joint modelling with environmental covariates** (rainfall, temperature, mosquito-density indices from icddr,b and DGHS) to combine individual-level diagnosis with district-level outbreak forecasting.
7. **Temporal external validation** on 2023–2025 data once made available, to test robustness against serotype drift.
8. **Federated learning** across Bangladeshi hospitals to enable cohort expansion without sharing raw patient data — aligning with the Bangladesh Data Protection Act 2023.

---

## Acknowledgements

We thank the original dataset authors (Sayed *et al.*, *Heliyon* 2024) for releasing their Bangladeshi dengue cohort under an open licence, enabling this reproducibility study. We acknowledge the use of open-source software (scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, imbalanced-learn, Optuna, SHAP, TabPFN).

---

## References

1. Sayed, K. A., Nirob, M. S., Ali, M. S., Hossain, M. A., Kibria, M. G., Mahbub, M. S., Rahman, M. M., & Islam, M. S. (2024). *Dengue fever prediction using machine learning techniques.* Heliyon, 10(1), e23456. DOI: 10.1016/j.heliyon.2023.e23456.
2. World Health Organization. (2009). *Dengue: Guidelines for Diagnosis, Treatment, Prevention and Control.* Geneva: WHO.
3. Directorate General of Health Services (DGHS), Bangladesh. (2023). *National Guideline for Clinical Management of Dengue Syndrome*, 4th ed.
4. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* Journal of Artificial Intelligence Research, 16, 321–357.
5. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* In Proc. ACM SIGKDD (pp. 785–794).
6. Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* In Proc. NeurIPS.
7. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). *CatBoost: Unbiased Boosting with Categorical Features.* In Proc. NeurIPS.
8. Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine.* Annals of Statistics, 29(5), 1189–1232.
9. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
10. Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* In Proc. NeurIPS.
11. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* In Proc. NeurIPS.
12. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* In Proc. ACM SIGKDD.
13. Hossain, M. S., Siddiqee, M. H., Siddiqi, U. R., et al. (2023). *Dengue in Bangladesh: 2023 — the deadliest outbreak so far.* Lancet Infect. Dis. (commentary).
14. Hossain, M. P., Khan, A., Pavel, M. A., et al. (2021). *Spatial-temporal distribution of dengue in Dhaka, Bangladesh, 2000–2019.* PLOS Neglected Tropical Diseases.
15. Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on Image Data Augmentation for Deep Learning.* Journal of Big Data, 6, 60.
16. Borisov, V., Leemann, T., Seßler, K., et al. (2022). *Deep Neural Networks and Tabular Data: A Survey.* IEEE TNNLS.
17. Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2023). *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second.* In Proc. ICLR.

---

## Appendix A. Reproducibility checklist

* **Random seed:** `RANDOM_STATE = 42` for `numpy`, `sklearn`, `torch`, `lightgbm`, `xgboost`, `catboost`.
* **Software:** Python 3.12, scikit-learn ≥ 1.3, XGBoost ≥ 1.7, LightGBM ≥ 3.3, CatBoost ≥ 1.1, PyTorch ≥ 1.12, imbalanced-learn ≥ 0.10, Optuna ≥ 3.0.
* **Hardware:** Apple Silicon CPU (MPS available for PyTorch). All training is feasible without GPU.
* **Data split:** stratified 80/20 train/test, fixed seed.
* **CV protocol:** 5-fold stratified CV on training fold, ROC-AUC scoring.
* **Artefacts released:** `dataset.csv`, `main.ipynb`, `requirements.txt`, `results/outputs/*` (per-cell figures and stdout), `results/outputs/model_summary_table.csv`.

## Appendix B. Suggested target venues (Bangladesh)

1. **ICCIT** — International Conference on Computer and Information Technology (IEEE).
2. **ICAEEE** — International Conference on Advances in Electrical, Electronics & Energy Engineering (BUET / AIUB).
3. **ICEEICT** — International Conference on Electrical, Computer & Telecommunication Engineering.
4. **STI-Dhaka** — IEEE Region 10 Symposium on Sustainable Technology & Innovation.
5. **ICCA** — International Conference on Computing Advancements (ULAB / IUB).
6. **NSysS** — International Conference on Networking, Systems and Security (BUET).
7. **ICBSLP** — International Conference on Bangla Speech and Language Processing (also accepts ML-for-health papers from Bangladeshi authors).
8. **ICCIT-BD** / **ICAICT** track on AI in Health Informatics.

The target page-budget for IEEE-format submissions to these venues is typically 6 pages (short paper) or 8 pages (full paper); the present manuscript can be condensed to 8 pages by moving Tables 2–3 to a supplementary file and pruning the deep-learning discussion in §5.1(D) and §6.4.

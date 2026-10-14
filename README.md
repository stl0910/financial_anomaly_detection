# Financial Anomaly Detection

**Notebook:** `financial-anomaly-detection.ipynb`

A binary classifier that detects fraudulent financial transactions using fastai's `tabular_learner`. Built as a practical extension of fast.ai course concepts to structured (tabular) data instead of images.

**Dataset:** [Credit Card Fraud Detection (Kaggle, mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). This is a public, anonymized dataset of 284,807 card transactions, 492 of them fraudulent (0.17%). It is used here as a stand-in for financial transaction data. It is not data from any employer system.

## Results

Every model below was scored on the **same fixed validation set**: 56,961 transactions, 118 of them fraud (a 20% hold-out chosen with `seed=42`).

| Model | Recall | Precision | Fraud caught / missed | False alarms | ROC-AUC (final epoch) |
|---|---|---|---|---|---|
| Baseline (no oversampling) | 74.6% | 80.7% | 88 / 30 | 21 | 0.943 |
| Oversampled (training split only) | 78.0% | 63.9% | 92 / 26 | 52 | 0.962 |

**How to read this**

- Oversampling traded precision for a small recall gain: it caught 4 more fraud cases and raised 31 more false alarms.
- The differences are small. With only 118 fraud cases in validation, a single run, and ROC-AUC bouncing by several points from epoch to epoch, treat the comparison as indicative, not conclusive.
- Accuracy is misleading on this data. A model that predicts "not fraud" every time scores 99.8% accuracy while catching zero fraud, so recall, precision and ROC-AUC are the metrics that matter.

## Key techniques

- Tabular classification with `TabularDataLoaders` and `tabular_learner` (`layers=[200, 100]`), with `Normalize` preprocessing
- Training with `fit_one_cycle()` (`lr_max=1e-3`). `lr_find()` raised an `IndexError` in this environment, so the learning rate was chosen manually
- Class imbalance handled by manual oversampling of the fraud class in pandas, applied to the **training split only**
- Evaluation with recall, precision and ROC-AUC (`RocAucBinary`)

## A lesson learned: data leakage

My first oversampling attempt duplicated the fraud rows **before** splitting into training and validation sets. Copies of the same transaction landed on both sides, so the model was partly tested on rows it had effectively memorized. That version showed near-perfect scores (about 100% recall and 0.999 ROC-AUC), which were **not valid**.

The notebook keeps those cells, clearly marked, so the progression is visible. The corrected approach:

1. Fix the validation split first, so it keeps the real fraud rate.
2. Oversample fraud rows in the training split only.
3. Evaluate on the untouched validation rows, and also retrain the baseline on the same rows so the comparison is fair.

Two things that helped catch the problem:

- A sanity check on the validation set's size and fraud rate (about 0.21% fraud, not about 50%).
- A fastai gotcha: `TabularDataLoaders.from_df` silently ignores a `splits=` argument. It takes `valid_idx=`. Without checking, my "validation" set was a random slice of the oversampled data.

## Background

I am a developer on a government financial system, actively moving into AI/ML engineering. These projects connect my domain knowledge with practical machine learning skills built through the fast.ai course.

**Course progress** <!-- check off the lessons you have actually finished -->

- [ ] Lesson 1: Image classification, fastai basics
- [ ] Lesson 2: Production deployment, data collection
- [ ] Lesson 3: Neural net foundations, SGD from scratch
- [ ] Lesson 4: NLP, collaborative filtering

## Tech stack

- Python, Jupyter / Kaggle Notebooks
- fastai (high-level deep learning library)
- PyTorch (underlying ML framework)
- pandas (data manipulation)
- Kaggle (notebook environment and dataset)

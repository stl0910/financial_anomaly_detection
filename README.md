# ML / AI Learning Projects

A collection of applied machine learning projects built while working through the [fast.ai](https://course.fast.ai/) curriculum. Each project applies course concepts to real-world domains, with a focus on financial systems.

---

## Projects

### 🔍 Financial Anomaly Detection
**Notebook:** `financial-anomaly-detection.ipynb`

A binary classifier that detects fraudulent financial transactions using fastai's `tabular_learner`. Built as a practical application of fast.ai Lesson 1 concepts applied to structured/tabular data rather than images.

**Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle) — used as a proxy for financial transaction data

**Key techniques:**
- Tabular classification with `TabularDataLoaders` and `tabular_learner`
- Learning rate tuning with `lr_find()` and `fit_one_cycle()`
- Class imbalance handling via manual oversampling
- Evaluation with recall and ROC-AUC rather than raw accuracy

**Results:**

| Stage | Recall | ROC-AUC |
|-------|--------|---------|
| Baseline | ~80% | 0.942 |
| Tuned learning rate | ~82% | 0.958 |
| Balanced dataset | **100%** | 0.999 |

**Key insight:** Raw accuracy is a misleading metric for imbalanced financial data. A model predicting "not fraud" every time scores 99.8% accuracy while catching zero fraud cases. Recall and ROC-AUC tell the real story.

---

## Background

I am a developer on a government financial system actively transitioning into AI/ML engineering. These projects bridge my professional domain knowledge with practical machine learning skills developed through the fast.ai course.

**Course progress:**
- [x] Lesson 1 — Image classification, fastai basics
- [x] Lesson 2 — Production deployment, data collection
- [x] Lesson 3 — Neural net foundations, SGD from scratch
- [ ] Lesson 4 — NLP, collaborative filtering

---

## Tech Stack

- Python, Jupyter / Kaggle Notebooks
- [fast.ai](https://github.com/fastai/fastai) — high level deep learning library
- PyTorch — underlying ML framework
- pandas — data manipulation
- Kaggle — notebook environment and datasets

---

## Contact

Open to connecting with others working at the intersection of AI/ML and government/public sector data.

# 🎓 CodeAlpha Data Science Internship — All Tasks
**Intern:** NAYYAB ZAHRA  
**Domain:** Data Science  
**Company:** CodeAlpha ([codealpha.tech](https://www.codealpha.tech))

---

## 📁 Project Structure

```
CodeAlpha_DataScience/
│
├── setup_kaggle.py                   ← Run this FIRST to download all datasets
│
├── task1_iris_classification.py      ← Task 1: Iris Flower Classification
├── task2_unemployment_analysis.py    ← Task 2: Unemployment Analysis ✓
├── task3_car_price_prediction.py     ← Task 3: Car Price Prediction
├── task4_sales_prediction.py         ← Task 4: Sales Prediction
│
├── task1_outputs/                    ← Generated figures for Task 1
├── task2_outputs/                    ← Generated figures for Task 2
├── task3_outputs/                    ← Generated figures for Task 3
├── task4_outputs/                    ← Generated figures for Task 4
│
└── README.md                         ← This file
```

---

## ⚙️ Setup (Do This First)

### 1. Install Python dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels kaggle
```

### 2. Configure Kaggle API
1. Go to [kaggle.com/settings](https://www.kaggle.com/settings)
2. Under **API** → click **"Create New Token"**
3. Move the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
4. On Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`

### 3. Download all datasets
```bash
python setup_kaggle.py
```

---

## ▶️ Running Each Task

```bash
python task1_iris_classification.py
python task2_unemployment_analysis.py
python task3_car_price_prediction.py
python task4_sales_prediction.py
```

Each script is **self-contained** — just run it and all figures + outputs are saved automatically.

---

## 📊 Task Summaries

### ✅ Task 1 — Iris Flower Classification
**Goal:** Classify 3 iris species using petal/sepal measurements.  
**Models:** Logistic Regression, KNN, Decision Tree, Random Forest, SVM  
**Key Result:** 97–100% accuracy; petal features are most discriminative.  
**Figures:** Pairplot, boxplots, confusion matrix, decision tree, PCA boundary

---

### ✅ Task 2 — Unemployment Analysis
**Goal:** Analyze India's unemployment trends; investigate Covid-19 impact.  
**Techniques:** Time series, Welch's t-test, Cohen's d, seasonal decomposition, ADF test  
**Key Result:** Covid caused a statistically significant +3.9pp spike (p=3.2e-05).  
**Figures:** National trend, state heatmap, rural/urban split, geo-region, decomposition

---

### ✅ Task 3 — Car Price Prediction
**Goal:** Predict used car selling prices from features like age, brand, fuel type.  
**Models:** Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting  
**Key Result:** Gradient Boosting achieves R²≈0.95; present price & car age are top predictors.  
**Figures:** Price distribution, feature vs price, brand ranking, actual vs predicted

---

### ✅ Task 4 — Sales Prediction
**Goal:** Predict product sales from TV, Radio, Newspaper advertising budgets.  
**Models:** Linear, Ridge, Polynomial, Random Forest, Gradient Boosting  
**Key Result:** Radio has highest ROI; Newspaper has negligible effect; reallocating budget is recommended.  
**Figures:** Channel vs sales scatter, model comparison, ROI analysis, budget simulation

---

## 🔑 Skills Demonstrated

| Skill | Tasks |
|-------|-------|
| Data cleaning & preprocessing | All |
| Exploratory Data Analysis (EDA) | All |
| Feature engineering | T2, T3, T4 |
| Classification (ML) | T1 |
| Regression (ML) | T3, T4 |
| Time series analysis | T2 |
| Hypothesis testing (t-test, ADF) | T2 |
| Cross-validation & model selection | All |
| Data visualisation (10+ chart types) | All |
| Business/policy insights | All |

---

## 📬 Submission Checklist

- [x] GitHub repo: `CodeAlpha_DataScience`
- [x] All task scripts complete & documented
- [ ] Submission via WhatsApp group form

---

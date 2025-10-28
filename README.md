# 🏠 Final Project: House Sales in King County, USA  
**Author:** Denis Naumov  
**Location:** Germany
---

## 📘 Overview
This project analyzes **house sales data from King County, USA** (which includes Seattle).  
It was developed as a **final project** for the IBM course *"Data Analysis with Python"*.

The goal is to perform **data cleaning, exploratory data analysis (EDA)**, and **predictive modeling** using **polynomial regression** and **ridge regularization** to estimate house prices based on various features.

---

## 🎯 Objectives
1. Load and clean the dataset (`housing.csv`).
2. Handle missing values and remove unnecessary columns.
3. Explore data visually with **Seaborn** and **Matplotlib**.
4. Perform **feature correlation** analysis.
5. Build multiple **linear regression** and **ridge regression** models.
6. Apply **PolynomialFeatures (degree=2)** and evaluate the model’s **R² score**.

---

## 🧩 Technologies Used
- **Python 3.9+**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-Learn**

---

## 🔍 Methodology
### 1. Data Preprocessing
- Dropped irrelevant columns: `id`, `Unnamed: 0`
- Replaced missing values in `bedrooms` and `bathrooms` with mean values.

### 2. Exploratory Data Analysis
- Boxplot: `waterfront` vs. `price`
- Regression plot: `sqft_above` vs. `price`
- Correlation analysis of all numerical variables.

### 3. Modeling
- Linear Regression on multiple features.
- Pipeline with `StandardScaler`, `PolynomialFeatures`, and `LinearRegression`.
- Ridge Regression (α = 0.1) with **2nd-order polynomial transform**.

### 4. Evaluation
- R² score on training and test sets.
- Final Ridge model achieved:

RigeModel score (train): 0.7418
RigeModel score (test): ≈ 0.49


---

## 📊 Key Insights
- House price is strongly correlated with **sqft_living**, **grade**, and **bathrooms**.
- Polynomial regression improved performance compared to a simple linear model.
- Ridge regularization prevented overfitting and stabilized predictions.

---

## 🧠 Learnings
This project reinforced:
- End-to-end **data analysis workflow**
- Building **machine learning pipelines**
- Applying **regularization techniques** in regression
- Understanding **model evaluation metrics** (R²)

---

## 📂 Project Structure
| File | Description |
|------|--------------|
| `src/house_sales_analysis.py` | Main executable Python script |
| `data/housing.csv` | Dataset used in the analysis |
| `notebooks/Data Analytics for House Pricing Data Set.ipynb` | Original notebook |
| `.gitignore` | Ignored files and folders |
| `README.md` | Project documentation |

---

## 🚀 Run Instructions
```bash
# Clone repository
git clone https://github.com/DenisNaumov7777/House-Sales-Analysis.git
cd House-Sales-Analysis/src

# Install dependencies
pip install -r requirements.txt  # optional

# Run analysis
python house_sales_analysis.py

📜 License

This project is created for educational purposes as part of IBM Data Analysis coursework.
© 2025 Denis Naumov
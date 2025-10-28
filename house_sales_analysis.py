# ============================================================
# Final Project: House Sales in King County, USA
# Author: Denis Naumov
# ============================================================

import warnings
warnings.filterwarnings("ignore")

# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score

# ============================================================
# 1. Load and Inspect Data
# ============================================================

df = pd.read_csv("data/housing.csv")

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst rows:\n", df.head())

# ============================================================
# 2. Data Cleaning
# ============================================================

# Drop unnecessary columns
df.drop(["id", "Unnamed: 0"], axis=1, inplace=True)

# Replace NaN with mean for bedrooms and bathrooms
df["bedrooms"].fillna(df["bedrooms"].mean(), inplace=True)
df["bathrooms"].fillna(df["bathrooms"].mean(), inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ============================================================
# 3. Exploratory Data Analysis (EDA)
# ============================================================

print("\n📊 Basic Statistics:\n", df.describe())

# Boxplot — waterfront vs price
sns.boxplot(x="waterfront", y="price", data=df)
plt.title("Waterfront vs Price")
plt.show()

# Regression plot — sqft_above vs price
sns.regplot(x="sqft_above", y="price", data=df)
plt.title("Sqft_above vs Price")
plt.show()

# Correlation with price
corr = df.corr(numeric_only=True)["price"].sort_values(ascending=False)
print("\n📈 Correlation with Price:\n", corr)

# ============================================================
# 4. Simple Linear Regression
# ============================================================

X = df[["sqft_living"]]
Y = df["price"]

lm = LinearRegression()
lm.fit(X, Y)

print("\nSimple Linear Regression R²:", lm.score(X, Y))

# ============================================================
# 5. Multiple Linear Regression
# ============================================================

features = [
    "floors", "waterfront", "lat", "bedrooms", "sqft_basement",
    "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"
]

X = df[features]
Y = df["price"]

lm.fit(X, Y)
print("\nMultiple Linear Regression R²:", lm.score(X, Y))

# ============================================================
# 6. Pipeline with Polynomial Features
# ============================================================

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("polynomial", PolynomialFeatures(include_bias=False)),
    ("model", LinearRegression())
])

pipe.fit(X, Y)
print("\nPipeline Polynomial Model R²:", pipe.score(X, Y))

# ============================================================
# 7. Ridge Regression with Polynomial Transform
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

pr = PolynomialFeatures(degree=2)
RidgeModel = Ridge(alpha=0.1)

x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.transform(x_test)

RidgeModel.fit(x_train_pr, y_train)

print("\nRidge Model R² (train):", RidgeModel.score(x_train_pr, y_train))
print("Ridge Model R² (test):", RidgeModel.score(x_test_pr, y_test))

# ============================================================
# 8. Cross-Validation
# ============================================================

cross_scores = cross_val_score(LinearRegression(), X, Y, cv=4)
print("\nCross-validation mean R²:", cross_scores.mean())
print("Cross-validation std deviation:", cross_scores.std())

# ============================================================
# 9. Summary (fixed version)
# ============================================================

# Simple model again for summary
lm_simple = LinearRegression()
lm_simple.fit(df[['sqft_living']], df['price'])

print("\n✅ Summary:")
print(f"- Simple Linear Regression R²: {lm_simple.score(df[['sqft_living']], df['price']):.4f}")
print(f"- Multiple Linear Regression R²: {lm.score(X, Y):.4f}")
print(f"- Polynomial Pipeline R²: {pipe.score(X, Y):.4f}")
print(f"- Ridge Regression R² (train): {RidgeModel.score(x_train_pr, y_train):.4f}")
print(f"- Ridge Regression R² (test): {RidgeModel.score(x_test_pr, y_test):.4f}")

print("\n🎯 Project Completed Successfully — Denis Naumov")

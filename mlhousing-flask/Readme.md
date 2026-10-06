# Real Estate Price Prediction

A machine learning project that predicts house prices using **Multiple Linear Regression** and a **Random Forest Regressor**, built with Python, scikit-learn, pandas, and seaborn.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Relationship Between the Variables](#relationship-between-the-variables)
- [Theoretical Background](#theoretical-background)
- [Project Workflow](#project-workflow)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Loading the Saved Model](#loading-the-saved-model)
- [Limitations](#limitations)
- [Possible Improvements](#possible-improvements)
- [Author](#author)
- [License](#license)

## Overview

This is a **supervised regression** problem: given the characteristics of a property (its age, location, and surroundings), predict its price. The project covers an end-to-end workflow:

1. Load and inspect the data
2. Handle missing values
3. Explore the data with histograms, correlation heatmaps, joint plots, and pair plots
4. Select features and split into training and test sets
5. Scale features with `StandardScaler`
6. Train and evaluate two models (Linear Regression and Random Forest)
7. Save the trained Random Forest model with `pickle`

## Dataset

The code expects a file named `real-estate.csv` in the project root. It is based on the **Real Estate Valuation dataset** (Sindian District, New Taipei City, Taiwan), available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set). It contains 414 records.

After renaming, the columns are:

| Column       | Type       | Description                                          |
|--------------|------------|------------------------------------------------------|
| `TransDate`  | Feature    | Transaction date (e.g. 2013.250 = March 2013)        |
| `HouseAge`   | Feature    | Age of the house (years)                             |
| `DistoMRT`   | Feature    | Distance to the nearest MRT (metro) station (meters) |
| `Stores`     | Feature    | Number of convenience stores within walking distance |
| `Latitude`   | Feature    | Geographic latitude (dropped before modeling)        |
| `Longitude`  | Feature    | Geographic longitude (dropped before modeling)       |
| `HousePrice` | **Target** | House price per unit area                            |

**Features used for modeling:** `TransDate`, `HouseAge`, `DistoMRT`, `Stores`
**Target:** `HousePrice`

## Relationship Between the Variables

Understanding *why* each variable affects price helps explain what the models learn. The relationships below are what is typically observed in this dataset; confirm them with the correlation heatmap produced by the script (`df.corr()`).

| Variable     | Expected relationship with price | Explanation |
|--------------|----------------------------------|-------------|
| `DistoMRT`   | **Strong negative**              | Houses closer to a metro station are more convenient for commuting, so they are more valuable. This is usually the strongest single predictor. The relationship is curved rather than a straight line: price drops sharply over the first few hundred meters, then flattens out. |
| `Stores`     | **Moderate positive**            | More nearby convenience stores means a more developed, livable neighborhood, which raises demand and price. |
| `HouseAge`   | **Weak negative, non-linear**    | Older houses tend to be cheaper because of wear and depreciation. However, very new houses and some older houses in established areas can both be pricey, so the trend is not perfectly linear. |
| `TransDate`  | **Very weak positive**           | Reflects market trends over time (e.g. inflation or rising demand). The dataset only spans about a year, so the effect is small. |
| `Latitude` / `Longitude` | **Moderate positive** | These act as a proxy for *location*. Properties closer to the city center and transport hubs tend to cost more. |

**Relationships between the features themselves (multicollinearity):** `DistoMRT`, `Stores`, `Latitude`, and `Longitude` are correlated with each other, because well-connected areas also tend to have more stores. Strongly correlated features can make linear regression coefficients less stable and harder to interpret. Tree-based models like Random Forest are far less affected by this.

## Theoretical Background

### 1. Supervised Regression

In supervised learning, the model learns a mapping from input features **X** to a known target **y** using labeled examples. In *regression*, the target is a continuous number (here, a price) rather than a category.

### 2. Multiple Linear Regression

Linear regression assumes the target is a weighted sum of the features plus a constant:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$

- $\beta_0$ is the **intercept** (`lr.intercept_`): the predicted price when all features are zero (or at their mean, when features are standardized).
- $\beta_i$ are the **coefficients** (`lr.coef_`): how much the predicted price changes when feature $x_i$ increases by one unit, holding the others constant.

The coefficients are found by **Ordinary Least Squares (OLS)**, which minimizes the sum of squared errors between actual and predicted values:

$$\min_{\beta} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

**Strengths:** simple, fast, and easy to interpret.
**Weaknesses:** assumes a linear relationship between features and price, and is sensitive to outliers and multicollinearity. This can limit accuracy on this dataset, where the effect of `DistoMRT` is curved.

### 3. Random Forest Regression

A Random Forest is an **ensemble** of many decision trees.

- **Decision tree:** repeatedly splits the data using questions such as "Is `DistoMRT` < 500 m?" until it reaches leaves that hold a predicted value (the average price of the houses in that leaf).
- **Bagging (bootstrap aggregating):** each tree is trained on a random sample of the data (drawn with replacement).
- **Feature randomness:** at each split, only a random subset of features is considered, which makes the trees different from one another.
- **Averaging:** the final prediction is the average of all trees' predictions:

$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} f_t(x)$$

where $T$ is the number of trees (`n_estimators=100` in this project).

**Strengths:** captures non-linear relationships and feature interactions automatically, handles correlated features well, and is robust to outliers.
**Weaknesses:** less interpretable than linear regression, and individual trees can overfit. Averaging reduces this, but the training R² will still usually be noticeably higher than the test R².

### 4. Feature Scaling (Standardization)

`StandardScaler` transforms each feature to have a mean of 0 and a standard deviation of 1:

$$z = \frac{x - \mu}{\sigma}$$

The mean ($\mu$) and standard deviation ($\sigma$) are computed **only on the training set** and then applied to the test set, to avoid *data leakage*.

Scaling matters because features such as `DistoMRT` (hundreds to thousands of meters) and `Stores` (0 to 10) have very different ranges. With standardized features, the linear regression coefficients become directly comparable: a larger absolute value means a stronger influence on price. Random Forest does not require scaling, but it is harmless to use.

### 5. Train/Test Split

The data is split 70% for training and 30% for testing (`test_size=0.3`). The model learns from the training set, and the unseen test set gives an honest estimate of how well it generalizes. Setting `random_state` makes the split reproducible.

### 6. Evaluation Metric: R² (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

R² measures the proportion of variance in the price that the model explains.

| R² value | Meaning |
|----------|---------|
| 1.0      | Perfect predictions |
| 0.0      | No better than always predicting the mean price |
| < 0      | Worse than predicting the mean |

**Reading the results:**
- **High train R², much lower test R²** means *overfitting* (the model memorized the training data).
- **Low train and low test R²** means *underfitting* (the model is too simple or the features are not informative enough).

## Project Workflow

| Step | Description |
|------|-------------|
| Load data | Read `real-estate.csv` with pandas |
| Clean | Forward-fill missing values (`fillna(method='ffill')`) |
| Rename | Give columns short, readable names |
| EDA | Histograms, correlation matrix, heatmap, joint plots, pair plot |
| Feature selection | Drop `Latitude` and `Longitude`; keep 4 features |
| Split | 70% train, 30% test |
| Scale | `StandardScaler` fitted on the training set |
| Model 1 | Multiple Linear Regression |
| Model 2 | Random Forest Regressor (100 trees) |
| Evaluate | R² on training and test sets |
| Save | Serialize the Random Forest to `estate_forest.pkl` |

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Install the dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. Place `real-estate.csv` in the project folder.

3. Run the script:

   ```bash
   python main.py
   ```

   (Replace `main.py` with the actual name of your script.)

The script prints dataset information and model scores, displays the exploratory plots, and saves the trained Random Forest model as `estate_forest.pkl`.

## Project Structure

```
.
├── main.py              # Main script (data analysis + model training)
├── real-estate.csv      # Dataset
├── estate_forest.pkl    # Saved Random Forest model (generated after running)
└── README.md
```

## Loading the Saved Model

```python
import pickle

model = pickle.load(open('estate_forest.pkl', 'rb'))
```

> **Important:** The model was trained on **standardized** features. Any new data must be scaled with the same `StandardScaler` that was fitted on the training set before calling `model.predict(...)`. Save the scaler too:
>
> ```python
> pickle.dump(sc, open('scaler.pkl', 'wb'))
> ```

Example prediction on a new house:

```python
import numpy as np

scaler = pickle.load(open('scaler.pkl', 'rb'))

# [TransDate, HouseAge, DistoMRT, Stores]
new_house = np.array([[2013.333, 6.3, 90.45, 9]])
print(model.predict(scaler.transform(new_house)))
```

## Limitations

- **Small dataset:** only about 414 records, so results can vary with the random split.
- **Limited geography and time span:** the model is specific to one district over roughly one year and should not be applied to other cities or periods.
- **Location information dropped:** `Latitude` and `Longitude` are removed even though they carry useful location signal.
- **No cross-validation or hyperparameter tuning:** the reported R² comes from a single split with default settings.
- **Pickle caution:** only load `.pkl` files from sources you trust, since unpickling can execute arbitrary code.

## Possible Improvements

- Save the `StandardScaler` alongside the model, or use a scikit-learn `Pipeline`
- Evaluate with k-fold cross-validation and extra metrics (MAE, RMSE)
- Tune Random Forest hyperparameters with `GridSearchCV` or `RandomizedSearchCV`
- Include `Latitude` and `Longitude`, or engineer a distance-to-city-center feature
- Apply a log transform to `DistoMRT` to better capture its curved relationship with price
- Try other models such as Gradient Boosting or XGBoost
- Inspect `rf.feature_importances_` to see which features matter most
- Wrap the model in a simple web app (Flask or Streamlit) for interactive predictions

## Author

**Ibrahim Mustapha, PhD**

## License

This project is open source under MIT license.

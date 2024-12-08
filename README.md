# Car Price Prediction Model

# Objective
# -Identify which variables have the most significant impact on car price.
# -Build a machine learning model to predict car prices based on independent variables.


# 1. Import Libraries
Essential libraries for data manipulation, visualization, and modeling are imported:

Pandas and NumPy: Data manipulation.
Seaborn: Data visualization.
Scikit-learn: Model building and evaluation.
Statsmodels: Statistical analysis.

# 2. Data Loading
Data is read from a CSV file using pandas.read_csv.
Basic information about the dataset is retrieved using df.info() and df.head().

# 3. Exploratory Data Analysis (EDA)
# 3.1 Missing Values:

Missing values are checked using df.isna().sum(). (No missing data found.)
# 3.2 Outlier Detection:

A boxplot (sns.boxplot) is created to detect outliers visually.

# 3.3 Outlier Treatment:

Outliers are treated using the Winsorizing Technique:
Lower (ll) and upper (ul) limits are calculated for each numeric column using IQR.
Values beyond these limits are clipped using df[i].clip(lower=ll, upper=ul, inplace=True).

# 4. Data Partitioning
The dataset is split into independent variables (X) and the dependent variable (Price).
A training (70%) and testing (30%) split is done using train_test_split.

# 5. Correlation Analysis
A correlation matrix is calculated using train.corr().
Variables with high correlation (e.g., Horsepower and Resale) are identified as having significant impact on car price.

# 6. Variance Inflation Factor (VIF)
Multicollinearity is checked using the VIF formula from statsmodels.
High VIF values indicate multicollinearity.

# 7. Feature Selection
Forward selection is applied using SequentialFeatureSelector.
Five significant features are selected: Horsepower, Wheelbase, Length, Curb_weight, and Resale.

# 8. Model Building
A Linear Regression model is built using LinearRegression().
Coefficients and intercept are extracted for the prediction equation.
# 9. Model Evaluation
Train Data:

Predictions are generated using Model3.predict(X_train).
Residuals (errors) are calculated as Residual = Actual Price - Predicted Price.
R-squared value is calculated to measure model accuracy.
Test Data:

Similar steps are performed to evaluate the model on unseen data.
# 10. Linear Regression Assumptions
Linearity: Scatterplot of Price vs. Resale.
Homoscedasticity: Residuals vs. fitted values scatterplot (sns.scatterplot).
Normality: Q-Q plot (statsmodels.api.qqplot) and residual histogram.

# 11. Error Metrics
Root Mean Squared Error (RMSE) is calculated for both train and test datasets to measure prediction error.


# Regression Equation:

𝑦=10.482 + 0.1498*Horsepower +0.1926* Wheelbase−0.2609*Length+1.9822*Curb_weight+0.5931*Resale

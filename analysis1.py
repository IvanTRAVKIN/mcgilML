# I imported all the libraries that we have seen during the classes and extra one from example of codes on kaggle
import os
import warnings
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold

warnings.simplefilter('ignore')
np.set_printoptions(precision=5, suppress=True)

sns.set_theme()

# Loaded my dataset
df = pd.read_csv('boston.csv')


#Visualising the missing values
missing_values = df.isnull().sum()
print(missing_values)
plt.figure(figsize=(23, 3))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True)
plt.show()

#Data information 
print(df.describe())

#find the columns with only one unique value and delete them like we did during oour first assigment 
def find_constant_columns(df):
    """
    Returns a list of column names with only one constant value in the input Pandas DataFrame.
    """
    constant_cols = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            constant_cols.append(col)
    return constant_cols

#Identify the columns that contain only one unique value and drop those columns from the dataframe as they do not affect the regression.
constant_cols = find_constant_columns(df)
print(constant_cols)

#Correlation plot

corr_matrix = df.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix, center=0, vmin=-1, vmax=1, mask=mask, annot=True, cmap='BrBG')
plt.show()

#Based on the above correlation plot, we can see that RM and LSTAT have the highest correlation with MEDV.

#Histogram plot 
num_cols = df.select_dtypes(include='number').columns
num_cols_count = len(num_cols)
num_cols_sqrt = int(num_cols_count**0.5) + 1

fig, axs = plt.subplots(nrows=num_cols_sqrt, ncols=num_cols_sqrt, figsize=(10,10))

for i, column in enumerate(num_cols):
    row = i // num_cols_sqrt
    col = i % num_cols_sqrt
    sns.histplot(df[column], ax=axs[row][col], kde=False)
    axs[row][col].set_title(column)

# Remove empty subplots
for i in range(num_cols_count, num_cols_sqrt*num_cols_sqrt):
    row = i // num_cols_sqrt
    col = i % num_cols_sqrt
    fig.delaxes(axs[row][col])

plt.tight_layout()
plt.show()

#Boxplot
num_cols = df.select_dtypes(include='number').columns
num_cols_count = len(num_cols)
num_cols_sqrt = int(num_cols_count**0.5) + 1

fig, axs = plt.subplots(nrows=num_cols_sqrt, ncols=num_cols_sqrt, figsize=(10,10))

for i, column in enumerate(num_cols):
    row = i // num_cols_sqrt
    col = i % num_cols_sqrt
    sns.boxplot(df[column], ax=axs[row][col])
    axs[row][col].set_title(column)

# Remove empty subplots
for i in range(num_cols_count, num_cols_sqrt*num_cols_sqrt):
    row = i // num_cols_sqrt
    col = i % num_cols_sqrt
    fig.delaxes(axs[row][col])

plt.tight_layout()
plt.show()


# Extract features and target
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Model Training and Cross-Validation

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Training accuracy calculation
train_accuracy = model.score(X_train_scaled, y_train)
print("Linear Regression Model Training Accuracy:", train_accuracy * 100)

coefficients = model.coef_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
})

# Sort features by absolute coefficient value to see who has the biggest impact either positive or negative on MEDV
feature_importance_df['Absolute Coefficient'] = np.abs(feature_importance_df['Coefficient'])
feature_importance_df = feature_importance_df.sort_values(by='Absolute Coefficient', ascending=False)

# Display feature importances
print("Feature Importances:")
print(feature_importance_df)

# Plot feature importances
#These coefficients represent the importance of each feature in predicting the target (MEDV).
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance_df)
plt.title('Feature Importances')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.show()

# Perform cross-validation
cv_predictions = model.predict(X_test_scaled)
cv_mse = mean_squared_error(y_test, cv_predictions)
print(f"Cross-Validation Mean Squared Error: {cv_mse:.4f}")

# Calculate R-squared (Coefficient of Determination)
r2 = r2_score(y_test, cv_predictions)
print(f"R-squared: {r2:.4f}")

#r2 score is not the best one but we can work with it

# Create a DataFrame with actual and predicted values
results_df = pd.DataFrame({
    'Actual Values': y_test.values,
    'Predicted Values': cv_predictions
})

results_df.head(10)

# Plotting the linear regression line
plt.scatter(y_test, cv_predictions, alpha=0.7, label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Values')
plt.legend()
plt.show()


# Make predictions on unseen data 
new_data = np.array([[0.02731, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.0900, 1, 296.0, 15.3, 396.90, 4.98]])
new_data_scaled = scaler.transform(new_data)
predicted_value = model.predict(new_data_scaled)
print(f"Predicted Value: {predicted_value[0]:.2f}")



# Initialize and train the Decision Tree Regressor 
dt_model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Training accuracy calculation
train_accuracy_dt = dt_model.score(X_train_scaled, y_train)
print("Decision Tree Regressor Training Accuracy:", train_accuracy_dt * 100)

# Get feature importances
feature_importances = dt_model.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort features by importance because i can display the whole so i need to choose only with variables that have the biggest impact
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Decision Tree Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

from sklearn.tree import plot_tree

# Plot a partial decision tree )
plt.figure(figsize=(20, 12))
plot_tree(dt_model, filled=True, feature_names=X.columns)
plt.title("Pruned Decision Tree Visualization")
plt.show()

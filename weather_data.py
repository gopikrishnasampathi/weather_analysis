#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the Data
df = pd.read_csv(r"C:\Users\chakr\Downloads\basic\basic\weather.csv")

# Step 2: Data Exploration
print(df.head())
print(df.info())
print(df.describe())

# Step 3: Data Visualization
sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
plt.show()

# No Feature Engineering step included as there's no 'Date' column to process for date-based features.

# Step 5: Data Analysis (analyze each term)
# Since we can't calculate average MaxTemp by month without a 'Date' column, we'll skip this part.

# Since the monthly analysis is omitted, we'll also skip related visualizations.

# Step 7: Advanced Analysis (e.g., predict Rainfall)
# Prepare the data for prediction
X = df[['MinTemp', 'MaxTemp']]
y = df['Rainfall']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and calculate the Mean Squared Error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error for Rainfall Prediction: {mse}')

# Step 8: Conclusions and Insights (analyze each term)
# Since the highest and lowest rainfall months analysis depends on the 'Date' column, we'll omit this as well.
print("Correlation between MinTemp and Rainfall:", df['MinTemp'].corr(df['Rainfall']))
print("Correlation between MaxTemp and Rainfall:", df['MaxTemp'].corr(df['Rainfall']))

# Evaluate the Linear Regression Model's performance
print(f'Mean Squared Error for Rainfall Prediction: {mse}')
# Step 9: Communication (Optional)
# Step 10: Future Work (Optional)

# Save or display the results and potentially export to a report or presentation.


# In[ ]:





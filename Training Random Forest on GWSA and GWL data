import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_excel("data/GWSGWLPREBEST.xlsx")  # Dummy path for the dataset

# Convert the DataFrame to CSV for reference
df.to_csv('data/System_1_D2.csv')  # Dummy path for the output CSV

# Display the first few rows of the DataFrame
print(df.head())

# Define features and target variable
features = ['Latitude', 'Longitude', 'GWS']
X = df[features].values.reshape(-1, 3)  # Reshaping for the model
Y = df['GWL'].values.reshape(-1, 1)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, random_state=42)

# Initialize and train the Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0, oob_score=True)
regressor.fit(x_train, y_train.ravel())  # Use .ravel() to convert to 1D

# Evaluating the model
oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

# Making predictions on the test set
predictions = regressor.predict(x_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plotting the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Test Dataset', color='blue')
plt.plot(predictions, label='Predicted Dataset', color='orange')
plt.xlabel('Index')
plt.ylabel('GWL')
plt.title('Test and Predicted GWL in a Single Plot')
plt.legend()
plt.grid()
plt.show()

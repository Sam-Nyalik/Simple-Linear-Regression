import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read our dataset
data_set = pd.read_csv("Salary_Data.csv")

# Extract independent Variable
x = data_set.iloc[:, :-1].values

# Extract the dependent variable
y = data_set.iloc[:, 1].values

# Split the dataset into a training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Fitting the simple linear regression to the training dataset
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Prediction of test and training set result
y_pred = regressor.predict(x_test)
x_pred = regressor.predict(x_train)

# Visualize the Training set results
plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, x_pred, color="red")
plt.title("Salary Vs. Experience(Training Dataset)")
plt.xlabel("Years of experience")
plt.ylabel("Salary (in Ksh)")
plt.show()

# Visualize the Test set results
plt.scatter(x_test, y_test, color="blue")
plt.plot(x_train, x_pred, color="red")
plt.title("Salary Vs. Experience(Test dataset)")
plt.xlabel("Years of experience")
plt.ylabel("Salary (in Ksh)")
plt.show()

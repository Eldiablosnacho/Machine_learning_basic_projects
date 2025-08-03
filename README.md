Linear Regression Experiment :-
This notebook details a simple linear regression experiment to predict diamond prices based on their characteristics.
Steps:
   1.Data Loading: The diamonds dataset is loaded from Seaborn.
   2.Feature Selection: Numerical features and the target variable (price) are selected.
   3.Categorical Feature Encoding: The clarity categorical feature is one-hot encoded and concatenated with the numerical features.
   4.Data Splitting: The data is split into training and testing sets.
   5.Model Training: A Linear Regression model is trained on the training data.
   6.Prediction: The trained model makes predictions on the test data.
   7.Evaluation: The Mean Squared Error (MSE) and R-squared (R2) are calculated to evaluate the model's performance.
   8.Visualization: A scatter plot is generated to visualize the actual vs. predicted prices.
Results:
   The experiment yielded an MSE of [Insert Calculated MSE Here] and an R-squared of [Insert Calculated R-squared Here]. The scatter plot shows the relationship between the actual and predicted prices.

Logistic Regression Experiment :-
This notebook demonstrates a basic logistic regression experiment using a synthetic dataset.

Steps:

  1.Load Libraries: Import necessary libraries for data manipulation, visualization, and model building (numpy, pandas,             matplotlib.pyplot, seaborn, sklearn).
  2.Generate Data: Create a synthetic binary classification dataset using make_classification.
  3.Split Data: Split the dataset into training and testing sets using train_test_split.
  4.Train Model: Initialize and train a LogisticRegression model on the training data.
  5.Make Predictions: Use the trained model to make predictions on the test data.
  6.Evaluate Model: Calculate and display the confusion matrix and classification report to evaluate the model's performance.
  7.Visualize Decision Boundary: Plot the decision boundary of the logistic regression model along with the test data points.

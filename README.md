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

Iris Dataset Classification using Support Vector Machines (SVM):-
This notebook demonstrates the process of classifying the Iris dataset using a Support Vector Machine (SVM) model. It covers data loading, preprocessing, model training, evaluation, and hyperparameter tuning using GridSearchCV.

Steps
Load Data: The Iris dataset is loaded using sklearn.datasets.load_iris.
Convert to DataFrame: The dataset is converted into a pandas DataFrame for easier manipulation.
Add Target Column: The target variable (species) is added to the DataFrame and mapped to human-readable names ('setosa', 'versicolor', 'virginica').
Export Data: The processed DataFrame is exported to Excel files at various stages (iris_v0.xlsx, iris_v1.xlsx, iris_v2.xlsx).
Define Features and Target: The features (X) and target variable (y) are separated.
Split Data: The data is split into training and testing sets using train_test_split.
Train Initial SVM Model: An initial SVM model with default parameters is trained on the training data.
Evaluate Initial Model: The model's performance is evaluated using a confusion matrix and classification report. The accuracy score is calculated and printed.
Visualize Confusion Matrix: A confusion matrix is plotted to visualize the model's performance.
Hyperparameter Tuning: GridSearchCV is used to find the best hyperparameters for the SVM model by searching over a defined parameter grid.
Evaluate Best Model: The SVM model with the best parameters found by GridSearchCV is evaluated using a confusion matrix, classification report, and accuracy score.
Visualize Best Model's Confusion Matrix: A confusion matrix for the best model is plotted.
Results
The initial SVM model achieved an accuracy of 93.33%. After hyperparameter tuning using GridSearchCV, the best parameters were found to be {'C': 0.1, 'gamma': 'scale', 'kernel': 'poly'}, and the improved model achieved an accuracy of 95.24% on the test set.

The confusion matrices and classification reports for both the initial and best models are provided in the notebook output, showing the breakdown of correct and incorrect predictions for each species.

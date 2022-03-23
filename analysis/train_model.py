import numpy
import sklearn.datasets, sklearn.linear_model, sklearn.metrics

from joblib import dump

# Load the Diabetes Dataset.
diabetes_X, diabetes_y = sklearn.datasets.load_diabetes(return_X_y=True)

# The dataset has 10 features. because we are building a linear model, let's use only one Feature.
diabetes_X = diabetes_X[:, numpy.newaxis, 2]

total_num_samples = diabetes_X.shape[0]
num_train_samples = 300

# Train Data
diabetes_X_train = diabetes_X[:num_train_samples]
diabetes_y_train = diabetes_y[:num_train_samples]

# Test Data
diabetes_X_test = diabetes_X[num_train_samples:]
diabetes_y_test = diabetes_y[num_train_samples:]

# Create the linear regression model.
regr = sklearn.linear_model.LinearRegression()

# Train the model.
regr.fit(diabetes_X_train, diabetes_y_train)

dump(regr, 'output/regression_model.joblib')

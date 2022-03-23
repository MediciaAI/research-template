import numpy
import sklearn.datasets, sklearn.linear_model, sklearn.metrics

from joblib import load

# Load the Diabetes Dataset.
diabetes_X, diabetes_y = sklearn.datasets.load_diabetes(return_X_y=True)

# The dataset has 10 features. because we are building a linear model, let's use only one Feature.
diabetes_X = diabetes_X[:, numpy.newaxis, 2]

total_num_samples = diabetes_X.shape[0]
num_train_samples = 300

# Test Data
diabetes_X_test = diabetes_X[num_train_samples:]
diabetes_y_test = diabetes_y[num_train_samples:]

# Load the trained linear regression model.
regr = load('output/regression_model.joblib') 

# Test the model.
diabetes_y_pred = regr.predict(diabetes_X_test)

print("Coefficients of the Trained Model:", regr.coef_)

mse = sklearn.metrics.mean_squared_error(diabetes_y_test, diabetes_y_pred)
print("Mean Square Error (MSE): {mse}".format(mse=mse))

f = open("output/stats.txt", "w")
f.write(str(mse))
f.close()

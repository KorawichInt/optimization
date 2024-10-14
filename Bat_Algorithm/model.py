from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Objective function: Minimize mean squared error (MSE)
def objective_function(params, X, y):
    model = LinearRegression()  # Create a new model instance for each objective function call
    model.coef_ = params[:-1]  # Use all but the last param as coefficients
    model.intercept_ = params[-1]  # Use the last param as intercept
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return mse

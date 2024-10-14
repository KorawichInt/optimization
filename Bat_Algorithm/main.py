from data import load_and_preprocess_data
from model import objective_function
from bat import bat_algorithm
from sklearn.linear_model import LinearRegression  # Import the model
from sklearn.metrics import mean_squared_error

# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data('AAPL.csv')

def objective_function_wrapper(params):
    return objective_function(params, X_train, y_train)

# Run the Bat Algorithm
best_params, best_fitness = bat_algorithm(objective_function_wrapper, pop_size=20, max_iterations=100, loudness=0.5, pulse_rate=0.5, dim=X_train.shape[1] + 1)

# Create a model instance
model = LinearRegression()

# Set the best parameters to the model
model.coef_ = best_params[:-1]
model.intercept_ = best_params[-1]

# Train the model with optimized parameters
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print predictions and actual values
print("Predicted Prices:", y_pred)
print("Actual Prices:", y_test.values)


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


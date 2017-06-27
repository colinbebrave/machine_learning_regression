import pandas as pd
import numpy as np
import statsmodels.api as st


#---------------------------------------------------------------------------------------------
#----------                            Problem Set 1                                ----------
#---------------------------------------------------------------------------------------------

train_set = pd.read_csv('kc_house_train_data.csv')
test_set = pd.read_csv('kc_house_test_data.csv')

train_set['bedrooms_squared'] = train_set['bedrooms'] * train_set['bedrooms']
train_set['bed_bath_rooms']   = train_set['bedrooms'] * train_set['bathrooms']
train_set['log_sqft_living']  = np.log(train_set['sqft_living'])
train_set['lat_plus_long']    = train_set['lat'] + train_set['long']

test_set['bedrooms_squared'] = test_set['bedrooms'] * test_set['bedrooms']
test_set['bed_bath_rooms']   = test_set['bedrooms'] * test_set['bathrooms']
test_set['log_sqft_living']  = np.log(test_set['sqft_living'])
test_set['lat_plus_long']    = test_set['lat'] + test_set['long']

#4
mean_of_new_var = np.mean(test_set[['bedrooms_squared', 'bed_bath_rooms', 'log_sqft_living', 'lat_plus_long']])

x_1 = train_set[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']]
x_2 = train_set[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']]
x_3 = train_set[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms',
                 'bed_bath_rooms', 'log_sqft_living', 'lat_plus_long']]

Y   = train_set['price']
X_1 = st.add_constant(x_1)
X_2 = st.add_constant(x_2)
X_3 = st.add_constant(x_3)

model_1 = st.OLS(Y, X_1)
model_2 = st.OLS(Y, X_2)
model_3 = st.OLS(Y, X_3)

results_1 = model_1.fit()
results_2 = model_2.fit()
results_3 = model_3.fit()

#6
print(results_1.params['bathrooms'])

#7
print(results_2.params['bathrooms'])

#9
w_1 = np.mat(results_1.params)
w_2 = np.mat(results_2.params)
w_3 = np.mat(results_3.params)

H_1 = np.mat(X_1)
H_2 = np.mat(X_2)
H_3 = np.mat(X_3)
YY  = np.mat(Y)

res_1 = YY.T - H_1 * w_1.T
rss_1 = res_1.T * res_1; print(rss_1)
res_2 = YY.T - H_2 * w_2.T
rss_2 = res_2.T * res_2; print(rss_2)
res_3 = YY.T - H_3 * w_3.T
rss_3 = res_3.T * res_3; print(rss_3)

#11
x_test_1 = test_set[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']]
x_test_2 = test_set[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']]
x_test_3 = test_set[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms',
                 'bed_bath_rooms', 'log_sqft_living', 'lat_plus_long']]

Y_test   = test_set['price']
X_test_1 = st.add_constant(x_test_1)
X_test_2 = st.add_constant(x_test_2)
X_test_3 = st.add_constant(x_test_3)

H_test_1 = np.mat(X_test_1)
H_test_2 = np.mat(X_test_2)
H_test_3 = np.mat(X_test_3)
YY_test  = np.mat(Y_test)

res_test_1 = YY_test.T - H_test_1 * w_1.T
rss_test_1 = res_test_1.T * res_test_1; print(rss_test_1)
res_test_2 = YY_test.T - H_test_2 * w_2.T
rss_test_2 = res_test_2.T * res_test_2; print(rss_test_2)
res_test_3 = YY_test.T - H_test_3 * w_3.T
rss_test_3 = res_test_3.T * res_test_3; print(rss_test_3)


#---------------------------------------------------------------------------------------------
#----------                            Problem Set 2                                ----------
#---------------------------------------------------------------------------------------------
def get_numpy_data(data_set, features, output):
    feature_matrix = np.mat(data_set[features])
    feature_matrix = np.column_stack((np.ones(feature_matrix.shape[0]), feature_matrix))
    output_array = np.mat(data_set[output])

    return (feature_matrix, output_array)

def predict_outcome(feature_matrix, weights):
    return feature_matrix * weights

def feature_derivative(errors, feature):
    return -2 * feature.T * errors

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights, dtype = np.float64)
    output  = np.array(output)
    while not converged:
        # compute the predictions based on feature_matrix and weights
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = output - predictions

        gradient_sum_squares = 0
        # while not converged, update each weight individually
        for i in range(len(weights)):
            deriv = feature_derivative(errors, feature_matrix[:, i])
            gradient_sum_squares += deriv ** 2
            weights[i] -= step_size * float(deriv)

        gradient_magnitude = np.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True

    return weights

features_label = ['sqft_living']
output_label = ['price']
feature_matrix, output = get_numpy_data(train_set, features_label, output_label)
initial_weights = np.matrix([-47000., 1.]).T
step_size = 7e-12
tolerance = 2.5e7

#9
the_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
print(the_weights[1])

#10
test_simple_feature_matrix, test_output = get_numpy_data(test_set, features_label, output_label)
test_predictions = predict_outcome(test_simple_feature_matrix, the_weights)

#11
print(test_predictions[0])

#12
res_model_1 = test_output - test_simple_feature_matrix * the_weights
rss_model_1 = res_model_1.T * res_model_1; print(rss_model_1)

#13
model_features = ['sqft_living', 'sqft_living15']
my_output = ['price']
feature_matrix, output = get_numpy_data(train_set, model_features, my_output)
initial_weights = np.matrix([-100000., 1., 1.]).T
step_size = 4e-12
tolerance = 1e9

"train_set['price'].shape"
"train_set[['price']].shape"

another_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
print(another_weights[1])
# st.OLS(train_set['price'], st.add_constant(train_set[['sqft_living', 'sqft_living15']])).fit().params

#14
test_simple_feature_matrix, test_output = get_numpy_data(test_set, model_features, my_output)
another_test_predictions = predict_outcome(test_simple_feature_matrix, another_weights)

#15
print(another_test_predictions[0])

#16
print(test_set['price'][0])

#17
# model 1 gives closer prediction of price of 1st house in test set

#18
res_model_2 = test_output - test_simple_feature_matrix * another_weights
rss_model_2 = res_model_2.T * res_model_2; print(rss_model_2)
import pandas as pd
import numpy as np

train_set = pd.read_csv('kc_house_train_data.csv')
test_set  = pd.read_csv('kc_house_test_data.csv')

input_feature = train_set['sqft_living']
output = train_set['price']

def simple_linear_regression(input_feature, output):
    y_sum = output.sum()
    x_sum = input_feature.sum()
    xy_sum = (input_feature * output).sum()
    w1 = (xy_sum - y_sum * x_sum / output.shape[0]) / ((input_feature * input_feature).sum() - x_sum * x_sum / output.shape[0])
    w0 = y_sum / output.shape[0] - w1 * x_sum / output.shape[0]

    return (w0, w1)

intercept, slope = simple_linear_regression(input_feature, output)

def get_regression_predictions(input_feature, intercept, slope):
    predicted_output = intercept + slope * input_feature

    return predicted_output

y_2650 = get_regression_predictions(2650, intercept, slope)

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    rss = np.sum((output - intercept - slope * input_feature) ** 2)

    return rss

rss = get_residual_sum_of_squares(input_feature, output, intercept, slope)

def inverse_regression_predictions(output, intercept, slope):
    return (output - intercept) / slope

input_feature_800000 = inverse_regression_predictions(800000, intercept, slope)

#11
bedrooms = train_set['bedrooms']

bedrooms_slope, bedrooms_intercept = simple_linear_regression(bedrooms, output)

#12
sqft_living_test = test_set['sqft_living']
bedrooms_test = test_set['bedrooms']
output_test = test_set['price']
rss_sqft_living = get_residual_sum_of_squares(sqft_living_test, output_test, intercept, slope)
rss_bedrooms = get_residual_sum_of_squares(bedrooms, output_test, bedrooms_intercept, bedrooms_slope)

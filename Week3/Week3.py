import numpy as np
import pandas as pd
import statsmodels.api as st
import matplotlib.pyplot as plt

def polynomial_dataframe(feature, degree):
    # initialize the dataframe
    poly_dataframe = pd.DataFrame()
    poly_dataframe['power_1'] = feature
    if degree > 1:
        for power in range(2, degree + 1):
            name = 'power_' + str(power)
            poly_dataframe[name] = feature.apply(lambda x: x **power)

    return poly_dataframe

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str,
              'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales = pd.read_csv('kc_house_data.csv', dtype = dtype_dict)
sales = sales.sort_values(by = ['sqft_living', 'price'])

#4
poly1_data = polynomial_dataframe(sales['sqft_living'], 1)

#5
poly1_data['price'] = sales['price']

#6
X_1 = st.add_constant(poly1_data)
model_1 = st.OLS(X_1['price'], X_1[['const', 'power_1']])
results_1 = model_1.fit()
print(results_1.summary())

#7
plt.plot(poly1_data['power_1'], poly1_data['price'], '.',
         poly1_data['power_1'], results_1.predict(X_1[['const', 'power_1']]), '-')

#8
poly2_data = polynomial_dataframe(sales['sqft_living'], 2)
poly2_data['price'] = sales['price']
poly3_data = polynomial_dataframe(sales['sqft_living'], 3)
poly3_data['price'] = sales['price']
X_2 = st.add_constant(poly2_data)
X_3 = st.add_constant(poly3_data)
results_2 = st.OLS(X_2['price'], X_2[['const', 'power_1', 'power_2']]).fit()
results_3 = st.OLS(X_3['price'], X_3[['const', 'power_1', 'power_2', 'power_3']]).fit()
plt.plot(poly2_data['power_1'], poly2_data['price'], '.',
         poly2_data['power_1'], results_2.predict(X_2[['const', 'power_1', 'power_2']]), '-')
plt.plot(poly3_data['power_1'], poly3_data['price'], '.',
         poly3_data['power_1'], results_3.predict(X_3[['const', 'power_1', 'power_2', 'power_3']]), '-')

#9

#10
set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv')
set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv')
set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv')
set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv')

#11
poly15_set_1 = polynomial_dataframe(set_1['sqft_living'], 15)
poly15_set_2 = polynomial_dataframe(set_2['sqft_living'], 15)
poly15_set_3 = polynomial_dataframe(set_3['sqft_living'], 15)
poly15_set_4 = polynomial_dataframe(set_4['sqft_living'], 15)

poly15_set_1['price'] = set_1['price']
poly15_set_2['price'] = set_2['price']
poly15_set_3['price'] = set_3['price']
poly15_set_4['price'] = set_4['price']

Poly15_set_1 = st.add_constant(poly15_set_1)
Poly15_set_2 = st.add_constant(poly15_set_2)
Poly15_set_3 = st.add_constant(poly15_set_3)
Poly15_set_4 = st.add_constant(poly15_set_4)

results_set_1 = st.OLS(Poly15_set_1['price'], Poly15_set_1.iloc[:, 0:-1]).fit()
results_set_2 = st.OLS(Poly15_set_2['price'], Poly15_set_2.iloc[:, 0:-1]).fit()
results_set_3 = st.OLS(Poly15_set_3['price'], Poly15_set_3.iloc[:, 0:-1]).fit()
results_set_4 = st.OLS(Poly15_set_4['price'], Poly15_set_4.iloc[:, 0:-1]).fit()

#14
train_data = pd.read_csv('wk3_kc_house_train_data.csv')
test_data = pd.read_csv('wk3_kc_house_test_data.csv')
valid_data = pd.read_csv('wk3_kc_house_valid_data.csv')
rss_list = []
for i in range(1, 16):
    poly_data = polynomial_dataframe(train_data['sqft_living'], i)
    poly_data['price'] = train_data['price']
    poly_data = st.add_constant(poly_data)
    poly_valid_data = polynomial_dataframe(valid_data['sqft_living'], i)
    poly_valid_data['price'] = valid_data['price']
    poly_valid_data = st.add_constant(poly_valid_data)
    result = st.OLS(poly_data['price'], poly_data.iloc[:, 0:-1]).fit()
    res = np.mat(poly_valid_data['price']).T - np.mat(poly_valid_data.iloc[:, 0:-1]) * np.mat(result.params).T
    rss = res.T * res
    rss_list.append(rss)

#17
# model with degree 6 gives the lowest RSS on validation data.



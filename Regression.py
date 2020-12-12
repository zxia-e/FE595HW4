import pandas as pd
from sklearn.datasets import load_boston
from sklearn import linear_model

# Load the Boston housing data
boston_dataset = load_boston()
#Converting our data to a data frame format
boston_df = pd.DataFrame(boston_dataset.data,columns = boston_dataset.feature_names)
#make a target for future use
boston_df['target'] = boston_dataset['target']

#Drop the target from original data frame and set the dependent and independent variable for our linear regression model
x = boston_df.drop('target', axis =1)
y = boston_df['target']
lm = linear_model.LinearRegression()
lm.fit(x, y)
print('coefficient : \n ', lm.coef_)

#Get the absolute value for impact and sort them from the greatest to the least
coef_df = pd.DataFrame(data = lm.coef_, columns=['coef_value'])
coef_df = coef_df.abs().sort_values(by = 'coef_value', ascending = False)
print(coef_df)
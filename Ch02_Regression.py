# Extreme Gradient Boosting with XGBoost - DataCamp - January 2020
# Ch2: Regression with XGBoost

#-------------------------------------
#### Setup ####
# Import modules
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # For sklearn API examples

#-------------------------------------
# Notes
''' Supervised learning: regression
Common metrics:
    Root mean squared error (RMSE)
    Mean absolute error (MAE) <- not affected as much by very large errors, but lacks nice mathematical properties
Common algorithms:
    Linear regression
    Decision trees
Objective (loss) functions: Quantifies how far off a prediction is from the actual target
    Aim: find the model that yields the minimum value of loss across ALL the (training) data points
    In xgboost:
        reg:linear <- for regression problems
        reg:logistic <- for classification when you want the decision, not probability
        binary:logistic <- when you want the probability, rather than the decision
Base learners, examples:
    Trees
    Linear models <- this often isn't helpful, because the linear sum of linear models is another linear model
        And we already know how to solve multivariate linear models (by matrix methods)
        However, if you stop the fitting early, it might provide some sort of regularisation
Regularisation = limit model complexity by adding a term to the objective function. Parameters in XGBoost:
    - gamma = minimum loss reduction allowed for a split to occur
    - alpha = L1 regularisation on LEAF weights (NOT on feature weights), larger alpha => more regularisation
    - lambda = L2 regularisation
'''

#-------------------------------------
# Example using sklearn API
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston # Get data (built-in)
boston_bunch = load_boston()
X_bost = pd.DataFrame(boston_bunch.data, columns=boston_bunch.feature_names)
y_bost = boston_bunch.target
y_bost[:10] # Response is continuous
X_train, X_test, y_train, y_test = train_test_split(X_bost, y_bost, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective='reg:linear', n_estimators=10 # Number of boosted trees to fit
                          , seed=123) # booster = "gbtree" is used as default
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % rmse)

# Example using xgboost API
# Get data same as above. Then convert:
DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test = xgb.DMatrix(data=X_test, label=y_test)
params = {"booster":"gblinear", "objective":"reg:linear"} # booster = "gblinear" => linear base learners
xg_reg2 = xgb.train(params=params, dtrain=DM_train, num_boost_round=10)
preds2 = xg_reg2.predict(DM_test)
rmse2 = np.sqrt(mean_squared_error(y_test, preds2))
print("RMSE: %f" % rmse2)

# Another example
df = pd.read_csv('.\\01_Data\\04_ames_housing_trimmed_processed.csv')
df.shape
df.columns # Last column ('SalePrice') must be the response
X_ames, y_ames = df.iloc[:,:-1], df.iloc[:,-1]
housing_dmatrix = xgb.DMatrix(data=X_ames, label=y_ames)
params = {"objective":"reg:linear", "max_depth":4}
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params
                    , nfold=4, num_boost_round=5, metrics="rmse" # Also: "mae"
                    , as_pandas=True, seed=123)
print(cv_results)
(cv_results["test-rmse-mean"]).tail(1) # Final boosting round metric
type(cv_results)

# Example with regularisation
boston_dmatrix = xgb.DMatrix(data=X_bost, label=y_bost) # Data defined above
params = {"objective":"reg:linear", "max_depth":4} # booster = "gbtree" is implied by default
l1_params = [1,10,100] # L1 values to try
rmses_l1 = [] # Initialise list to store results
for reg in l1_params:
    params["alpha"] = reg # L1 values need to go into the 'alpha' parameter
    cv_results = xgb.cv(dtrain=boston_dmatrix, params=params
                    , nfold=4, num_boost_round=5, metrics="rmse"
                    , as_pandas=True, seed=123)
    rmses_l1.append(cv_results["test-rmse-mean"].tail(1).values[0]) # Store final result
print("Best rmse as a function of l1:"); 
print(pd.DataFrame(list(zip(l1_params, rmses_l1)), columns=["l1","rmse"]))
    
# Side note: Common syntax for converting many equal-length lists to a DataFrame:
# pd.DataFrame(list(zip(list_1, list_2)), columns=["list_1","list_2"])
# Uses zip() to get from [1,2,3], [a,b,c] to [1,a],[2,b],[3.c]
# In Python 3, zip() returns a GENERATOR which needs to be cast using list()
    
# Plotting trees
import graphviz as gv
xgb.plot_tree(xg_reg, num_trees=0); plt.show() # First tree
xgb.plot_tree(xg_reg, num_trees=9, rankdir="LR"); plt.show() # Tenth tree, sideways

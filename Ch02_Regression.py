# Extreme Gradient Boosting with XGBoost - DataCamp - January 2020
# Ch2: Regression with XGBoost

# -------------------------------------
# ---- Setup ----
# Import built-in modules
import warnings
import io  # For parsing data scraped from a website (if using an internet proxy: see config_public.py)

# Import external modules
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import __version__ as mpl_version
from pyprojroot import here
from sklearn import __version__ as skl_version
from sklearn.model_selection import train_test_split  # For sklearn API examples
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston  # For Boston housing data set
from requests import get as requests_get  # For getting data from a website (if using an internet proxy)

# Check they have imported OK
print("xgboost version: " + str(xgb.__version__))
print("numpy version: " + str(np.__version__))
print("pandas version: " + str(pd.__version__))
print("matplotlib version: " + str(mpl_version))
print("sklearn version: " + str(skl_version))

# Project locations
data_folder_path = here('.') / 'data'

# Project environment variables - see config_public.py for explanation
try:
    import config_private as project_config
except ImportError:
    import config_public as project_config

# -------------------------------------
# ---- Notes ----
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
        reg:squarederror <- for regression problems (previously reg:linear)
        reg:logistic <- for classification when you want the decision, not probability
        binary:logistic <- when you want the probability, rather than the decision
Base learners, examples:
    Trees
    Linear models <- this often isn't helpful, because the linear sum of linear models is another linear model
        And we already know how to solve multivariate linear models (by matrix methods)
        However, if you stop the fitting early, it might provide some sort of regularisation
Regularisation = limit model complexity by adding a term to the objective function. Parameters in XGBoost:
    - gamma = minimum loss reduction allowed for a split to occur. So: larger gamma => fewer splits
    - alpha = L1 regularisation on LEAF weights (NOT on feature weights). So: larger alpha => more regularisation
        Higher alpha causes many leaf weights to go to zero
    - lambda = L2 regularisation on leaf weights
'''

# -------------------------------------
# ---- Ex01a: Boston housing using sklearn API ----
# Get data and pre-process
boston_bunch = load_boston()
boston_bunch.keys()  # Available attributes
X_boston = pd.DataFrame(boston_bunch.data, columns=boston_bunch.feature_names)
y_boston = pd.Series(boston_bunch.target, name="MEDV")
print(y_boston[:10])  # Response is continuous
X_train, X_test, y_train, y_test = train_test_split(X_boston, y_boston, test_size=0.2, random_state=123)

# Fit model
xg_reg = xgb.XGBRegressor(  # booster = "gbtree" is used as default
    objective='reg:squarederror',  # For linear regression
    n_estimators=10,  # Number of boosted trees to fit
    random_state=123,  # Can't see that this makes any difference
)  # We see that the metric is still decreasing. This is only an example, not tuned for optimal fitting parameters
_ = xg_reg.fit(  # Suppress returning the object
    X_train, y_train,
    eval_set=[(X_test, y_test)],  # Set this to get evals_result out of the model objects
)

# Check the model metric
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("Manual RMSE: %f" % rmse)
print("RMSE from model object: %f" % xg_reg.evals_result()['validation_0']['rmse'][-1])  # Slightly different

# ---- Ex01b: Same, but using xgboost API ----
# Get data same as above. Then convert:
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Series.base is deprecated")
    DM_train = xgb.DMatrix(data=X_train, label=y_train)
    DM_test = xgb.DMatrix(data=X_test, label=y_test)

# Fit the model
params = {
    "booster": "gbtree",  # => linear base learners
    "objective": "reg:squarederror",
}
reg2_evals_result = dict()  # To collect the evaluated metric, we need to store it in a variable
xg_reg2 = xgb.train(
    dtrain=DM_train, params=params, num_boost_round=10,
    evals=[(DM_test, "Test_data")], verbose_eval=True,
    evals_result=reg2_evals_result,
)  # We see that the metric is still decreasing. This is only an example, not tuned for optimal fitting parameters
preds2 = xg_reg2.predict(DM_test)
rmse2 = np.sqrt(mean_squared_error(y_test, preds2))
print("Manual RMSE: %f" % rmse2)
print("RMSE from model object: %f" % reg2_evals_result['Test_data']['rmse'][-1])  # The same to this many dps
# This result does *not* match that generated by the sklearn API example.
# Looks like it could be initialising at a different point. Couldn't pin-point the exact difference.

# -------------------------------------
# ---- Ex02: Ames housing data ----
# Load data and format (should work even if you are not logged in to DataCamp)
ames_url = (
    'https://assets.datacamp.com/production/repositories/943/datasets/'
    '4dbcaee889ef06fb0763e4a8652a4c1f268359b2/ames_housing_trimmed_processed.csv'
)
if project_config.proxy_dict['http'] is None:  # That is, if you have not set a proxy to connect to the internet
    ames_df = pd.read_csv(ames_url)  # If you have not set a proxy, but you need one, this line might hang
else:
    data_str = requests_get(ames_url, proxies=project_config.proxy_dict).text
    ames_df = pd.read_csv(io.StringIO(data_str))
print(ames_df.shape)  # Check it has loaded OK
print(ames_df.columns)  # Last column ('SalePrice') must be the response
X_ames, y_ames = ames_df.iloc[:, :-1], ames_df.iloc[:, -1]
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Series.base is deprecated")
    housing_dmatrix = xgb.DMatrix(data=X_ames, label=y_ames)

# Fit model
params = {"objective": "reg:squarederror", "max_depth": 4}
cv_results = xgb.cv(
    dtrain=housing_dmatrix,
    params=params, nfold=4, num_boost_round=20, metrics="rmse",  # Alternatively: "mae"
    as_pandas=True, seed=123, verbose_eval=True
)
print(cv_results)
print(cv_results["test-rmse-mean"].tail(1))  # Final boosting round metric

# Extra: Plot the learning curve
x_axis = range(1, len(cv_results)+1)
plt.figure()
plt.plot(x_axis, cv_results['train-rmse-mean'], label='Train')
plt.fill_between(
    x_axis,
    cv_results['train-rmse-mean'] - 10*cv_results['train-rmse-std'],  # Wider margin for train so we can actually see
    cv_results['train-rmse-mean'] + 10*cv_results['train-rmse-std'],  # the region (otherwise it is too small)
    alpha=0.1,
)
plt.plot(x_axis, cv_results['test-rmse-mean'], label='Test')
plt.fill_between(
    x_axis,
    cv_results['test-rmse-mean'] - cv_results['test-rmse-std'],
    cv_results['test-rmse-mean'] + cv_results['test-rmse-std'],
    alpha=0.1,
)
plt.legend(); plt.ylabel('RMSE'); plt.title('XGBoost RMSE')
plt.show()

#-------------------------------------
# ---- Ex03: Example with regularisation ----
# Load and transform the data
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Series.base is deprecated")
    boston_dmatrix = xgb.DMatrix(data=X_boston, label=y_boston)  # Data defined above

# Fit various hyper-parameter values to find a suitable model
params = {"objective": "reg:squarederror", "max_depth": 4}  # booster = "gbtree" is implied by default
l1_params = [1,10,100]  # L1 values to try
rmses_l1 = []  # Initialise list to store results
for reg in l1_params:
    params["alpha"] = reg  # L1 values need to go into the 'alpha' parameter
    cv_results = xgb.cv(
        dtrain=boston_dmatrix, params=params,
        nfold=4, num_boost_round=5, metrics="rmse",
        as_pandas=True, seed=123
    )
    rmses_l1.append(cv_results["test-rmse-mean"].tail(1).values[0])  # Store final result

print("We want the best (i.e. lowest) rmse from the available alpha values")
print(pd.DataFrame(list(zip(l1_params, rmses_l1)), columns=["l1", "rmse"]))
    
# Side note: Common syntax for converting many equal-length lists to a DataFrame:
# pd.DataFrame(list(zip(list_1, list_2)), columns=["list_1","list_2"])
# Uses zip() to get from [1,2,3], [a,b,c] to [1,a],[2,b],[3.c]
# In Python 3, zip() returns a GENERATOR which needs to be cast using list()

# Visualising individual trees
xgb.plot_tree(xg_reg, num_trees=0); plt.show() # First tree
xgb.plot_tree(xg_reg, num_trees=9, rankdir="LR"); plt.show() # Tenth tree, sideways

# Note: To use xgb.plot_tree() you need to have graphviz installed.
# This must be done in two conda installs:
# - graphviz <- for the external (non-Python) software.
# - python-graphviz <- for the Python interface that plot_tree uses.
# See: <https://stackoverflow.com/a/47043173>
# It is *not* necessary to import the Python module into the code.

# Extreme Gradient Boosting with XGBoost - DataCamp - January 2020
# Ch3: Fine-tuning XGBoost models

# -------------------------------------
# ---- Setup ----
# Import built-in modules
import warnings
import io  # For parsing data scraped from a website (if using an internet proxy: see config_public.py)

# Import external modules
import xgboost as xgb
import pandas as pd
import numpy as np
from matplotlib import __version__ as mpl_version
from pyprojroot import here
from sklearn import __version__ as skl_version
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
''' Tunable parameters in xgboost
For tree based learner:
- learning_rate (alias: eta) = how quickly the model fits the residual error using additional base learners
    low => more iterations needed [but could fit a more detailed model]
- gamma, alpha, lambda [see Ch02 notes]
- max_depth = how deeply each tree is allowed to grow during each training round
- subsample = % samples used per tree (i.e. fraction of the total training set used for each round)
    low => could find specific patterns in subsets of data, but might underfit
- colsample_bytree = % of columns used per tree
    low => maybe consider this as additional regularisation

For linear learner:
- alpha, lambda [as above]
- lambda_bias = L2 regularisation term on the bias 

Both: Number of boosting round [and early stopping]
'''

# -------------------------------------
# ---- Load and format data ----
# (should work even if you are not logged in to DataCamp)
ames_url = (
    'https://assets.datacamp.com/production/repositories/943/datasets/'
    '4dbcaee889ef06fb0763e4a8652a4c1f268359b2/ames_housing_trimmed_processed.csv'
)
if project_config.proxy_dict['http'] is None:  # That is, if you have not set a proxy to connect to the internet
    housing_data = pd.read_csv(ames_url)  # If you have not set a proxy, but you need one, this line might hang
else:
    data_str = requests_get(ames_url, proxies=project_config.proxy_dict).text
    housing_data = pd.read_csv(io.StringIO(data_str))
print(housing_data.shape)  # Check it has loaded OK
print(housing_data.columns)  # Last column ('SalePrice') must be the response
X, y = housing_data.iloc[:, :-1], housing_data.iloc[:, -1]
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Series.base is deprecated")
    housing_dmatrix = xgb.DMatrix(data=X, label=y)

# -------------------------------------
# ---- Ex01: Basic untuned vs tuned model ----
# Fit *untuned* model
untuned_params = {
    'objective': 'reg:squarederror',
    # The following are the default values for each parameter
    # 'colsample_bytree': 1,  # Percentage of columns to be randomly sampled for each tree
    # 'learning_rate': 0.3,  # Shrink weights on each step (i.e. stop it converging too quickly). Alias 'eta'
    # 'max_depth': 6,  # Maximum depth of a tree
}
untuned_cv_results = xgb.cv(
    dtrain=housing_dmatrix, params=untuned_params, metrics='rmse', num_boost_round=200,
    nfold=4, seed=123, as_pandas=True,
)
print("Untuned rmse: %f" % untuned_cv_results['test-rmse-mean'].tail(1))

# Fit *tuned* model
tuned_params = {
    'objective': 'reg:squarederror',
    # With some non-default (i.e. cutomised) hyper-parameter values
    'colsample_bytree': 0.3, 'learning_rate': 0.1, 'max_depth': 5,
}
tuned_cv_results = xgb.cv(
    dtrain=housing_dmatrix, params=tuned_params, metrics='rmse', num_boost_round=200,
    nfold=4, seed=123, as_pandas=True,
)
print("Tuned rmse: %f" % tuned_cv_results['test-rmse-mean'].tail(1))

# -------------------------------------
# ---- Ex02: Tuning a single hyper-parameter ----
# Load data and format (should work even if you are not logged in to DataCamp)
params = {"objective": "reg:squarederror", "max_depth": 3}

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]  # Could also do this for other parameters
best_rmse = []

# Systematically vary the eta
for curr_val in eta_vals:
    params["eta"] = curr_val

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(
        dtrain=housing_dmatrix,
        params=params, metrics='rmse',
        num_boost_round=10, early_stopping_rounds=5,
        nfold=3, seed=123, as_pandas=True,
    )

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta", "best_rmse"]))

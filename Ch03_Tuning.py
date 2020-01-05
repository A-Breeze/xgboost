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
# TODO: Write notes
''' 
'''

# -------------------------------------
# ---- Ex01: Basic untuned vs tuned model ----
# Load data and format (should work even if you are not logged in to DataCamp)
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

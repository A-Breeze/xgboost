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
from pyprojroot import here
from sklearn import __version__ as skl_version
from requests import get as requests_get  # For getting data from a website (if using an internet proxy)

# Check they have imported OK
print("xgboost version: " + str(xgb.__version__))
print("numpy version: " + str(np.__version__))
print("pandas version: " + str(pd.__version__))
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
# TODO: Write the notes
'''
'''

# -------------------------------------
# ---- Load and format data ----
# TODO: Load the data

# -------------------------------------
# ---- Ex01: TBA ----
# TODO: Write example 1

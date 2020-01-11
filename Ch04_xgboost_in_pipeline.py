# Extreme Gradient Boosting with XGBoost - DataCamp - January 2020
# Ch4: Use XGBoost models in a pipeline

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
from sklearn.datasets import load_boston  # For Boston housing data set
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
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
''' Pipelines in sklearn
- Pipe elements are objects that take a list of "steps" = named 2-tuples (name, pipeline_step)
    Where pipeline_step = an sklearn transformer or estimator object
- Pipeline implements fit/predict methods
- Pipelines can be used as input estimator into e.g. grid search / cross_val_score / etc.

Example pre-processing steps:
- LabelEncoder: converts a categorical column of strings into integers
- OneHotEncoder: takes a column of integers and encodes them as dummy variables
    But... you cannot do LabelEncoder *then* OneHotEncoder in an sklearn Pipeline
- DictVectorizer: Usually used in text processing. 
'''

# -------------------------------------
# ---- Ex01: sklearn pre-processing steps and pipeline ----
# Load data
# (should work even if you are not logged in to DataCamp)
ames_raw_url = (
    # This is a subset of the full Ames data set. Full version is on Kaggle here:
    # <https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data>
    'https://assets.datacamp.com/production/repositories/943/datasets/'
    '17a7c5c0acd7bfa253827ea53646cf0db7d39649/ames_unprocessed_data.csv'
)
if project_config.proxy_dict['http'] is None:  # That is, if you have not set a proxy to connect to the internet
    housing_data = pd.read_csv(ames_raw_url)  # If you have not set a proxy, but you need one, this line might hang
else:
    data_str = requests_get(ames_raw_url, proxies=project_config.proxy_dict).text
    housing_data = pd.read_csv(io.StringIO(data_str))
print(housing_data.shape)  # Check it has loaded OK
print(housing_data.columns)  # Last column ('SalePrice') must be the response

# Explore the data
print(housing_data.info())
print(housing_data.isna().apply(lambda x: x.value_counts()).T)  # Only LotFrontage has missing values
# ...and more exploring is possible

# Pre-process the data
housing_data.LotFrontage = housing_data.LotFrontage.fillna(0)  # Fill missing values

# Want to convert categorical columns to numerical labels
categorical_mask = (housing_data.dtypes == object)
categorical_columns = housing_data.columns[categorical_mask].tolist()
print(housing_data[categorical_columns].head())  # These are the categorical columns
# Note: The sklearn docs say LabelEncoder is for the *target* not the input features as we are using here
# As per: <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>
# Should be using OrdinalEncoder if that's what we want, or OneHotEncoder straight off
housing_data[categorical_columns] = housing_data[categorical_columns].apply(lambda x: LabelEncoder().fit_transform(x))
print(housing_data[categorical_columns].head())  # Check result

# Now want to convert the numerical labels to dummy variables
# Apply OneHotEncoder to categorical columns - output is a numpy array, not a dataframe
data_encoded_arr = np.concatenate((
        housing_data[housing_data.columns[~categorical_mask].to_list()].to_numpy(),
        OneHotEncoder(sparse=False).fit_transform(housing_data[categorical_columns])
), axis=1)
# Look at resulting shapes
print(housing_data.shape)
print(data_encoded_arr.shape)

# Instead, perform the above using a DictVectorizer
dv = DictVectorizer(sparse=False)
df_encoded = dv.fit_transform(housing_data.to_dict("records"))  # list like [{column -> value}, … , {column -> value}]
print(df_encoded[:5,:]) # Print the resulting first five rows
print(dv.vocabulary_)  # Look at the "vocabulary" that has been derived

# Put it into a pipeline
xgb_pipeline = Pipeline([
    ("ohe_onestep", DictVectorizer(sparse=False)),  # One-hot encode the input matrix
    ("xgb_model", xgb.XGBRegressor(objective='reg:squarederror'))
])
xgb_pipeline.fit(  # Fit the pipeline
    X=housing_data.iloc[:, :-1].to_dict("records"),  # X needs to be a dict for the DictVectorizer
    y=housing_data.iloc[:, -1]
)

# -------------------------------------
# ---- Ex02: an sklearn pipeline with cross validation ----
# Get data and pre-process
boston_bunch = load_boston()
print(boston_bunch.keys())  # Available attributes
names = [
    'crime', 'zone', 'industry', 'charles', 'no', 'rooms',
    'age', 'distance', 'radial', 'tax', 'pupil', 'aam', 'lower', 'med_price',
    ]
X = pd.DataFrame(boston_bunch.data, columns=names[:-1])
y = pd.Series(boston_bunch.target, name=names[-1])

# Set up pipeline
rf_pipeline = Pipeline([
    ('st_scaler', StandardScaler()),
    ('rf_model', RandomForestRegressor()),  # Random Forest with the default parameters
])

# Run cross validation (to find the value of the metric on the hold-off of every cv fold) and show result
scores = cross_val_score(
    rf_pipeline, X, y,
    cv=10,  # Uses KFold which does *not* shuffle the data by default, so no random_state is required
    scoring='neg_mean_squared_error',
)
final_avg_rmse = np.mean(np.sqrt(np.abs(scores)))  # Average over all folds
print("Final RMSE: ", final_avg_rmse)

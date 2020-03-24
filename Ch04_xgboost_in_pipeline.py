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
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from requests import get as requests_get  # For getting data from a website (if using an internet proxy)
from sklearn_pandas import __version__ as skl_pandas_version
from sklearn_pandas import DataFrameMapper, CategoricalImputer

# Check they have imported OK
print("xgboost version: " + str(xgb.__version__))
print("numpy version: " + str(np.__version__))
print("pandas version: " + str(pd.__version__))
print("sklearn version: " + str(skl_version))
print("sklearn-pandas version: " + str(skl_pandas_version))

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
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Series.base is deprecated")
    xgb_pipeline.fit(  # Fit the pipeline
        X=housing_data.iloc[:, :-1].to_dict("records"),  # X needs to be a dict for the DictVectorizer
        y=housing_data.iloc[:, -1]
    )

# -------------------------------------
# ---- Ex02: an sklearn pipeline with cross validation ----
# Get data and pre-process
boston_bunch = load_boston()
print(boston_bunch.keys())  # Available attributes
boston_names = [
    'crime', 'zone', 'industry', 'charles', 'no', 'rooms',
    'age', 'distance', 'radial', 'tax', 'pupil', 'aam', 'lower', 'med_price',
    ]
X = pd.DataFrame(boston_bunch.data, columns=boston_names[:-1])
y = pd.Series(boston_bunch.target, name=boston_names[-1])

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
print("Final RMSE: ", final_avg_rmse)  # 4.1585

# ---- Ex02b: Same but with xgboost ----
# Set up pipeline
xgb_pipeline_2b = Pipeline([
    ('st_scaler', StandardScaler()),
    ('xgb_model', xgb.XGBRegressor(objective='reg:squarederror')),  # xgboost step goes here
])
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Series.base is deprecated")
    scores_xgb = cross_val_score(
        xgb_pipeline_2b, X, y,
        cv=10,
        scoring='neg_mean_squared_error',
    )
final_avg_rmse_xgb = np.mean(np.sqrt(np.abs(scores_xgb)))  # Average over all folds
print("Final RMSE for xgb model: ", final_avg_rmse_xgb)  # 4.0272

# ---- Ex02c: Same but with ames housing data ----
# The pipeline was created above
# Use it with CV here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Series.base is deprecated")
    cross_val_scores = cross_val_score(
        xgb_pipeline, 
        X=housing_data.iloc[:, :-1].to_dict("records"),  # X needs to be a dict for the DictVectorizer
        y=housing_data.iloc[:, -1],
        cv=10,
        scoring='neg_mean_squared_error',
    )
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))  # 28440.80

# -------------------------------------
# ---- Ex03: More sklearn pipelines ----
kidney_raw_url = (
    # This is the course copy of the Chronic Kidney data set. Full version is on UCI here:
    # <https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease>
    'https://assets.datacamp.com/production/repositories/943/datasets/'
    '82c231cd41f92325cf33b78aaa360824e6b599b9/chronic_kidney_disease.csv'
)
kidney_feature_names_raw_order = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
    'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
    'wc', 'rc', 
    'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
kidney_feature_names = [
    'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 
    'rc', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
kidney_target_name = 'class'

if project_config.proxy_dict['http'] is None:
    kidney_data = pd.read_csv(
        kidney_raw_url, 
        names=kidney_feature_names_raw_order + [kidney_target_name], index_col=False,
        na_values='?',
        )
else:
    data_str = requests_get(kidney_raw_url, proxies=project_config.proxy_dict).text
    kidney_data = pd.read_csv(
        io.StringIO(data_str), 
        names=kidney_feature_names_raw_order + [kidney_target_name], index_col=False,
        na_values='?',
        )
kidney_data = kidney_data[kidney_feature_names + [kidney_target_name]]  # Match the column order in the exercise
print(kidney_data.shape)  # Check it has loaded OK
print(kidney_data.head())

X = kidney_data.iloc[:, :-1]
y = kidney_data.iloc[:, -1]
print(X.isnull().sum())  # Check number of nulls in each feature column

# Get lists of categorical and non-categorical column names
categorical_feature_mask = X.dtypes == object  # Create a boolean mask for categorical columns
categorical_columns = X.columns[categorical_feature_mask].tolist()
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
    [([numeric_feature], SimpleImputer(strategy="median")) for numeric_feature in non_categorical_columns],
    input_df=True, df_out=True)

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
    [(category_feature, CategoricalImputer()) for category_feature in categorical_columns],
    input_df=True, df_out=True)

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
    ("num_mapper", numeric_imputation_mapper),
    ("cat_mapper", categorical_imputation_mapper)])

# Additional transformer needed
class Dictifier(BaseEstimator, TransformerMixin):
    """Pipeline element (transformer) for calling .to_dict("records") on a DataFrame"""
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if type(X) == pd.core.frame.DataFrame:
            return X.to_dict("records")
        else:
            return pd.DataFrame(X).to_dict("records")

# Create full pipeline
pipeline_3 = Pipeline([
    ("featureunion", numeric_categorical_union),
    ("dictifier", Dictifier()),
    ("vectorizer", DictVectorizer(sort=False)),
    ("clf", xgb.XGBClassifier(max_depth=3))
])

# Perform cross-validation
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Series.base is deprecated")
    warnings.filterwarnings("ignore", category=FutureWarning)
    cross_val_scores_3 = cross_val_score(
        pipeline_3, X, y, 
        scoring="roc_auc", cv=3
    )

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores_3))  # 0.99864

# -------------------------------------
# ---- Ex04: Tuning xgboost hyper-parameters ----
# Use the Boston housing data loaded above
X_bos = pd.DataFrame(boston_bunch.data, columns=boston_names[:-1])
y_bos = pd.Series(boston_bunch.target, name=boston_names[-1])

# Set up pipeline as usual
xgb_pipeline_4 = Pipeline([
    ('st_scaler', StandardScaler()),
    ('xgb_model', xgb.XGBRegressor(objective='reg:squarederror', random_state=5))
])

# Create grid of hyper-parameters to search over
gbm_param_grid = {
    # Key for each element must be: <pipeline-step-name>__<hyper-parameter-name>
    'xgb_model__subsample': np.linspace(.8, 1, 3, endpoint=True),
    'xgb_model__max_depth': np.arange(3, 6, 1),
    'xgb_model__colsample_bytree': np.linspace(.4, 1, 3, endpoint=True),
}

# Initialise random search object
randomized_neg_mse = RandomizedSearchCV(
    estimator=xgb_pipeline_4,
    param_distributions=gbm_param_grid, n_iter=10,
    scoring='neg_mean_squared_error', cv=4,
    random_state=123,
)

# Run random search and view results
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Series.base is deprecated")
    search_results = randomized_neg_mse.fit(X_bos, y_bos)

print("Best RMSE: ", np.sqrt(np.abs(search_results.best_score_)))  # 4.41504

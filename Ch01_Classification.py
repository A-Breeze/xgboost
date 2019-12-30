# Extreme Gradient Boosting with XGBoost - DataCamp - January 2020
# Ch1: Classification with XGBoost

#-------------------------------------
#### Setup ####
# Import modules
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # For sklearn API examples

#-------------------------------------
# Notes
''' Supervised learning: classification
Response is either: 
    Binary
    Multi-class 
Common evaluation metric for BINARY classification problems: 
    AUC = Area under the ROC (= Receiver Operating Characteristic) curve
        = Probability that a randomly chosen positive data point (i.e. with response 1) 
        will have a higher rank than than a randomly chosen negative data point (i.e. with response 0)
        where 'rank' means the prediction from your model
For MULTI-CLASS models, common to use:
    Accuracy score (for each class) = (tp + tn) / (tp + tn + fp + fn), where:
        tp = true +ve (correcly predict the point IS from the class)
        tn = true -ve (correctly predict the point is NOT from the class)
        fp = false +ve (predict it IS from the class when it is NOT)
        fn = false -ve (predict it is NOT from the class when it IS)
Common algorithms:
    Logistic regression
    Decision trees
Features (aka attributes, predictors, explanatory variables, exogenous variables) can be:
    Categorical - these are usually encoded, e.g. by one-hot encoding (= creating dummy variables that is each binary)
    Numeric (i.e. continuous) - these are usually scaled (aka Z-scored, normalised)
Other types of supervised learning problems:
    Ranking problems -> predict the ORDERING
    Recommendation problems -> recommend an item to a user based on consumption history and profile
'''

''' Introducing XGBoost
What is XGBoost:
    Optimised gradient-boosting machine learning library
    Originally written in C++
    Has APIs (aka bindings) in various languages: Python, R, Scala, Julia, Java
    Core algorithm is parallelisable

Base learner = an individual learning algorithm in an ensemble algorithm
    Want a base learner to be good at predicting on a subset of the dataset...
    ...and uniformly bad at predicting the rest of the dataset
e.g. a decision tree = series of binary questions
    Constructed iteratively (i.e. one binary decision at a time), until a stopping criterion is met (e.g. depth of tree)
    Want to choose a split point to separate the target values better => each leaf should be largely one category
XGBoost uses a Classification and Regression Tree (CART):
    Each leaf ALWAYS contains a real-valued score (whether this is a classification or regresssion problem)

Boosting = an ensemble meta-algorithm to convert many weak learners to a strong learner
    Weak learner = predictions are slightly better than chance
    Strong learner = can be tuned to achieve arbitrarily high performance
    How? 
        - Iteratively learning a set of weak models on subsets of the data
        - Weighting each weak prediction according to the weak learner's performance
        => Combine to obtain a single weighted prediction
    Model evaluation through cross-validation

When to use XGBoost:
    - You have a large number of training samples (e.g. less than 100 features, at least 1000 obs)
    - Mixture of categorical and numeric features, or just numeric
'''

#-------------------------------------
# Example 01
# Quick example - using xgboost, but with the sklearn data structure
# Get data
class_data = pd.read_csv('.\\01_Data\\02_titanic_all_numeric.csv') # Info: https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease
class_data.head() # Check it has loaded OK
class_data.shape
# Explanatory and response variables
X, y = class_data.iloc[:,1:], class_data.iloc[:,0] # Response is first column: 0 or 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
# Compile and run model
xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed = 123)
xg_cl.fit(X_train, y_train)
# Get predictions on test set and evaluate accuracy
preds = xg_cl.predict(X_test)
accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
print("accuracy: %f" % (accuracy))

# Decision tree example from sklearn
from sklearn.datasets import load_breast_cancer # Data built-in with sklearn
from sklearn.tree import DecisionTreeClassifier
# Get data and split into training and test
bc_data_bunch = load_breast_cancer()
X = pd.DataFrame(bc_data_bunch.data, columns=bc_data_bunch.feature_names)
y = bc_data_bunch.target
X.head() # Check it looks OK
y[:10] # Response is 0 or 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# Modelling
dt_clf_4 = DecisionTreeClassifier(max_depth=4) # Instantiate the classifier. Tree will stop at level 4
dt_clf_4.fit(X_train, y_train) # Fit
y_pred_4 = dt_clf_4.predict(X_test) # Predict on test set
accuracy = float(np.sum(y_pred_4==y_test))/y_test.shape[0]
print("accuracy:", accuracy) # 0.974

# Quick example - amended to use xgboost data structure
# Explanatory and response variables
class_dmatrix = xgb.DMatrix(data = class_data.iloc[:,1:], label = class_data.iloc[:,0])
params = {"objective":"binary:logistic", "max_depth":4} # Create parameters dictionary
# Run it using cross validation
cv_results = xgb.cv(dtrain=class_dmatrix, params=params
                    , nfold=4 # Num of cross validation folds
                    , num_boost_round=10 # How many trees to build (i.e. number of boosting iterations)
                    , metrics="error" # Other options: "auc"
                    , as_pandas=True # Output to be a pandas df
                    , seed=123) # For reproducibility
cv_results # Note that the test error decreases
print("Accuracy: %f" %((1-cv_results["test-error-mean"]).iloc[-1])) # Print resulting error on test
# The same but with AUC as metric
cv_results_auc = xgb.cv(dtrain=class_dmatrix, params=params
                    , nfold=4, num_boost_round=10, metrics="auc"
                    , as_pandas=True, seed=123)
cv_results_auc
print((cv_results_auc["test-auc-mean"]).iloc[-1]) # Closer to 1 is better
# Note this is "-mean" because it is the mean over cross-validation folds

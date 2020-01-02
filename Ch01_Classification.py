# Extreme Gradient Boosting with XGBoost - DataCamp - January 2020
# Ch1: Classification with XGBoost

# -------------------------------------
# ---- Setup ----
# Import built-in modules
import warnings

# Import external modules
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyprojroot import here
from sklearn.model_selection import train_test_split  # For sklearn API examples
from sklearn.datasets import load_breast_cancer  # Data built-in with sklearn
from sklearn.tree import DecisionTreeClassifier  # For Decision Tree example
from sklearn.metrics import roc_curve, roc_auc_score

# Check they have imported OK
print("xgboost version: " + str(xgb.__version__))
print("numpy version: " + str(np.__version__))
print("pandas version: " + str(pd.__version__))

# Project locations
data_folder_path = here('.') / 'data'

# -------------------------------------
# ---- Notes ----
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
        tp = true +ve (correctly predict the point IS from the class)
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
    Want a base learner to be good at predicting on a subset of the data set...
    ...and uniformly bad at predicting the rest of the data set
e.g. a decision tree = series of binary questions
    Constructed iteratively (i.e. one binary decision at a time), until a stopping criterion is met (e.g. depth of tree)
    Want to choose a split point to separate the target values better => each leaf should be largely one category
    Individual decision trees tend to overfit = low bias + high variance [i.e. worse fit test than training data]
XGBoost uses a Classification and Regression Tree (CART):
    Each leaf ALWAYS contains a real-valued score (whether this is a classification or regression problem)
    For classification, the real-valued score can be threshold-ed to convert to a categorical prediction

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

# -------------------------------------
# ---- Ex01: Basic xgboost, with the sklearn workflow ----
# Load data
churn_data = pd.read_csv(data_folder_path / 'churn_data.csv')
print(churn_data.head())  # Check it has loaded OK
# Can also look at in PyCharm: Python Console - On the right of the variable name "View as DataFrame"
# Example: <https://www.jetbrains.com/help/pycharm/viewing-as-array.html>
print(churn_data.shape)

# Explanatory and response variables
X, y = churn_data.iloc[:, :-1], churn_data.iloc[:, -1]  # Response is last column: 0 or 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Compile and run model
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, random_state=123)
xg_cl.fit(X_train, y_train)
help(xgb.XGBClassifier)
# Get predictions on test set and evaluate accuracy
preds = xg_cl.predict(X_test)
accuracy = float(np.sum(preds == y_test)) / y_test.shape[0]
print("accuracy: %f" % accuracy)

# -------------------------------------
# ---- Ex02: Decision tree from sklearn ----
# Get data and split into training and test
bc_data_bunch = load_breast_cancer()
X = pd.DataFrame(bc_data_bunch.data, columns=bc_data_bunch.feature_names)
y = bc_data_bunch.target
print(X.head())  # Check it looks OK
print(y[:10])  # Response is 0 or 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Modelling
dt_clf_4 = DecisionTreeClassifier(  # Instantiate the classifier.
    max_depth=4,  # Tree will stop at level 4
    random_state=123  # Random fitting process, so want to ensure reproducibility
    # The fitting process is random because it is a greedy algorithm, as per: <https://stackoverflow.com/a/39158831>
)
dt_clf_4.fit(X_train, y_train)  # Fit
y_pred_4 = dt_clf_4.predict(X_test)  # Predict on test set
accuracy = float(np.sum(y_pred_4 == y_test))/y_test.shape[0]
print("accuracy:", accuracy)  # 0.965

# -------------------------------------
# ---- Ex03: Ex01 amended to use xgboost API ----
# Data loaded as in Ex01
# Put explanatory and response variables into an xgboost data format
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Series.base is deprecated")
    churn_dmatrix = xgb.DMatrix(data=churn_data.iloc[:, :-1], label=churn_data.iloc[:, -1])
    # Gives warning about "Series.base will be deprecated". It can be ignored.
    # Should be fixed in xgboost 1.0.0, as per: https://github.com/dmlc/xgboost/issues/4300#issuecomment-508589063

# Run it using cross validation
params = {"objective": "binary:logistic", "max_depth": 4}  # Create parameters dictionary
cv_results = xgb.cv(
    dtrain=churn_dmatrix, params=params,
    nfold=4,  # Num of cross validation folds
    num_boost_round=10,  # How many trees to build (i.e. number of boosting iterations)
    metrics="error",  # Other options: "auc"
    as_pandas=True,  # Output to be a pandas df
    seed=123,  # For reproducibility
)
print(cv_results)  # Note that the train error decreases, but the test error is fluctuating
# Note this is "-mean" because it is the mean over cross-validation folds
print("Accuracy: %f" % ((1 - cv_results["test-error-mean"]).iloc[-1]))  # Print resulting error on test. 0.741

# The same but with AUC as metric
cv_results_auc = xgb.cv(
    dtrain=churn_dmatrix, params=params,
    nfold=4, num_boost_round=10, metrics="auc",
    as_pandas=True, seed=123
)
print(cv_results_auc)
print((cv_results_auc["test-auc-mean"]).iloc[-1])  # AUC closer to 1 is better

# ---- Extras ----
# The same but with early-stopping
cv_results_auc_es = xgb.cv(
    dtrain=churn_dmatrix, params=params,
    nfold=4, metrics="auc",
    early_stopping_rounds=10,  # i.e. the metric needs to improve at least once in every X rounds
    num_boost_round=50,  # Set to a large number and hope it early stops before then
    as_pandas=True, seed=123,
    verbose_eval=True,  # Show progress
)
print(cv_results_auc_es)
print((cv_results_auc_es["test-auc-mean"]).iloc[-1])  # AUC closer to 1 is better


# Once you've found your desired hyper parameters, you need to fit a model object with them
# You still need to have a train-test split in order to early stop based on the metric evaluated on the test
# Note: We'll see that the performance is sensitive to the choice of data, which we can see if we change the seed
def get_model(random_state, verbose_eval=False):
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        churn_data.iloc[:, :-1], churn_data.iloc[:, -1],
        test_size=1/4, random_state=random_state
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Series.base is deprecated")
        churn_dtrain = xgb.DMatrix(data=X_train, label=y_train)
        churn_dtest = xgb.DMatrix(data=X_test, label=y_test)
    # Fit model
    churn_model = xgb.train(
        dtrain=churn_dtrain, verbose_eval=verbose_eval,
        params={**params, **{'eval_metric': 'auc'}},  # Add the metric to the params argument
        evals=[(churn_dtest, "Test_data")],
        early_stopping_rounds=10, num_boost_round=50,
    )
    return churn_model, {'churn_dtrain': churn_dtrain, 'churn_dtest': churn_dtest}


# Try one seed
churn_model1, _ = get_model(123)
print("Best AUC: {:.2f} in {} rounds".format(churn_model1.best_score, churn_model1.best_iteration+1))

# ...get quite a different result with another seed
churn_model2, data2 = get_model(42)
print("Best AUC: {:.2f} in {} rounds".format(churn_model2.best_score, churn_model2.best_iteration+1))

# Plot the ROC
predicted_scores = churn_model2.predict(data2['churn_dtest'], ntree_limit=churn_model2.best_ntree_limit)
roc_auc = roc_auc_score(
    data2['churn_dtest'].get_label(),  # True binary labels
    predicted_scores
)
fpr, tpr, _ = roc_curve(
    data2['churn_dtest'].get_label(),  # True binary labels
    churn_model2.predict(data2['churn_dtest'])  # Target scores
)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

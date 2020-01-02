# Extreme Gradient Boosting with XGBoost - DataCamp - January 2020
# Rough notes only

# -------------------------------------
# ---- Setup ----
# Import built-in modules
import os

# Import external modules
import pandas as pd
from pyprojroot import here

# -------------------------------------
# ---- Get data from the DataCamp console ----
""" When data is used in an exercise but not available from the dashboard, we need to get it from the console.
One way is to print it as if it were a CSV and then copy-paste
"""

# Code in the console
churn_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})  # Example data in the console that we want to download

# The DataCamp console crashed for over 10,000 rows.
# So I decide to limit it to 1000, just for practice purposes (and to limit the git repo size)
x = churn_data[:1000].to_string().split('\n')
df_str = '\n'.join([','.join(line.split()) for line in x])
print(df_str)  # Copy the output from the console screen and save it as a variable called df_str

# Save it into a file
data_folder_path = here('.') / 'data'
with open(str(data_folder_path / 'test_data.csv'), 'w') as f:
    print(df_str, file=f)

# Reload it to check it works
df_reloaded = pd.read_csv(data_folder_path / 'test_data.csv')
print(df_reloaded.shape)

# Check summary results against those in the console
df_reloaded.info()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_reloaded.describe(include='all'))

# Delete the file because it was only a test
os.remove(str(data_folder_path / 'test_data.csv'))

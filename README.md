<!-- To view this file rendered, try opening VSCode and clicking to open the "Preview" pane -->
# xgboost - notes

## Contents
<!-- This contents is kept up to date *manually* -->
1. [How to start this project environment](#How-to-start-this-project-environment)
1. [Links to resources](#Further-resources)

<div align="right"><a href="#contents">Back to top</a></div>

## How to start this project environment
1. Clone the repo and navigate to the root directory (in Anaconda Prompt).
1. Set up the conda environment:
    ```cmd
    (base) > conda env create -f environment.yml
    ```
1. Activate the conda env and start PyCharm. This ensures that the PyCharm terminal window opens in the correct conda env.
    ```
    (base) > conda activate xgboost_training
    (xgboost_training) > "%LOCALAPPDATA%\JetBrains\PyCharm Community Edition 2019.3\bin\pycharm64.exe"  # Or wherever your PyCharm is installed
    ```
1. Create a new project in the root directory, with the project's interpreter set to the conda-env's interpreter, which you get into the clipboard by:
    ```
    (xgboost_training) > where python | clip
    ```
<div align="right"><a href="#contents">Back to top</a></div>

## Further resources
DataCamp course: <https://www.datacamp.com/courses/extreme-gradient-boosting-with-xgboost>

Further resources:
- CambridgeSpark article for "Hyperparameter tuning in XGBoost": <https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f>
- `xgboost` Python package intro and API reference: <https://xgboost.readthedocs.io/en/latest/python/index.html>
- Guide to `xgboost` parameters: <https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/>
    - Also in the `xgboost` documentation: <https://xgboost.readthedocs.io/en/latest/parameter.html>

<div align="right"><a href="#contents">Back to top</a></div>

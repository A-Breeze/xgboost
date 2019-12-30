<!-- To view this file rendered, try opening VSCode and clicking to open the "Preview" pane -->
# xgboost - DataCamp course notes
Link to course here: <https://www.datacamp.com/courses/extreme-gradient-boosting-with-xgboost>

## Contents
<!-- This contents is kept up to date *manually* -->
1. [How to start this project environment](#How-to-start-this-project-environment)
1. [Further resources](#Further-resources)

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
TODO: Add further resources

<div align="right"><a href="#contents">Back to top</a></div>

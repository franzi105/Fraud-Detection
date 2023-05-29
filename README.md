# Machine Learning Project on Fraud Detection 

This is a machine learning project based on the Zindi challenge [here](https://zindi.africa/learn/fraud-detection-in-electricity-and-gas-consumption-challenge-tutorial).
In this first machine learning project we are starting to explore the dataset using exploratory data analysis and visualisation tools in python. We are testing different algorithms to detect electricity fraud. By running hyperparameter tuning we are optimizing the best models to improve our metric (f1 score).

## Notebooks overview
* Notebook 01: Data Preparation

    * This notebook takes in the raw data of the Zindi challenge and runs the initial EDA and cleaning of the dataset. In the end the final dataset used for further training of the model is generated and saved as csv file.
* Notebook 02: Baseline Model

    * This notebook creates an initial baseline model using the Dummy Classifier. This result will be taken as a benchmark to compare the performance of our final models to.
* Notebook 03: Model

    * In this notebook the final dataset is being used to test different algorithms and detect which model works the best for our dataset. In the end hyperparemeter tuning will be performed to improve the performance of the model.

---
## Requirements and Environment

Requirements:
- pyenv with Python: 3.9.8

Environment: 

For installing the virtual environment you can either use the Makefile and run `make setup` or install it manually with the following commands: 

```Bash
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

Machine Learning project based on `skikit-learn` to identify credit card customers that are most likely to churn.

## Files and data description

### data

`bank_data.csv` data for the training

### images

Generated images will be found here

### logs

Pytest log files

### models

generated models will be found here

### src

source files

### .pylintrc

`pylint configuration

### mypy.ini

`mypy` typecheck configuration

### pyproject.toml

Configuration for `pytest` logging

### README.md

This README

### requirements.txt

Requirements file

## Installation

- Make sure you have an up to date python `3.10`
- create a virtual environment: `virtualenv venv`
- activate virtual environment: `. venv/bin/activate`
- install dependencies: `pip install -r requirements.txt`

## Code Quality

### pylint

Run `pylint` on codebase

`pylint src/`

### mypy

Typechecking with `mypy`

`mypy src/`

### pytest

Start the testsuite with

`pytest src/`

## Running Files

To start the training, run:

`python src/churn_main.py`

You should see the following output

```
Importing data from ./data/bank_data.csv
Performing EDA...
Encoding...
Perform Feature Engineering...
Train Model....
Done!
```

Then you find the generates models in `models/` and images/plots in `images/`

### Cleanup

To remove all generated files run:

`python src/churn_clean.py`


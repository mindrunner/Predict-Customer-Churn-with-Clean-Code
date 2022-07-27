"""Churn Library

Author: Lukas Elsner
Date: 2022-07-27

This library contains all important functions for the Churn machine learning library

"""

from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import BaseForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV


def import_data(pth: str) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    data_frame = pd.read_csv(pth)
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data_frame


def perform_eda(data_frame: pd.DataFrame) -> None:
    """
    perform eda on df and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    """
    sns.set()

    figsize = (20, 10)

    plt.figure(figsize=figsize)

    data_frame["Churn"].hist()
    plt.savefig('images/eda/Churn.png')

    plt.figure(figsize=figsize)
    data_frame["Customer_Age"].plot.hist()
    plt.savefig('images/eda/Customer_Age.png')

    plt.figure(figsize=figsize)
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/Martial_Status.png')

    plt.figure(figsize=figsize)
    sns.histplot(
        data_frame['Total_Trans_Ct'],
        stat='density',
        kde=True).figure.savefig('images/eda/Total_Trans_Ct.png')
    plt.figure(figsize=figsize)
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r',
                linewidths=2).figure.savefig('images/eda/Heatmap.png')


def encoder_helper(
        data_frame: pd.DataFrame,
        category_lst: list[str],
        response: str) -> pd.DataFrame:
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
                      [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    for category in category_lst:
        encoded_lst = []
        groups = data_frame.groupby(category).mean()['Churn']

        for val in data_frame[category]:
            encoded_lst.append(groups.loc[val])

        data_frame[f'{category}_{response}'] = encoded_lst

    return data_frame


def perform_feature_engineering(
        data_frame: pd.DataFrame,
        response: str) -> Any:
    """
    input:
              df: pandas dataframe
              response: string of response name
                        [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    train_df = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        f'Gender_{response}',
        f'Education_Level_{response}',
        f'Marital_Status_{response}',
        f'Income_Category_{response}',
        f'Card_Category_{response}']

    train_df[keep_cols] = data_frame[keep_cols]

    return train_test_split(
        train_df,
        data_frame['Churn'],
        test_size=0.3,
        random_state=42)


def classification_report_image(y_train: pd.Series,
                                y_test: pd.Series,
                                y_train_preds_lr: npt.NDArray[np.int_],
                                y_train_preds_rf: npt.NDArray[np.int_],
                                y_test_preds_lr: npt.NDArray[np.int_],
                                y_test_preds_rf: npt.NDArray[np.int_]) -> None:
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/results/Classification_Report.png')


def feature_importance_plot(
        model: BaseForest,
        x_data: pd.DataFrame,
        output_pth: str) -> None:
    """
    creates and stores the feature importance in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # Calculate feature importance
    importances = model.feature_importances_
    # Sort feature importance in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importance
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series) -> None:
    """
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf: npt.NDArray[np.int_] = cv_rfc.best_estimator_.predict(
        x_train)
    y_test_preds_rf: npt.NDArray[np.int_] = cv_rfc.best_estimator_.predict(
        x_test)

    y_train_preds_lr: npt.NDArray[np.int_] = lrc.predict(x_train)
    y_test_preds_lr: npt.NDArray[np.int_] = lrc.predict(x_test)

    feature_importance_plot(
        cv_rfc.best_estimator_,
        x_train,
        './images/results/feature_importance.png')
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

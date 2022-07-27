"""Churn Entry Point

Author: Lukas Elsner
Date: 2022-07-27

Main entry point to start the Churn Training

"""

import logging
import sys

from churn.churn_library import perform_feature_engineering, import_data, encoder_helper, train_models, perform_eda
from churn.constants import BANK_DATA_CSV, CATEGORY_COLUMNS, RESPONSE

logger = logging.getLogger('Churn')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':
    logger.info('Importing data from {}', BANK_DATA_CSV)
    data_frame = import_data(BANK_DATA_CSV)
    logger.info('Performing EDA...')
    perform_eda(data_frame)
    logger.info('Encoding...')
    encoded_df = encoder_helper(data_frame, CATEGORY_COLUMNS, RESPONSE)
    logger.info('Perform Feature Engineering...')
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        encoded_df, RESPONSE)
    logger.info('Train Model....')
    train_models(x_train, x_test, y_train, y_test)
    logger.info('Done!')

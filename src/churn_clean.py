"""Churn Library Cleanup

Author: Lukas Elsner
Date: 2022-07-27

This script is used to clean all generated files.

"""

import os

if __name__ == '__main__':

    folders = [
        'images/eda',
        'images/results',
        'logs',
        'models'
    ]

    for folder in folders:
        filelist = [f for f in os.listdir(
            f'./{folder}') if not f.startswith('.')]
        for f in filelist:
            os.remove(os.path.join(f'./{folder}', f))

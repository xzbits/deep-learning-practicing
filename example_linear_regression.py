import os
from src.linear_regression import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Sources: https://www.kaggle.com/mirichoi0218/insurance
# Content: 
# age: age of primary beneficiary
#
# sex: insurance contractor gender, female, male
#
# bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
# objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
#
# children: Number of children covered by health insurance / Number of dependents
#
# smoker: Smoking
#
# region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
#
# charges: Individual medical costs billed by health insurance


#       age     sex     bmi  children smoker     region      charges
# 0      19  female  27.900         0    yes  southwest  16884.92400
# 1      18    male  33.770         1     no  southeast   1725.55230
# 2      28    male  33.000         3     no  southeast   4449.46200
# 3      33    male  22.705         0     no  northwest  21984.47061
# ...   ...     ...     ...       ...    ...        ...          ...
# 1336   21  female  25.800         0     no  southwest   2007.94500
# 1337   61  female  29.070         0    yes  northwest  29141.36030

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "database", "insurance.csv")
    data = pd.read_csv(data_path)

    # Pre-processing data
    data = data.replace(['female', 'male', 'no', 'yes', 'southwest', 'southeast', 'northwest', 'northeast'],
                        [0, 1, 0, 1, 0, 1, 2, 3])
    data = data.to_numpy()
    x = data[:, :6]
    y = np.array([data[:, 6]]).T
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Solve
    solution = LinearRegression(x_train, y_train)

    # Info:
    print("Length of test set:  {}".format(x_test.shape[0]))
    print("RMSD of predicted outputs and test values:  {}".format(solution.validate(x_test, y_test)))
    print("Test set mean value:    {}".format(np.mean(y_test)))

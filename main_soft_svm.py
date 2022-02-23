import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from src.support_vector_machine import SupportVectorMachine


def line_c(x, line_coef):
    return (-line_coef[0] * x - line_coef[2]) / line_coef[1]


if __name__ == "__main__":
    # Data is not linear separable
    data_path = os.path.join(os.path.dirname(__file__), "database", "iris.data")
    data = pd.read_csv(data_path)
    data = data.replace(["Iris-setosa", "Iris-versicolor", "Iris-virginica"], [0, 1, 2])
    data = data.to_numpy()
    data_points = data[51:150, :2]
    data_labels = np.where(np.array([data[51:150, 4]]).T == 2, 1, -1)

    # Classified by Support Vector Machine
    x_train, x_test, y_train, y_test = train_test_split(data_points, data_labels, test_size=0.2, random_state=200)
    svm_solve = SupportVectorMachine(x_train, y_train)
    w_b = svm_solve.solver_soft_svm_dual_problem(0.1)
    w_b_1 = svm_solve.solver_soft_svm_dual_problem(1)
    w_b_10 = svm_solve.solver_soft_svm_dual_problem(10)
    w_b_100 = svm_solve.solver_soft_svm_dual_problem(100)

    # The dataset (1st and 2nd column) of Iris-versicolor and Iris-virginica is not linearly separable,
    # therefore, the result do not achieve 100%.
    print("Validating C = 0.1:  {}%".format(svm_solve.validate(x_test, y_test, w_b)))
    print("Validating C = 1:  {}%".format(svm_solve.validate(x_test, y_test, w_b_1)))
    print("Validating C = 10:  {}%".format(svm_solve.validate(x_test, y_test, w_b_10)))
    print("Validating C = 100:  {}%".format(svm_solve.validate(x_test, y_test, w_b_100)))

    # Visualizing data and result
    x0 = data_points[:50, 0]
    x1 = data_points[:50, 1]

    y0 = data_points[50:100, 0]
    y1 = data_points[50:100, 1]

    d1 = [4.5, 8]
    dc = [line_c(d1[0], w_b), line_c(d1[1], w_b)]
    dc_1 = [line_c(d1[0], w_b_1), line_c(d1[1], w_b_1)]
    dc_10 = [line_c(d1[0], w_b_10), line_c(d1[1], w_b_10)]
    dc_100 = [line_c(d1[0], w_b_100), line_c(d1[1], w_b_100)]

    plt.plot(x0, x1, 'go', label='versicolor')
    plt.plot(y0, y1, 'r^', label='Virginica')
    plt.plot(d1, dc, 'b-', label='C = 0.1')
    plt.plot(d1, dc_1, 'y-', label='C = 1')
    plt.plot(d1, dc_10, 'm-', label='C = 10')
    plt.plot(d1, dc_100, 'r-', label='C = 100')
    plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.0)
    plt.show()

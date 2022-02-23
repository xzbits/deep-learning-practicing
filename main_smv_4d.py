import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from src.support_vector_machine import SupportVectorMachine


if __name__ == "__main__":
    # Data is linear separable
    data_path = os.path.join(os.path.dirname(__file__), "database", "iris.data")
    data = pd.read_csv(data_path)
    data = data.replace(["Iris-setosa", "Iris-versicolor", "Iris-virginica"], [0, 1, 2])
    data = data.to_numpy()
    data_points_3d = data[:100, :4]
    data_labels_3d = np.where(np.array([data[:100, 4]]).T == 0, 1, -1)

    # Classified by Support Vector Machine
    x_train, x_test, y_train, y_test = train_test_split(data_points_3d, data_labels_3d, test_size=0.2, random_state=200)
    svm_solve = SupportVectorMachine(x_train, y_train)
    w_b = svm_solve.solver_svm()
    w_b_dual = svm_solve.solve_dual_problem()

    print("Solve: {}".format(w_b.T))
    print("Solve dual problem: {}".format(w_b_dual))

    # The dataset for Iris-setosa and Iris-versicolor is linearly separable,
    # therefore, the result will achieve 100% easily
    print("Validating:  {}% test sample classified correctly".format(svm_solve.validate(x_test, y_test, w_b_dual)))


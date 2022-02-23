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
    data_points_3d = data[:100, :3]
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


    def plan(x, y, p):
        z = (-p[0]*x - p[1]*y - p[3])/p[2]
        return z


    # Plotting data
    x0 = data_points_3d[:50, 0]
    x1 = data_points_3d[:50, 1]
    x2 = data_points_3d[:50, 2]

    y0 = data_points_3d[50:100, 0]
    y1 = data_points_3d[50:100, 1]
    y2 = data_points_3d[50:100, 2]

    xx = np.array([[2.0, 6.0, 2.0, 6.0]])
    yy = np.array([[2.0, 2.0, 6.0, 6.0]])
    xx1, yy1 = np.meshgrid(np.array(xx), np.array(yy))
    zz = np.array(plan(np.ravel(xx1), np.ravel(yy1), w_b.T[0]))
    zz1 = zz.reshape(xx1.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x0, x1, x2, c='r', marker='o')
    ax.scatter(y0, y1, y2, c='b', marker='^')
    ax.plot_surface(xx1, yy1, zz1)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

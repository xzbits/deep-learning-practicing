import numpy as np


class LinearRegression:
    def __init__(self, x, y):
        """
        x = [[x_1_1, x_1_2, ..., x_1_j],
             [x_2_1, x_2_2, ..., x_2_j],
             .
             .
             .
             [x_i_1, x_i_2, ..., x_i_j]]

        y = [[y_1, y_2, ..., y_i]]
        :param x: data points matrix with dimension is ixj, with i - number of data points, j - number of data point
        dimension
        :param y: outputs vector with dimension is ix1, with i - number of data points
        """
        self.x = x
        self.y = y
        self.w = self.solver()

    def no_data_points(self):
        """
        Get data points property
        :return: number of data point.
        """
        return self.x.shape[0]

    def no_data_point_dim(self):
        """
        Get data points property
        :return: number of dimension of data point
        """
        return self.x.shape[1]

    def solver(self):
        """
        Minimizing the loss function L = 0.5 * (sum(y_i - x_datapoint_i * w))^2
        with:   x_datapoint_i is i-th data point
                y_i is i-th output
                w is the optimal coefficients
        :return: w - the optimal coefficients
        """
        ones = np.ones((self.no_data_points(), 1), dtype=float)
        x_bar = np.concatenate((self.x, ones), axis=1)

        A = np.dot(x_bar.T, x_bar)
        A = A.astype("float")
        b = np.dot(x_bar.T, self.y)
        w = np.dot(np.linalg.pinv(A), b)

        return w

    def cal_predicted_output(self, x):
        """
        Calculating predicted output
        x = [x_1, x_2, ..., x_i, 1]
        with: i - data point dimension
        :param x: data points
        :return: Predicted output
        """
        return np.dot(x, self.w)

    def validate(self, x_test, y_test):
        """
        Validating the linear regression result by RMSD (Root-mean-square deviation)
        RMSD = Square(SUM((y_i-y'_i)^2)/N)
        with:   i       = 1, 2, ..., N
                N       - Number of data points
                y'_i    - Predicted value
                y_i     - Test value
        :param x_test: Test data points
        :param y_test: Test output
        :return: Root-mean-square deviation
        """
        y_sum = 0.0
        ones = np.ones((x_test.shape[0], 1))
        x_test = np.concatenate((x_test, ones), axis=1)

        for i in range(0, x_test.shape[0]):
            y_prime = np.dot(x_test[i], self.w)
            y_sum += (y_test[i][0] - y_prime[0]) ** 2

        return np.sqrt(y_sum/x_test.shape[0])

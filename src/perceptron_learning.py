import numpy as np


class PerceptronLearning:
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

    @staticmethod
    def act_fcn(w, x):
        return np.sign(np.dot(w.T, x))

    def is_converged(self, x, y, w):
        return np.equal(self.act_fcn(w, x), y)

    def solver(self, x, y, w_init, eta):
        """
        Loss function: J = argMin(sum(-y_i * (w * x_i)))
            with:   y_i - i-th data point label
        :param x: data points
        :param eta: learning rate
        :param y: outputs
        :param w_init: initiated coefficient
        :return:
        """
        w = [w_init]
        mis_points = list()  # List of mis-classified points
        i = 0
        while i < 1000:
            mix_id = np.random.permutation(self.no_data_points())

            for i in range(0, self.no_data_points()):
                xi = x[:, mix_id[i]].reshape(self.no_data_point_dim(), 1)
                yi = y[0, mix_id[i]]

                # Point is misclassified
                if self.act_fcn(w[-1], xi)[0] != yi:
                    mis_points.append(mix_id[i])
                    w_new = w[-1] + eta * yi * xi
                    w.append(w_new)
            i += 1
            if self.is_converged(x, y, w[-1]):
                break

        return w, mis_points

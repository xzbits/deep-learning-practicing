import numpy as np
from cvxopt import matrix, solvers


class SupportVectorMachine:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def no_data_points(self):
        """
        Get data points property
        :return: number of data point.
        """
        return self.x.shape[0]

    def no_dim(self):
        """
        Get data points property
        :return: number of dimension of data point
        """
        return self.x.shape[1]

    def solver_svm(self):
        """
        With support vector machine, we have the loss function and its constraints:
        (w, b) = min(1/norm(w)**2)
        subject to:
            -y_i * (w * x_i + b) <= -1
        with:   y_i - i-th label
                x_i - i-th data point
                b - bias
                w - Coefficients

        Due to 1/norm(w)**2 is convex function and the inequality 1 - y_i * (w * x_i + b) <= 0 is linear with respect
        to (w, b), it is feasible to apply CVXOPT package to solve a problem (with data linearly separated)
        Refer https://cvxopt.org/userguide/coneprog.html#quadratic-programming

        :return: Optimized w, b
        """
        # Generate matrix for CVXOPT
        f = np.zeros(self.no_dim() + 1)
        H = np.eye(self.no_dim() + 1)
        H[self.no_dim(), self.no_dim()] = 0
        A = np.c_[self.x, np.ones(self.no_data_points())]
        z_mat = -1 * self.y
        for _ in range(self.no_dim()):
            z_mat = np.c_[z_mat, -1 * self.y]
        A = A * z_mat
        P = matrix(H)
        q = matrix(f.T)
        G = matrix(A)
        h = matrix(-1 * np.ones(self.no_data_points()))

        sol = solvers.qp(P, q, G, h)
        solvers.options['show_progress'] = False
        return np.array(sol["x"])

    def solve_dual_problem(self):
        """
        Applying Lagrangian dual problem into its support vector machine
        lambda = arg(-1/2 lambda.T * K * lambda + lambda), subject to: lambda >= 0

        Refer https://cvxopt.org/userguide/coneprog.html#quadratic-programming
        :return: Optimized w, b
        """
        # Build argument matrix
        V = self.y * self.x
        P = matrix(V.dot(V.T))
        p = matrix(-1 * np.ones((self.no_data_points(), 1)))

        # Build constraints matrix
        G = matrix(-1 * np.eye(self.no_data_points()))
        h = matrix(np.zeros((self.no_data_points(), 1)))
        A = matrix(self.y.T.astype(float))
        b = matrix(np.array([[0]]).astype(float))

        sol = solvers.qp(P, p, G, h, A, b)
        lamp = np.array(sol["x"])

        # Find support points for the vector which have lamda >= 0 (with tolerance 1e-6):
        sup_points = np.where(lamp > 1e-6)[0]

        # Get the support points
        v_sup = V[sup_points, :]
        x_sup = self.x[sup_points, :]
        y_sup = self.y[sup_points, :]
        l_sup = lamp[sup_points]

        # Get coefficients and bias based on support points
        w = v_sup.T.dot(l_sup)
        b = np.mean(y_sup - w.T.dot(x_sup.T))
        return np.append(w, b)

    def solver_soft_svm(self, C):
        """
        With soft support vector machine, we have a loss function and its constraints:
        Loss_func = 0.5 * norm(w) ** 2 + C * slack_var
        Subject to:
            1 - slack_var - label * (w * x + b) <= 0
            AND
            -slack_var <= 0

        with:   slack_var - (n, d)
                C - weighting number (the importance of the slack variables)

        Due to 0.5 * norm(w) ** 2 + C * slack_var is convex function and the inequality 1 - y_i * (w * x_i + b) <= 0 and
        -slack_var <= 0 are linear with respect to (w, b), it is feasible to apply CVXOPT package to solve a problem
        Refer https://cvxopt.org/userguide/coneprog.html#quadratic-programming
        :param C: weighting number (the importance of the slack variables)
        :return: Optimized (w, b, slack_var)
        """
        no_var = self.no_data_points() + self.no_dim() + 1
        P = np.zeros((no_var, no_var))
        for i in range(self.no_dim()):
            P[i, i] = 1
        P = matrix(P)
        q = np.ones((no_var, 1))
        for i in range(self.no_dim() + 1):
            q[i] = 1
        q = matrix(q * C)
        h = matrix(-1 * np.ones((self.no_data_points(), 1)))
        G_dp_1 = self.y * self.x
        G_sv_1 = np.eye(self.no_data_points(), self.no_data_points())
        G_1 = -1.0 * np.c_[G_dp_1, self.y, G_sv_1]
        G_dp_2 = np.zeros((self.no_data_points(), self.no_dim() + 1), dtype=float)
        G_sv_2 = -1.0 * np.eye(self.no_data_points(), self.no_data_points())
        G_2 = np.c_[G_dp_2, G_sv_2]
        G = np.r_[G_1, G_2]

        sol = solvers.qp(P, q, G, h)
        solvers.options['show_progress'] = False
        w_b = np.array(sol["x"])[:self.no_dim() + 2]
        slack_var = np.array(sol["x"])[self.no_dim() + 2:]
        return w_b, slack_var

    def solver_soft_svm_dual_problem(self, C):
        """
        Applying Lagrangian dual problem into its support vector machine
        lambda = arg(-1/2 lambda.T * K * lambda + lambda), subject to: 0 <= lambda <= C

        Refer https://cvxopt.org/userguide/coneprog.html#quadratic-programming
        :return: Optimized w, b
        """
        # Build argument matrix
        V = self.y * self.x
        P = matrix(V.dot(V.T))
        p = matrix(-1 * np.ones((self.no_data_points(), 1)))

        # Build constraints matrix
        G = matrix(np.r_[-1 * np.eye(self.no_data_points()), np.eye(self.no_data_points())])
        h = matrix(np.r_[np.zeros((self.no_data_points(), 1)), C * np.ones((self.no_data_points(), 1))])
        A = matrix(self.y.T.astype(float))
        b = matrix(np.array([[0]]).astype(float))

        sol = solvers.qp(P, p, G, h, A, b)
        lamp = np.array(sol["x"])

        # Find support points for the vector which have 0 <= lamda <= C (with tolerance 1e-6):
        sup_points = np.where((lamp > 1e-6) & (lamp < C * (1 - 1e-6)))[0]

        # Find support points for the vector which have 0 <= lamda (with tolerance 1e-6):
        sup_points_s = np.where(lamp > 1e-6)[0]

        # Get the support points
        v_sup = V[sup_points_s, :]
        x_sup = self.x[sup_points, :]
        y_sup = self.y[sup_points, :]
        l_sup = lamp[sup_points_s]

        # Get coefficients and bias based on support points
        w = v_sup.T.dot(l_sup)
        b = np.mean(y_sup - w.T.dot(x_sup.T))

        return np.append(w, b)

    @staticmethod
    def validate(x_test, y_test, w_b):
        x_test = np.c_[x_test, np.ones(x_test.shape[0])]
        y_predict = np.where(np.dot(x_test, w_b) < 0, -1, 1)
        y_test = y_test.reshape(y_test.shape[0], -1)
        y_predict = y_predict.reshape(y_test.shape[0], -1)
        b = y_test == y_predict
        return np.where(b == True)[0].shape[0] / x_test.shape[0] * 100

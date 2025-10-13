"""
Mathematics for Machine Learning: Practical Implementations
==========================================================

This module provides practical implementations and demonstrations of mathematical
concepts essential for machine learning and deep learning.
"""

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import sympy as sp
import warnings
warnings.filterwarnings('ignore')


class LinearAlgebraTools:
    """Linear algebra utilities for machine learning"""

    @staticmethod
    def vector_operations_demo():
        """Demonstrate basic vector operations"""
        print("=== Vector Operations Demo ===")

        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])

        print(f"v1: {v1}")
        print(f"v2: {v2}")
        print(f"Addition: {v1 + v2}")
        print(f"Scalar multiplication (2*v1): {2 * v1}")
        print(f"Dot product: {np.dot(v1, v2)}")
        print(f"Vector norm ||v1||: {np.linalg.norm(v1):.4f}")
        cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        print(f"Cosine similarity: {cosine_sim:.4f}")

    @staticmethod
    def matrix_operations_demo():
        """Demonstrate matrix operations"""
        print("\n=== Matrix Operations Demo ===")

        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])

        print(f"A:\n{A}")
        print(f"B:\n{B}")
        print(f"A + B:\n{A + B}")
        print(f"A * B (element-wise):\n{A * B}")
        print(f"A @ B (matrix multiplication):\n{A @ B}")
        print(f"A transpose:\n{A.T}")
        print(f"A inverse:\n{np.linalg.inv(A)}")
        print(f"Determinant of A: {np.linalg.det(A):.4f}")

    @staticmethod
    def eigenvalue_analysis():
        """Demonstrate eigenvalue and eigenvector analysis"""
        print("\n=== Eigenvalue Analysis ===")

        # Symmetric matrix for real eigenvalues
        A = np.array([[4, 2], [2, 3]])
        eigenvalues, eigenvectors = np.linalg.eig(A)

        print(f"Matrix A:\n{A}")
        print(f"Eigenvalues: {eigenvalues}")
        print(f"Eigenvectors:\n{eigenvectors}")

        # Verify: A * v = λ * v
        for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            a_v = A @ eigenvec
            lambda_v = eigenval * eigenvec
            print(f"Eigenpair {i+1}: A*v = {a_v}, λ*v = {lambda_v}")
            print(f"Difference: {np.abs(a_v - lambda_v).max():.2e}")

    @staticmethod
    def pca_implementation():
        """Implement Principal Component Analysis from scratch"""
        print("\n=== PCA Implementation ===")

        # Generate sample data
        rng = np.random.default_rng(42)
        x_data = rng.random((100, 3))
        x_data = x_data @ np.array([[2, 0, 0], [0, 1, 0], [0, 0, 0.5]])  # Stretch data

        def pca_from_scratch(x_data, n_components=2):
            # Center the data
            x_centered = x_data - np.mean(x_data, axis=0)

            # Covariance matrix
            cov_matrix = np.cov(x_centered.T)

            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # Sort by eigenvalues
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Select top components
            w_pca = eigenvectors[:, :n_components]

            # Project data
            x_pca = x_centered @ w_pca

            return x_pca, w_pca, eigenvalues[:n_components]

        x_pca, w_pca, explained_variance = pca_from_scratch(x_data, n_components=2)

        print(f"Original data shape: {x_data.shape}")
        print(f"PCA data shape: {x_pca.shape}")
        print(f"Explained variance: {explained_variance}")
        print(f"Principal components:\n{w_pca}")

        # Compare with sklearn
        pca_sklearn = PCA(n_components=2)
        x_pca_sklearn = pca_sklearn.fit_transform(x_data)

        print(f"sklearn explained variance: {pca_sklearn.explained_variance_}")
        print(f"Reconstruction error: {np.linalg.norm(x_pca - x_pca_sklearn):.2e}")

    @staticmethod
    def svd_analysis():
        """Demonstrate Singular Value Decomposition"""
        print("\n=== SVD Analysis ===")

        A = np.array([[1, 2, 3], [4, 5, 6]])
        u_matrix, s_values, v_transpose = np.linalg.svd(A)

        print(f"Original matrix A:\n{A}")
        print(f"U matrix:\n{u_matrix}")
        print(f"Singular values: {s_values}")
        print(f"V transpose:\n{v_transpose}")

        # Reconstruction
        s_matrix = np.zeros_like(A, dtype=float)
        np.fill_diagonal(s_matrix, s_values)
        a_reconstructed = u_matrix @ s_matrix @ v_transpose

        print(f"Reconstructed matrix:\n{a_reconstructed}")
        print(f"Reconstruction error: {np.linalg.norm(A - a_reconstructed):.2e}")

        # Low-rank approximation
        k = 1  # Keep only top singular value
        a_low_rank = u_matrix[:, :k] @ np.diag(s_values[:k]) @ v_transpose[:k, :]
        print(f"Rank-{k} approximation:\n{a_low_rank}")


class CalculusTools:
    """Calculus utilities for machine learning"""

    @staticmethod
    def symbolic_differentiation():
        """Demonstrate symbolic differentiation"""
        print("=== Symbolic Differentiation ===")

        x = sp.Symbol('x')

        # Various functions
        functions = [
            x**2 + 3*x + 1,
            sp.sin(x) * sp.exp(x),
            sp.log(x**2 + 1),
            (x**2 - 1)/(x + 1)
        ]

        for f in functions:
            df_dx = sp.diff(f, x)
            d2f_dx2 = sp.diff(f, x, 2)

            print(f"f(x) = {f}")
            print(f"f'(x) = {df_dx}")
            print(f"f''(x) = {d2f_dx2}")
            print()

    @staticmethod
    def numerical_differentiation():
        """Implement numerical differentiation methods"""
        print("=== Numerical Differentiation ===")

        def forward_difference(f, x, h=1e-5):
            return (f(x + h) - f(x)) / h

        def central_difference(f, x, h=1e-5):
            return (f(x + h) - f(x - h)) / (2 * h)

        def second_derivative(f, x, h=1e-5):
            return (f(x + h) - 2*f(x) + f(x - h)) / h**2

        # Test function
        def f(x):
            return x**3 + 2*x**2 - x + 1

        def f_prime(x):
            return 3*x**2 + 4*x - 1

        def f_double_prime(x):
            return 6*x + 4

        x_test = 2.0

        print("Function: f(x) = x³ + 2x² - x + 1")
        print(f"At x = {x_test}:")
        print(f"Analytical f'(x): {f_prime(x_test):.6f}")
        print(f"Forward difference: {forward_difference(f, x_test):.6f}")
        print(f"Central difference: {central_difference(f, x_test):.6f}")
        print(f"Analytical f''(x): {f_double_prime(x_test):.6f}")
        print(f"Numerical f''(x): {second_derivative(f, x_test):.6f}")

    @staticmethod
    def gradient_descent_2d():
        """2D gradient descent visualization"""
        print("\n=== 2D Gradient Descent ===")

        def f(x, y):
            return x**2 + y**2 + 2*x*y  # Can be written as (x+y)²

        def gradient_f(x, y):
            return np.array([2*x + 2*y, 2*y + 2*x])

        def gradient_descent(start_point, learning_rate=0.1, max_iter=50, tolerance=1e-6):
            x, y = start_point
            path = [(x, y)]

            for _ in range(max_iter):
                grad = gradient_f(x, y)
                x_new = x - learning_rate * grad[0]
                y_new = y - learning_rate * grad[1]

                path.append((x_new, y_new))

                if np.sqrt((x_new - x)**2 + (y_new - y)**2) < tolerance:
                    break

                x, y = x_new, y_new

            return np.array(path)

        # Run gradient descent from different starting points
        start_points = [(5, 5), (-3, 4), (1, -2)]

        for start in start_points:
            path = gradient_descent(start)
            final_point = path[-1]
            final_value = f(final_point[0], final_point[1])

            print(f"Start: {start} -> Final: ({final_point[0]:.4f}, {final_point[1]:.4f})")
            print(f"Function value: {final_value:.6f} (True minimum: 0.0 at (0,0))")

    @staticmethod
    def backpropagation_demo():
        """Simple backpropagation implementation"""
        print("\n=== Backpropagation Demo ===")

        class SimpleNetwork:
            def __init__(self):
                # Initialize weights and biases
                rng = np.random.default_rng(42)
                self.w1 = rng.random((3, 2)) * 0.2 - 0.1  # (hidden_size, input_size)
                self.b1 = np.zeros(3)
                self.w2 = rng.random((1, 3)) * 0.2 - 0.1  # (output_size, hidden_size)
                self.b2 = np.zeros(1)

            def sigmoid(self, x):
                return 1 / (1 + np.exp(-x))

            def sigmoid_derivative(self, x):
                s = self.sigmoid(x)
                return s * (1 - s)

            def forward(self, x):
                # Layer 1
                self.z1 = self.w1 @ x + self.b1
                self.a1 = self.sigmoid(self.z1)

                # Layer 2
                self.z2 = self.w2 @ self.a1 + self.b2
                self.a2 = self.sigmoid(self.z2)

                return self.a2

            def backward(self, x, y, learning_rate=0.1):
                # Forward pass
                output = self.forward(x)

                # Compute loss (MSE)
                loss = 0.5 * (output - y)**2

                # Output layer gradients
                d_loss_da2 = output - y
                da2_dz2 = self.sigmoid_derivative(self.z2)
                dz2_da1 = self.w2.T
                dz2_db2 = 1

                # Backpropagate
                d_loss_dz2 = d_loss_da2 * da2_dz2
                d_loss_da1 = dz2_da1 @ d_loss_dz2
                d_loss_dw2 = np.outer(d_loss_dz2, self.a1)
                d_loss_db2 = d_loss_dz2 * dz2_db2

                # Hidden layer gradients
                da1_dz1 = self.sigmoid_derivative(self.z1)
                dz1_db1 = 1

                d_loss_dz1 = d_loss_da1 * da1_dz1
                d_loss_dw1 = np.outer(d_loss_dz1, x)
                d_loss_db1 = d_loss_dz1 * dz1_db1

                # Update weights
                self.w2 -= learning_rate * d_loss_dw2
                self.b2 -= learning_rate * d_loss_db2
                self.w1 -= learning_rate * d_loss_dw1
                self.b1 -= learning_rate * d_loss_db1

                return loss.item()

        # Training demo
        network = SimpleNetwork()
        x = np.array([1.0, 0.5])
        y = np.array([0.8])

        print("Training simple neural network...")
        for epoch in range(100):
            network.backward(x, y, learning_rate=0.5)
            if epoch % 20 == 0:
                print(".4f")

        final_prediction = network.forward(x)
        print(f"Final prediction: {final_prediction[0]:.4f}, Target: {y[0]}")


class ProbabilityStatisticsTools:
    """Probability and statistics utilities"""

    @staticmethod
    def probability_distributions():
        """Demonstrate common probability distributions"""
        print("=== Probability Distributions ===")

        # Normal distribution
        mu, sigma = 0, 1
        normal = stats.norm(mu, sigma)

        print("Normal Distribution N(0,1):")
        print(f"P(X ≤ 0): {normal.cdf(0):.4f}")
        print(f"P(X > 1): {1 - normal.cdf(1):.4f}")
        print(f"95th percentile: {normal.ppf(0.95):.4f}")

        # Binomial distribution
        n, p = 10, 0.3
        binomial = stats.binom(n, p)

        print(f"\nBinomial Distribution B({n},{p}):")
        print(f"P(X = 3): {binomial.pmf(3):.4f}")
        print(f"P(X ≤ 5): {binomial.cdf(5):.4f}")
        print(f"Expected value: {n*p}")
        print(f"Variance: {n*p*(1-p)}")

        # Poisson distribution
        lambda_param = 3.5
        poisson = stats.poisson(lambda_param)

        print(f"\nPoisson Distribution Pois({lambda_param}):")
        print(f"P(X = 3): {poisson.pmf(3):.4f}")
        print(f"P(X ≤ 5): {poisson.cdf(5):.4f}")

    @staticmethod
    def maximum_likelihood_estimation():
        """Demonstrate MLE for different distributions"""
        print("\n=== Maximum Likelihood Estimation ===")

        # Generate sample data
        rng = np.random.default_rng(42)

        # Normal distribution MLE
        true_mu, true_sigma = 2.0, 1.5
        normal_data = rng.normal(true_mu, true_sigma, 1000)

        mu_hat = np.mean(normal_data)
        sigma_hat = np.std(normal_data, ddof=0)  # Population standard deviation

        print("Normal Distribution MLE:")
        print(f"True parameters: μ={true_mu}, σ={true_sigma}")
        print(f"MLE estimates: μ̂={mu_hat:.4f}, σ̂={sigma_hat:.4f}")

        # Bernoulli distribution MLE
        true_p = 0.7
        bernoulli_data = rng.binomial(1, true_p, 1000)

        p_hat = np.mean(bernoulli_data)

        print("\nBernoulli Distribution MLE:")
        print(f"True parameter: p={true_p}")
        print(f"MLE estimate: p̂={p_hat:.4f}")

    @staticmethod
    def bayesian_inference():
        """Demonstrate Bayesian inference"""
        print("\n=== Bayesian Inference ===")

        def beta_binomial_update(alpha_prior, beta_prior, successes, trials):
            """Bayesian updating for Beta-Binomial model"""
            alpha_posterior = alpha_prior + successes
            beta_posterior = beta_prior + (trials - successes)

            return alpha_posterior, beta_posterior

        # Example: Coin flipping
        alpha_prior, beta_prior = 2, 2  # Prior belief (fair coin)
        trials = 20
        successes = 15  # 15 heads out of 20 flips

        alpha_post, beta_post = beta_binomial_update(alpha_prior, beta_prior, successes, trials)

        # Posterior mean
        posterior_mean = alpha_post / (alpha_post + beta_post)

        print("Bayesian Coin Flipping:")
        print(f"Prior: Beta({alpha_prior}, {beta_prior})")
        print(f"Data: {successes} successes out of {trials} trials")
        print(f"Posterior: Beta({alpha_post}, {beta_post})")
        print(f"Posterior mean: {posterior_mean:.4f}")

        # Credible interval approximation
        lower_bound = 0.5 * (successes + alpha_prior - 1) / (trials + alpha_prior + beta_prior - 2)
        upper_bound = 0.5 * (successes + alpha_prior + 1) / (trials + alpha_prior + beta_prior + 2)

        print(f"Approximate 95% credible interval: [{lower_bound:.4f}, {upper_bound:.4f}]")

    @staticmethod
    def information_theory():
        """Demonstrate information theory concepts"""
        print("\n=== Information Theory ===")

        def entropy(probabilities):
            """Shannon entropy in bits"""
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))

        def kl_divergence(p, q):
            """Kullback-Leibler divergence"""
            return np.sum(p * np.log2((p + 1e-10) / (q + 1e-10)))

        def cross_entropy(y_true, y_pred):
            """Cross-entropy loss"""
            return -np.sum(y_true * np.log2(y_pred + 1e-10))

        # Example distributions
        p = np.array([0.4, 0.6])  # True distribution
        q = np.array([0.5, 0.5])  # Model distribution

        h_p = entropy(p)
        h_q = entropy(q)
        kl_pq = kl_divergence(p, q)

        print("Information Theory Measures:")
        print(f"Distribution P: {p}")
        print(f"Distribution Q: {q}")
        print(f"Entropy H(P): {h_p:.4f} bits")
        print(f"Entropy H(Q): {h_q:.4f} bits")
        print(f"KL(P||Q): {kl_pq:.4f} bits")

        # Cross-entropy example
        y_true = np.array([1, 0, 0])  # One-hot encoded
        y_pred = np.array([0.8, 0.1, 0.1])  # Softmax predictions

        ce_loss = cross_entropy(y_true, y_pred)
        print("\nCross-entropy example:")
        print(f"True labels: {y_true}")
        print(f"Predictions: {y_pred}")
        print(f"Cross-entropy loss: {ce_loss:.4f} bits")


class OptimizationTools:
    """Optimization algorithms for machine learning"""

    @staticmethod
    def convex_optimization():
        """Demonstrate convex optimization concepts"""
        print("=== Convex Optimization ===")

        def f(x):
            return x**2 + 2*x + 1  # Convex quadratic

        def f_prime(x):
            return 2*x + 2

        # Gradient descent for convex function
        def gradient_descent_convex(start_x, learning_rate=0.1, tolerance=1e-6, max_iter=100):
            x = start_x
            history = [x]

            for _ in range(max_iter):
                grad = f_prime(x)
                x_new = x - learning_rate * grad

                if abs(x_new - x) < tolerance:
                    break

                x = x_new
                history.append(x)

            return x, history

        # Test from different starting points
        start_points = [5.0, -3.0, 0.5]

        for start in start_points:
            optimal_x, _ = gradient_descent_convex(start)
            print(f"Start: {start:.1f} -> Optimal: {optimal_x:.6f} (True: -1.0)")
            print(f"Function value: {f(optimal_x):.6f} (True: 0.0)")

    @staticmethod
    def lagrange_multipliers():
        """Demonstrate Lagrange multipliers for constrained optimization"""
        print("\n=== Lagrange Multipliers ===")

        # Problem: Minimize f(x,y) = x² + y² subject to g(x,y) = x + y - 1 = 0

        # Analytical solution using Lagrange:
        # L(x,y,λ) = x² + y² + λ(x + y - 1)
        # ∂L/∂x = 2x + λ = 0
        # ∂L/∂y = 2y + λ = 0
        # ∂L/∂λ = x + y - 1 = 0

        # Solution: x = 0.5, y = 0.5, λ = -1

        x_opt, y_opt, lambda_opt = 0.5, 0.5, -1.0

        print("Constrained optimization example:")
        print("Minimize: f(x,y) = x² + y²")
        print("Subject to: x + y = 1")
        print(f"Optimal solution: x = {x_opt}, y = {y_opt}")
        print(f"Lagrange multiplier: λ = {lambda_opt}")
        print(f"Constraint satisfied: x + y = {x_opt + y_opt}")
        print(f"Function value: {x_opt**2 + y_opt**2}")

    @staticmethod
    def sgd_implementations():
        """Implement and compare different SGD variants"""
        print("\n=== SGD Variants Comparison ===")

        # Generate synthetic classification data
        np.random.seed(42)
        X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)

        # Add bias term
        X = np.column_stack([np.ones(X.shape[0]), X])

        # True parameters (for comparison)
        true_w = np.array([0.0, 1.0, -1.0])

        def sigmoid(z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

        def logistic_loss(w, x_batch, y_batch):
            z = x_batch @ w
            pred = sigmoid(z)
            loss = -np.mean(y_batch * np.log(pred + 1e-10) +
                            (1 - y_batch) * np.log(1 - pred + 1e-10))
            return loss

        def logistic_gradient(w, x_batch, y_batch):
            z = x_batch @ w
            pred = sigmoid(z)
            grad = x_batch.T @ (pred - y_batch) / len(y_batch)
            return grad

        class SGDMomentum:
            def __init__(self, learning_rate=0.01, momentum=0.9):
                self.lr = learning_rate
                self.momentum = momentum
                self.velocity = None

            def step(self, gradient):
                if self.velocity is None:
                    self.velocity = np.zeros_like(gradient)

                self.velocity = self.momentum * self.velocity - self.lr * gradient
                return self.velocity

        class Adam:
            def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
                self.lr = learning_rate
                self.beta1 = beta1
                self.beta2 = beta2
                self.epsilon = epsilon
                self.m = None
                self.v = None
                self.t = 0

            def step(self, gradient):
                self.t += 1

                if self.m is None:
                    self.m = np.zeros_like(gradient)
                    self.v = np.zeros_like(gradient)

                self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
                self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2

                m_hat = self.m / (1 - self.beta1**self.t)
                v_hat = self.v / (1 - self.beta2**self.t)

                return -self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        def train_optimizer(optimizer_class, optimizer_name, **kwargs):
            rng = np.random.default_rng(42)
            w = rng.standard_normal(3) * 0.1
            optimizer = optimizer_class(**kwargs)

            losses = []
            batch_size = 32

            for epoch in range(50):
                # Shuffle data
                indices = rng.permutation(len(X))
                x_shuffled, y_shuffled = X[indices], y[indices]

                epoch_loss = 0
                num_batches = 0

                for i in range(0, len(X), batch_size):
                    x_batch = x_shuffled[i:i+batch_size]
                    y_batch = y_shuffled[i:i+batch_size]

                    grad = logistic_gradient(w, x_batch, y_batch)
                    update = optimizer.step(grad)
                    w += update

                    loss = logistic_loss(w, x_batch, y_batch)
                    epoch_loss += loss
                    num_batches += 1

                avg_loss = epoch_loss / num_batches
                losses.append(avg_loss)

                if epoch % 10 == 0:
                    print(f"{optimizer_name} - Epoch {epoch}, Loss: {avg_loss:.4f}")

            return w, losses[-1]

        # Compare optimizers
        print("Training comparison:")

        w_sgd, loss_sgd = train_optimizer(SGDMomentum, "SGD+Momentum",
                                          learning_rate=0.1, momentum=0.9)
        w_adam, loss_adam = train_optimizer(Adam, "Adam", learning_rate=0.01)

        print("\nFinal results:")
        print(f"True weights: {true_w}")
        print(f"SGD+Momentum weights: {w_sgd}")
        print(f"Adam weights: {w_adam}")
        print(f"SGD+Momentum final loss: {loss_sgd:.4f}")
        print(f"Adam final loss: {loss_adam:.4f}")


class AdvancedMathTools:
    """Advanced mathematical concepts for ML"""

    @staticmethod
    def matrix_calculus():
        """Demonstrate matrix calculus concepts"""
        print("=== Matrix Calculus ===")

        def numerical_jacobian(f, x, h=1e-5):
            """Compute numerical Jacobian matrix"""
            n = len(x)
            m = len(f(x))
            J = np.zeros((m, n))

            for i in range(n):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h

                J[:, i] = (f(x_plus) - f(x_minus)) / (2 * h)

            return J

        # Example function: f: R^2 -> R^3
        def f(x):
            return np.array([
                x[0]**2 + x[1],
                x[0] * x[1],
                np.sin(x[0]) + np.cos(x[1])
            ])

        x = np.array([1.0, 2.0])
        j_numerical = numerical_jacobian(f, x)

        print("Jacobian Matrix Example:")
        print(f"Point: {x}")
        print(f"Numerical Jacobian:\n{j_numerical}")

        # Analytical Jacobian for verification
        j_analytical = np.array([
            [2*x[0], 1],  # ∂f1/∂x0, ∂f1/∂x1
            [x[1], x[0]],  # ∂f2/∂x0, ∂f2/∂x1
            [np.cos(x[0]), -np.sin(x[1])]  # ∂f3/∂x0, ∂f3/∂x1
        ])

        print(f"Analytical Jacobian:\n{j_analytical}")
        print(f"Error: {np.linalg.norm(j_numerical - j_analytical):.2e}")

    @staticmethod
    def hessian_matrix():
        """Compute and analyze Hessian matrices"""
        print("\n=== Hessian Matrix ===")

        def numerical_hessian(f, x, h=1e-5):
            """Compute numerical Hessian matrix"""
            n = len(x)
            H = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    # Mixed partial derivative
                    x_pp = x.copy()
                    x_pp[i] += h
                    x_pp[j] += h
                    x_pm = x.copy()
                    x_pm[i] += h
                    x_pm[j] -= h
                    x_mp = x.copy()
                    x_mp[i] -= h
                    x_mp[j] += h
                    x_mm = x.copy()
                    x_mm[i] -= h
                    x_mm[j] -= h

                    H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)

            return H

        # Example function: f(x,y) = x² + 2y² + 2xy
        def f(xy):
            x, y = xy
            return x**2 + 2*y**2 + 2*x*y

        xy = np.array([1.0, 1.0])
        h_numerical = numerical_hessian(f, xy)

        print("Hessian Matrix Example:")
        print("Function: f(x,y) = x² + 2y² + 2xy")
        print(f"Point: {xy}")
        print(f"Numerical Hessian:\n{h_numerical}")

        # Analytical Hessian
        h_analytical = np.array([
            [2, 2],  # ∂²f/∂x², ∂²f/∂x∂y
            [2, 4]   # ∂²f/∂y∂x, ∂²f/∂y²
        ])

        print(f"Analytical Hessian:\n{h_analytical}")
        print(f"Error: {np.linalg.norm(h_numerical - h_analytical):.2e}")

        # Check if positive definite (convex function)
        eigenvalues = np.linalg.eigvals(h_analytical)
        print(f"Eigenvalues: {eigenvalues}")
        print(f"Positive definite (convex): {np.all(eigenvalues > 0)}")

    @staticmethod
    def fourier_analysis():
        """Demonstrate Fourier analysis for signal processing"""
        print("\n=== Fourier Analysis ===")

        # Generate a complex signal
        t = np.linspace(0, 4*np.pi, 1000)
        rng = np.random.default_rng(42)
        signal = (2*np.sin(2*t) + 1.5*np.sin(5*t) + np.cos(3*t) +
                  0.5*rng.standard_normal(len(t)))  # Add noise

        # Discrete Fourier Transform
        fft_result = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(t), t[1] - t[0])

        # Power spectrum
        power_spectrum = np.abs(fft_result)**2

        # Find dominant frequencies
        positive_freq_mask = frequencies > 0
        freqs_pos = frequencies[positive_freq_mask]
        power_pos = power_spectrum[positive_freq_mask]

        # Get top 3 frequencies
        top_indices = np.argsort(power_pos)[-3:]
        dominant_freqs = freqs_pos[top_indices]
        dominant_powers = power_pos[top_indices]

        print("Fourier Analysis of Signal:")
        print("Signal components: 2*sin(2t) + 1.5*sin(5t) + cos(3t) + noise")
        print(f"Dominant frequencies: {dominant_freqs}")
        print(f"Corresponding powers: {dominant_powers}")

        # Convolution theorem demonstration
        print("\nConvolution Theorem:")

        f = np.array([1, 2, 3, 2, 1])
        g = np.array([1, 1, 1])

        # Direct convolution
        conv_direct = np.convolve(f, g, mode='same')

        # Convolution via FFT
        n = len(f) + len(g) - 1
        f_padded = np.pad(f, (0, n - len(f)), 'constant')
        g_padded = np.pad(g, (0, n - len(g)), 'constant')

        fft_f = np.fft.fft(f_padded)
        fft_g = np.fft.fft(g_padded)
        conv_fft = np.real(np.fft.ifft(fft_f * fft_g))[:len(conv_direct)]

        print(f"Signal f: {f}")
        print(f"Kernel g: {g}")
        print(f"Direct convolution: {conv_direct}")
        print(f"FFT convolution: {conv_fft}")
        print(f"Max error: {np.max(np.abs(conv_direct - conv_fft)):.2e}")


def main():
    """Run all mathematical demonstrations"""
    print("Mathematics for Machine Learning - Practical Demonstrations")
    print("=" * 60)

    # Linear Algebra
    linear_tools = LinearAlgebraTools()
    linear_tools.vector_operations_demo()
    linear_tools.matrix_operations_demo()
    linear_tools.eigenvalue_analysis()
    linear_tools.pca_implementation()
    linear_tools.svd_analysis()

    # Calculus
    calculus_tools = CalculusTools()
    calculus_tools.symbolic_differentiation()
    calculus_tools.numerical_differentiation()
    calculus_tools.gradient_descent_2d()
    calculus_tools.backpropagation_demo()

    # Probability and Statistics
    prob_stats_tools = ProbabilityStatisticsTools()
    prob_stats_tools.probability_distributions()
    prob_stats_tools.maximum_likelihood_estimation()
    prob_stats_tools.bayesian_inference()
    prob_stats_tools.information_theory()

    # Optimization
    opt_tools = OptimizationTools()
    opt_tools.convex_optimization()
    opt_tools.lagrange_multipliers()
    opt_tools.sgd_implementations()

    # Advanced Topics
    advanced_tools = AdvancedMathTools()
    advanced_tools.matrix_calculus()
    advanced_tools.hessian_matrix()
    advanced_tools.fourier_analysis()

    print("\n" + "=" * 60)
    print("All mathematical demonstrations completed!")


if __name__ == "__main__":
    main()

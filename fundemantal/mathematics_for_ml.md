# Mathematics for Machine Learning: From Basics to Advanced

## Overview

This side lesson provides a comprehensive mathematical foundation for machine learning and deep learning. Understanding these concepts is crucial for grasping how algorithms work, optimizing models, and developing new techniques.

## 1. Linear Algebra

### 1.1 Vectors and Matrices

**Vectors** are ordered arrays of numbers representing points in space:

```python
import numpy as np
import torch

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"Vector addition: {v1 + v2}")
print(f"Scalar multiplication: {2 * v1}")
print(f"Dot product: {np.dot(v1, v2)}")
print(f"Vector norm: {np.linalg.norm(v1)}")
```

**Matrices** are 2D arrays used for linear transformations:

```python
# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matrix multiplication:\n{np.dot(A, B)}")
print(f"Matrix transpose:\n{A.T}")
print(f"Matrix inverse:\n{np.linalg.inv(A)}")
print(f"Determinant: {np.linalg.det(A)}")
```

### 1.2 Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors are fundamental to understanding matrix transformations:

```python
# Eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Principal Component Analysis (PCA) concept
def pca_concept(X, n_components=2):
    """Simplified PCA implementation"""
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Covariance matrix
    cov_matrix = np.cov(X_centered.T)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components
    W = eigenvectors[:, :n_components]

    # Project data
    X_pca = X_centered @ W

    return X_pca, W, eigenvalues[:n_components]
```

### 1.3 Singular Value Decomposition (SVD)

SVD decomposes a matrix into three matrices with important applications in ML:

```python
# SVD decomposition
U, s, Vt = np.linalg.svd(A)
print(f"U matrix:\n{U}")
print(f"Singular values: {s}")
print(f"V transpose:\n{Vt}")

# Matrix reconstruction
S = np.zeros_like(A, dtype=float)
np.fill_diagonal(S, s)
A_reconstructed = U @ S @ Vt
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed)}")
```

## 2. Calculus

### 2.1 Derivatives and Gradients

Derivatives measure how functions change with respect to their inputs:

```python
import sympy as sp

# Symbolic differentiation
x = sp.Symbol('x')
f = x**3 + 2*x**2 + x + 1

df_dx = sp.diff(f, x)
d2f_dx2 = sp.diff(f, x, 2)

print(f"Function: {f}")
print(f"First derivative: {df_dx}")
print(f"Second derivative: {d2f_dx2}")

# Numerical differentiation
def numerical_derivative(f, x, h=1e-5):
    """Compute numerical derivative using central difference"""
    return (f(x + h) - f(x - h)) / (2 * h)

def f_example(x):
    return x**3 + 2*x**2 + x + 1

x_val = 2.0
analytical_deriv = 3*x_val**2 + 4*x_val + 1
numerical_deriv = numerical_derivative(f_example, x_val)

print(f"Analytical derivative at x={x_val}: {analytical_deriv}")
print(f"Numerical derivative at x={x_val}: {numerical_deriv}")
print(f"Error: {abs(analytical_deriv - numerical_deriv)}")
```

### 2.2 Partial Derivatives and Gradients

In machine learning, we often deal with functions of multiple variables:

```python
# Partial derivatives
x, y = sp.symbols('x y')
g = x**2 + y**2 + 2*x*y

dg_dx = sp.diff(g, x)
dg_dy = sp.diff(g, y)

print(f"Function: {g}")
print(f"∂g/∂x: {dg_dx}")
print(f"∂g/∂y: {dg_dy}")

# Gradient vector
def gradient_descent_2d(start_point, learning_rate=0.1, iterations=50):
    """Simple gradient descent for 2D function"""
    x, y = start_point

    for i in range(iterations):
        # Compute gradients (analytical)
        grad_x = 2*x + 2*y
        grad_y = 2*y + 2*x

        # Update parameters
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y

        # Compute function value
        f_val = x**2 + y**2 + 2*x*y

        if i % 10 == 0:
            print(".4f")

    return x, y

# Run gradient descent
final_x, final_y = gradient_descent_2d([5.0, 5.0])
print(f"Final point: ({final_x:.4f}, {final_y:.4f})")
```

### 2.3 Chain Rule and Backpropagation

The chain rule is fundamental to backpropagation in neural networks:

```python
# Chain rule demonstration
def chain_rule_demo():
    """Demonstrate chain rule in neural network context"""

    # Simple neural network: y = σ(W*x + b)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)

    # Forward pass
    x = np.array([1.0, 2.0])
    W = np.array([[0.5, 0.3], [0.2, 0.8]])
    b = np.array([0.1, 0.4])

    z = W @ x + b
    y = sigmoid(z)

    print(f"Input: {x}")
    print(f"Hidden layer: {z}")
    print(f"Output: {y}")

    # Backward pass (backpropagation)
    dy_dz = sigmoid_derivative(z)  # dσ/dz

    # For simplicity, assume loss = 0.5 * sum((y - target)^2)
    target = np.array([0.8, 0.3])
    dy = y - target  # dL/dy

    dz = dy * dy_dz  # dL/dz

    # Gradients w.r.t. parameters
    dW = np.outer(dz, x)  # dL/dW
    db = dz  # dL/db

    print(f"Gradients w.r.t. weights:\n{dW}")
    print(f"Gradients w.r.t. biases: {db}")

chain_rule_demo()
```

## 3. Probability and Statistics

### 3.1 Probability Distributions

Understanding probability distributions is crucial for modeling uncertainty:

```python
from scipy import stats
import matplotlib.pyplot as plt

# Normal distribution
mu, sigma = 0, 1
normal_dist = stats.norm(mu, sigma)

x = np.linspace(-3, 3, 100)
pdf_values = normal_dist.pdf(x)
cdf_values = normal_dist.cdf(x)

print(f"Normal distribution - Mean: {mu}, Std: {sigma}")
print(f"P(X ≤ 0): {normal_dist.cdf(0):.4f}")
print(f"P(X > 1): {1 - normal_dist.cdf(1):.4f}")

# Bernoulli distribution (for binary outcomes)
p = 0.3
bernoulli_dist = stats.bernoulli(p)

print(f"Bernoulli distribution - p: {p}")
print(f"P(X = 1): {bernoulli_dist.pmf(1)}")
print(f"P(X = 0): {bernoulli_dist.pmf(0)}")

# Maximum Likelihood Estimation (MLE)
def mle_normal(data):
    """Maximum likelihood estimation for normal distribution"""
    mu_hat = np.mean(data)
    sigma_hat = np.std(data, ddof=0)  # Population standard deviation

    return mu_hat, sigma_hat

# Generate sample data
np.random.seed(42)
sample_data = np.random.normal(2, 1.5, 1000)

mu_mle, sigma_mle = mle_normal(sample_data)
print(f"MLE - True μ: 2.0, Estimated μ: {mu_mle:.4f}")
print(f"MLE - True σ: 1.5, Estimated σ: {sigma_mle:.4f}")
```

### 3.2 Bayesian Statistics

Bayesian methods provide a framework for updating beliefs with data:

```python
def bayesian_updating():
    """Demonstrate Bayesian updating with Beta-Binomial model"""

    # Prior: Beta(α, β)
    alpha_prior, beta_prior = 2, 2  # Prior belief about coin fairness

    # Likelihood: Binomial(n, p)
    n_trials = 10
    n_successes = 7  # Observed 7 heads out of 10 tosses

    # Posterior: Beta(α + successes, β + failures)
    alpha_posterior = alpha_prior + n_successes
    beta_posterior = beta_prior + (n_trials - n_successes)

    # Posterior mean
    posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)

    print(f"Prior: Beta({alpha_prior}, {beta_prior})")
    print(f"Likelihood: Binomial({n_trials}, p) with {n_successes} successes")
    print(f"Posterior: Beta({alpha_posterior}, {beta_posterior})")
    print(f"Posterior mean: {posterior_mean:.4f}")

    # Credible interval (95%)
    from scipy.special import beta as beta_func
    from scipy.optimize import brentq

    def beta_cdf(p, a, b):
        return beta_func(p, a, b) / beta_func(a, b)

    # Find 2.5th and 97.5th percentiles
    lower_bound = brentq(lambda p: beta_cdf(p, alpha_posterior, beta_posterior) - 0.025,
                        0.001, 0.999)
    upper_bound = brentq(lambda p: beta_cdf(p, alpha_posterior, beta_posterior) - 0.975,
                        0.001, 0.999)

    print(f"95% credible interval: [{lower_bound:.4f}, {upper_bound:.4f}]")

bayesian_updating()
```

### 3.3 Information Theory

Information theory provides tools for measuring uncertainty and information:

```python
def entropy_and_kl_divergence():
    """Demonstrate entropy and KL divergence"""

    # Entropy calculation
    def entropy(probabilities):
        """Shannon entropy"""
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    # KL divergence
    def kl_divergence(p, q):
        """Kullback-Leibler divergence"""
        return np.sum(p * np.log2((p + 1e-10) / (q + 1e-10)))

    # Example distributions
    p = np.array([0.4, 0.6])  # True distribution
    q = np.array([0.5, 0.5])  # Approximate distribution

    H_p = entropy(p)
    H_q = entropy(q)
    KL_pq = kl_divergence(p, q)

    print(f"Distribution P: {p}")
    print(f"Distribution Q: {q}")
    print(f"Entropy H(P): {H_p:.4f} bits")
    print(f"Entropy H(Q): {H_q:.4f} bits")
    print(f"KL(P||Q): {KL_pq:.4f} bits")

    # Cross-entropy (used in classification loss)
    def cross_entropy(y_true, y_pred):
        """Cross-entropy loss"""
        return -np.sum(y_true * np.log2(y_pred + 1e-10))

    # Example for binary classification
    y_true = np.array([1, 0])  # True labels (one-hot)
    y_pred = np.array([0.8, 0.2])  # Predicted probabilities

    ce_loss = cross_entropy(y_true, y_pred)
    print(f"Cross-entropy loss: {ce_loss:.4f}")

entropy_and_kl_divergence()
```

## 4. Optimization

### 4.1 Convex Optimization

Many ML problems can be formulated as optimization problems:

```python
def convex_optimization_demo():
    """Demonstrate convex optimization concepts"""

    # Quadratic function: f(x) = x^2 + 2x + 1
    def f(x):
        return x**2 + 2*x + 1

    def f_prime(x):
        return 2*x + 2

    def f_double_prime(x):
        return 2  # Second derivative is constant > 0, so convex

    # Gradient descent for convex function
    def gradient_descent_convex(start_x, learning_rate=0.1, tolerance=1e-6, max_iter=100):
        x = start_x
        history = [x]

        for i in range(max_iter):
            grad = f_prime(x)
            x_new = x - learning_rate * grad

            if abs(x_new - x) < tolerance:
                break

            x = x_new
            history.append(x)

        return x, history

    # Run optimization
    optimal_x, trajectory = gradient_descent_convex(5.0)
    print(f"Optimal solution: x = {optimal_x:.6f}")
    print(f"Function value: f(x) = {f(optimal_x):.6f}")
    print(f"Analytical minimum: x = -1.0, f(x) = 0.0")

    # Lagrange multipliers for constrained optimization
    def lagrange_example():
        """Minimize f(x,y) = x^2 + y^2 subject to g(x,y) = x + y - 1 = 0"""

        # Lagrangian: L(x,y,λ) = x^2 + y^2 + λ(x + y - 1)
        # ∂L/∂x = 2x + λ = 0
        # ∂L/∂y = 2y + λ = 0
        # ∂L/∂λ = x + y - 1 = 0

        # Solution: x = y = 0.5, λ = -1
        x, y, lam = 0.5, 0.5, -1.0

        print(f"Lagrange solution: x = {x}, y = {y}")
        print(f"Constraint satisfied: x + y = {x + y}")
        print(f"Function value: {x**2 + y**2}")

    lagrange_example()

convex_optimization_demo()
```

### 4.2 Stochastic Gradient Descent

SGD and its variants are the workhorses of deep learning optimization:

```python
def sgd_variants_demo():
    """Demonstrate different SGD variants"""

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Linear decision boundary

    # Add bias term
    X = np.column_stack([np.ones(X.shape[0]), X])

    # True parameters
    true_w = np.array([0.0, 1.0, 1.0])

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def logistic_loss(w, X_batch, y_batch):
        z = X_batch @ w
        pred = sigmoid(z)
        return -np.mean(y_batch * np.log(pred + 1e-10) + (1 - y_batch) * np.log(1 - pred + 1e-10))

    def logistic_gradient(w, X_batch, y_batch):
        z = X_batch @ w
        pred = sigmoid(z)
        return X_batch.T @ (pred - y_batch) / len(y_batch)

    # SGD with momentum
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

    # Adam optimizer
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

    # Training loop comparison
    def train_optimizer(optimizer_class, optimizer_name, **kwargs):
        w = np.random.randn(3) * 0.1
        optimizer = optimizer_class(**kwargs)

        losses = []
        batch_size = 32

        for epoch in range(100):
            # Shuffle data
            indices = np.random.permutation(len(X))
            X_shuffled, y_shuffled = X[indices], y[indices]

            epoch_loss = 0
            num_batches = 0

            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                grad = logistic_gradient(w, X_batch, y_batch)
                update = optimizer.step(grad)
                w += update

                loss = logistic_loss(w, X_batch, y_batch)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

            if epoch % 20 == 0:
                print(f"{optimizer_name} - Epoch {epoch}, Loss: {avg_loss:.4f}")

        return w, losses

    # Compare optimizers
    print("Training with different optimizers:")

    w_sgd, losses_sgd = train_optimizer(SGDMomentum, "SGD+Momentum", learning_rate=0.1, momentum=0.9)
    w_adam, losses_adam = train_optimizer(Adam, "Adam", learning_rate=0.01)

    print(f"\nFinal weights comparison:")
    print(f"True weights: {true_w}")
    print(f"SGD+Momentum: {w_sgd}")
    print(f"Adam: {w_adam}")

sgd_variants_demo()
```

## 5. Advanced Topics

### 5.1 Matrix Calculus

Matrix calculus is essential for understanding neural network gradients:

```python
def matrix_calculus_demo():
    """Demonstrate matrix calculus concepts"""

    # Vector-by-vector derivatives
    def jacobian_demo():
        """Compute Jacobian matrix"""

        # Function f: R^n -> R^m
        def f(x):
            return np.array([
                x[0]**2 + x[1],
                x[0] * x[1],
                np.sin(x[0]) + np.cos(x[1])
            ])

        def jacobian(x, h=1e-5):
            """Numerical Jacobian"""
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

        x = np.array([1.0, 2.0])
        J = jacobian(x)

        print(f"Point: {x}")
        print(f"Jacobian matrix:\n{J}")

        # Analytical Jacobian for verification
        # df1/dx0 = 2*x[0], df1/dx1 = 1
        # df2/dx0 = x[1], df2/dx1 = x[0]
        # df3/dx0 = cos(x[0]), df3/dx1 = -sin(x[1])

        J_analytical = np.array([
            [2*x[0], 1],
            [x[1], x[0]],
            [np.cos(x[0]), -np.sin(x[1])]
        ])

        print(f"Analytical Jacobian:\n{J_analytical}")
        print(f"Error: {np.linalg.norm(J - J_analytical)}")

    jacobian_demo()

    # Hessian matrix (second derivatives)
    def hessian_demo():
        """Compute Hessian matrix"""

        # Function f: R^2 -> R
        def f(x, y):
            return x**2 + 2*y**2 + 2*x*y

        def hessian(x, y, h=1e-5):
            """Numerical Hessian"""
            H = np.zeros((2, 2))

            # Second derivatives
            # H[0,0] = d²f/dx²
            f_xx = (f(x+h, y) - 2*f(x, y) + f(x-h, y)) / h**2
            H[0, 0] = f_xx

            # H[1,1] = d²f/dy²
            f_yy = (f(x, y+h) - 2*f(x, y) + f(x, y-h)) / h**2
            H[1, 1] = f_yy

            # H[0,1] = H[1,0] = d²f/dxdy
            f_xy = (f(x+h, y+h) - f(x+h, y-h) - f(x-h, y+h) + f(x-h, y-h)) / (4 * h**2)
            H[0, 1] = f_xy
            H[1, 0] = f_xy

            return H

        x, y = 1.0, 1.0
        H = hessian(x, y)

        print(f"\nHessian at ({x}, {y}):\n{H}")

        # Analytical Hessian
        # d²f/dx² = 2, d²f/dy² = 4, d²f/dxdy = 2
        H_analytical = np.array([[2, 2], [2, 4]])
        print(f"Analytical Hessian:\n{H_analytical}")

    hessian_demo()

matrix_calculus_demo()
```

### 5.2 Fourier Analysis

Fourier analysis is useful for signal processing and some ML applications:

```python
def fourier_analysis_demo():
    """Demonstrate Fourier analysis concepts"""

    # Generate a signal with multiple frequencies
    t = np.linspace(0, 2*np.pi, 1000)
    signal = (np.sin(2*t) + 0.5*np.sin(10*t) + 0.3*np.cos(5*t) +
              0.1*np.random.randn(len(t)))  # Add noise

    # Discrete Fourier Transform
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(t), t[1] - t[0])

    # Power spectrum
    power_spectrum = np.abs(fft_result)**2

    # Find dominant frequencies
    positive_freq_idx = frequencies > 0
    dominant_freqs = frequencies[positive_freq_idx][np.argsort(power_spectrum[positive_freq_idx])[-3:]]

    print(f"Dominant frequencies: {dominant_freqs}")

    # Convolution theorem demonstration
    def convolution_theorem_demo():
        """Demonstrate convolution theorem: conv(f,g) = ifft(fft(f) * fft(g))"""

        # Simple signals
        f = np.array([1, 2, 3, 4, 3, 2, 1])
        g = np.array([1, 1, 1])

        # Direct convolution
        conv_direct = np.convolve(f, g, mode='same')

        # Convolution via FFT
        f_padded = np.pad(f, (0, len(f)), 'constant')
        g_padded = np.pad(g, (0, len(f_padded) - len(g)), 'constant')

        fft_f = np.fft.fft(f_padded)
        fft_g = np.fft.fft(g_padded)
        conv_fft = np.real(np.fft.ifft(fft_f * fft_g))[:len(conv_direct)]

        print(f"Direct convolution: {conv_direct}")
        print(f"FFT convolution: {conv_fft}")
        print(f"Max error: {np.max(np.abs(conv_direct - conv_fft))}")

    convolution_theorem_demo()

fourier_analysis_demo()
```

## Summary

This mathematics lesson covers the essential mathematical foundations for machine learning:

1. **Linear Algebra**: Vectors, matrices, eigenvalues, SVD
2. **Calculus**: Derivatives, gradients, chain rule, backpropagation
3. **Probability & Statistics**: Distributions, MLE, Bayesian inference, information theory
4. **Optimization**: Convex optimization, SGD variants
5. **Advanced Topics**: Matrix calculus, Fourier analysis

Mastering these concepts provides the mathematical intuition needed to understand and develop machine learning algorithms effectively.
# Chapter 2: The Mathematics Behind Convolutional Neural Networks

## 2.1 Convolution Operation in Detail

### Mathematical Foundation
The convolution operation is the core of CNNs. For a 2D image I and kernel K:

**Discrete Convolution:**
$$(I * K)(i,j) = \sum_m \sum_n I(i-m, j-n) \cdot K(m,n)$$

**Continuous Convolution:**
$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau$$

### Convolution with Stride and Padding

**Stride (s):** How much the kernel moves each step
**Padding (p):** Zero-padding added around the input

**Output Size Formula:**

$$
O = \left\lfloor \frac{I - K + 2P}{S} + 1 \right\rfloor
$$

Where:
- O = output size
- I = input size
- K = kernel size
- P = padding
- S = stride

### Example: 5x5 Input, 3x3 Kernel, Stride=1, Padding=1

Input (5x5):
```
1  2  3  4  5
6  7  8  9  10
11 12 13 14 15
16 17 18 19 20
21 22 23 24 25
```

Kernel (3x3):
```
1  0  -1
1  0  -1
1  0  -1
```

Output (5x5 with padding):
```
-6  -6  -6  -6  -6
-6  -6  -6  -6  -6
-6  -6  -6  -6  -6
-6  -6  -6  -6  -6
-6  -6  -6  -6  -6
```

## 2.2 Pooling Operations

### Max Pooling
$$P_{max}(i,j) = \max_{m,n \in R} I(i+m, j+n)$$

### Average Pooling
$$P_{avg}(i,j) = \frac{1}{|R|} \sum_{m,n \in R} I(i+m, j+n)$$

### Global Average Pooling
Reduces entire feature map to single value per channel:
$$GAP_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W F_c(i,j)$$

## 2.3 Activation Functions

### Sigmoid
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- Range: (0, 1)
- Problem: Vanishing gradients for large |x|

### Tanh (Hyperbolic Tangent)
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$
- Range: (-1, 1)
- Zero-centered, but still vanishing gradients

### ReLU (Rectified Linear Unit)
$$f(x) = \max(0, x)$$
- Advantages: No vanishing gradient, computationally efficient
- Problem: "Dying ReLU" (neurons stuck at 0)

### Leaky ReLU
$$f(x) = \max(0.01x, x)$$
- Solves dying ReLU problem

### Parametric ReLU (PReLU)
$$f(x) = \max(\alpha x, x)$$
- α is learned during training

### ELU (Exponential Linear Unit)
$$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}$$

## 2.4 Backpropagation in CNNs

### Chain Rule for Convolution
For a convolutional layer with input X, kernel W, output Y = X * W:

**Forward:** $$Y = X * W$$
**Loss:** $$L = f(Y)$$
**Gradient w.r.t. output:** $$\frac{\partial L}{\partial Y}$$

**Gradient w.r.t. kernel:**
$$\frac{\partial L}{\partial W} = X * \frac{\partial L}{\partial Y}$$

**Gradient w.r.t. input:**
$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} * W^T$$

### Cross-Entropy Loss
For multi-class classification:
$$L = -\sum_{c=1}^C y_c \log(\hat{y}_c)$$

Where:
- C = number of classes
- y_c = true label (one-hot)
- ŷ_c = predicted probability

### Gradient of Cross-Entropy + Softmax
The combined gradient simplifies to:
$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$

Where z_i are the logits (pre-softmax outputs).

## 2.5 Batch Normalization

### Forward Pass
$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i$$
$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

### Backward Pass
BatchNorm has learnable parameters γ and β, and computes gradients for stable training.

### Benefits
1. **Reduces Internal Covariate Shift**
2. **Allows Higher Learning Rates**
3. **Acts as Regularization**
4. **Reduces Gradient Vanishing**

## 2.6 Gradient Descent Optimization

### Stochastic Gradient Descent (SGD)
$$w_{t+1} = w_t - \eta \frac{\partial L}{\partial w_t}$$

### Momentum
$$v_t = \beta v_{t-1} + (1 - \beta) \frac{\partial L}{\partial w_t}$$
$$w_{t+1} = w_t - \eta v_t$$

### Adam (Adaptive Moment Estimation)
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

## 2.7 Summary

- **Convolution:** Feature extraction with parameter sharing
- **Pooling:** Spatial reduction and invariance
- **Activation:** Non-linearity (ReLU preferred)
- **Backprop:** Chain rule through convolutional operations
- **BatchNorm:** Stabilizes and accelerates training
- **Optimization:** Adam often best for CNNs

---

**Next:** Chapter 3 - Building Your First CNN Implementation!

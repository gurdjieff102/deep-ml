import numpy as np

def adam_optimizer(
    f,
    grad,
    x0,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    num_iterations=10
):
    x = x0.copy()

    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, num_iterations + 1):
        # gradient
        g = grad(x)

        # First moment
        m = beta1 * m + (1 - beta1) * g

        # Second moment
        v = beta2 * v + (1 - beta2) * (g ** 2)

        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Parameter update
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x
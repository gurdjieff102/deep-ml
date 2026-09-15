import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    
    m = X.shape[0]

    for _ in range(n_epochs):

        if method == 'batch':
            # Use all samples
            predictions = X @ weights
            gradient = (2 / m) * X.T @ (predictions - y)
            weights -= learning_rate * gradient

        elif method == 'stochastic':
            # Use one sample at a time
            for i in range(m):
                prediction = X[i] @ weights
                error = prediction - y[i]
                gradient = 2 * X[i] * error
                weights -= learning_rate * gradient

        elif method == 'mini_batch':
            # Use consecutive mini-batches
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)

                X_batch = X[start:end]
                y_batch = y[start:end]

                predictions = X_batch @ weights
                gradient = (2 / len(X_batch)) * X_batch.T @ (predictions - y_batch)
                weights -= learning_rate * gradient

        else:
            raise ValueError("method must be 'batch', 'stochastic', or 'mini_batch'")

    return weights
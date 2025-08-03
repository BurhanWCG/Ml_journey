import numpy as np

class GradientDescentOptimizer:

    def __init__(self, learning_rate=0.1, n_iterations=1000, tolerance=1e-6, verbose=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.loss_history = []

    def loss(self, w):
        c = np.array([3, -1, 4])
        return np.sum((w - c) ** 2)

    def gradient(self, w):
        c = np.array([3, -1, 4])
        return 2 * (w - c)

    def optimize(self, initial_w):

        w = initial_w.copy()
        self.loss_history = []
        
        for i in range(self.n_iterations):
            grad = self.gradient(w)
            w_new = w - self.learning_rate * grad
            current_loss = self.loss(w_new)
            self.loss_history.append(current_loss)

            if self.verbose:
                print(f"Iteration {i+1}: w = {w_new}, loss = {current_loss:.6f}")

            # Check for early stopping
            if i > 0 and abs(self.loss_history[-2] - current_loss) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {i+1}")
                break

            w = w_new

        return w


if __name__ == "__main__":
    np.random.seed(42)
    initial_weights = np.random.randn(3)
    print(f"Initial weights: {initial_weights}")

    optimizer = GradientDescentOptimizer(learning_rate=0.1, n_iterations=100, verbose=True)
    final_weights = optimizer.optimize(initial_weights)

    print(f"Final weights after gradient descent: {final_weights}")

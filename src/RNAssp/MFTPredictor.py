import random
import rna
import math
import numpy as np
from BasePredictor import BasePredictor


class MFTPredictor(BasePredictor):
    """Mean Field Theory Predictor class.

    The predictor that acts similar to Hopfield networks. However, it allows continuous numbers as neuron states."""

    def __init__(self, num_epoch=100, alpha=15, beta=15, gamma=20, delta=10, mi=10, ni=10, temperature=1 / 100):
        """Produce MFT Predictor.

        Args:
            num_epoch: Number of epochs in prediction process.
            alpha: Penalty for many active neurons in one matrix row.
            beta: Penalty for many active neurons in one matrix column.
            gamma: Penalty for forming knots.
            delta: Penalty for sharp loops.
            mi: Prize for forming stems.
            ni: Prize for base pairs.
            temperature: Randomness of neurons activation.
        """
        super().__init__()
        self.num_epoch = num_epoch
        self.alpha = alpha  # 6 3 7
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.mi = mi
        self.ni = ni
        self.start_t = temperature

    def predict(self, molecule):
        """Predict molecule's secondary structure.

        Args:
            molecule: Molecule object whose structure is to be predicted.
        """
        if not molecule.dot:
            molecule.dot = '.' * len(molecule.seq)
        self.t = self.start_t
        self.seq = molecule.seq
        self.n = len(molecule.seq)
        self.neurons = np.random.uniform(0, 0, self.n * (self.n - 1) // 2)
        self.w = self.compute_weights()
        for i in range(self.num_epoch):
            self.epoch()
            self.t += self.t / (i + 1)

        dot = molecule.dot
        for k in range(len(self.neurons)):
            x = self.n - 2 - math.floor(math.sqrt(-8 * k + 4 * self.n * (self.n - 1) - 7) / 2.0 - 0.5)
            y = int(k + x + 1 - self.n * (self.n - 1) / 2 + (self.n - x) * ((self.n - x) - 1) / 2)
            if dot[x] == '.' and dot[y] == '.':
                if self.node_weight(x, y) == self.ni:
                    if self.neurons[k] > 0.5:
                        dot = dot[:x] + '(' + dot[x + 1: y] + ')' + dot[y + 1:]
                if self.node_weight(x, y) == self.ni / 2:
                    if self.neurons[k] > 0.7:
                        dot = dot[:x] + '(' + dot[x + 1: y] + ')' + dot[y + 1:]
        return rna.Molecule(molecule.seq, dot)

    def epoch(self):
        """Do one epoch of computation."""
        beta = 1 / self.t
        neurons = list(range(self.n * (self.n - 1) // 2))
        random.shuffle(neurons)
        for k in neurons:
            x = self.n - 2 - math.floor(math.sqrt(-8 * k + 4 * self.n * (self.n - 1) - 7) / 2.0 - 0.5)
            y = int(k + x + 1 - self.n * (self.n - 1) / 2 + (self.n - x) * ((self.n - x) - 1) / 2)
            sum = np.dot(self.neurons, self.w[k])
            sum += self.node_weight(x, y)
            self.neurons[k] = (math.tanh(beta * sum) + 1) / 2

    def train(self, X=None, eta=0.001, limit=10, num_iter=5, log=False):
        """Train predictor on example data.

        Args:
            X: List of Molecule objects that are known examples. (use loaded data if None)
            eta: Learning ratio.
            limit: Maximum number of examples to train.
            num_iter: Number of training iterations.
        """
        if X is None:
            if self.X.shape[0] == 0:
                raise Exception("Too few examples.")
            X = []
            for i in self.X:
                if len(i[0, 0]) < 50:
                    X.append(rna.Molecule(i[0, 0], i[0, 1]))
        X = X[:limit]
        for k in range(num_iter):
            for s in X:
                self.predict(s)
                example = []
                pair = rna.pair_matrix(s)
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        example.append(pair[i, j])
                example = np.array(example)
                for i in range(len(self.neurons)):
                    r, c = self.get_upper_triangular_coordinates(i)
                    for j in range(i + 1, len(self.neurons)):
                        x, y = self.get_upper_triangular_coordinates(j)
                        dif = eta * (math.tanh(np.dot(example, self.w[i])) - math.tanh(np.dot(self.neurons, self.w[i])))
                        if r == x:
                            self.alpha -= dif
                        elif c == y:
                            self.beta -= dif
                        elif r < i < c < j or i < r < j < c:
                            self.gamma -= dif
                        else:
                            self.mi -= dif
                if log:
                    print("Sequence {} trained...".format(s))
        print(self.alpha, self.beta, self.gamma, self.mi)

    def get_upper_triangular_coordinates(self, k):
        """Calculate coordinates of RNA matrix from linear form.

        Args:
            k: Index of object in 1D list.

        Returns:
            x, y: Coordinates in RNA matrix.
        """
        x = self.n - 2 - math.floor(math.sqrt(-8 * k + 4 * self.n * (self.n - 1) - 7) / 2.0 - 0.5)
        y = int(k + x + 1 - self.n * (self.n - 1) / 2 + (self.n - x) * ((self.n - x) - 1) / 2)
        return x, y

    def compute_weights(self):
        """Calculate weights between all neuron pairs.

        Returns:
            weights: Calculated weights.
        """
        weights = []
        for r in range(self.n):
            for c in range(r + 1, self.n):
                w = []
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        w.append(self.weight(r, c, i, j))
                weights.append(w)
        weights = np.array(weights)
        np.fill_diagonal(weights, 0)
        return weights

    def weight(self, r, c, i, j):
        """Calculate weight of the edge between two neurons.

        Args:
            r, c: Coordinates of the first neuron.
            i, j: Coordinates of the second neuron.

        Returns:
            e: Energy between two neurons (which is the edge's weight).
        """
        e = 0
        if r == i:
            e -= self.alpha
        if c == j:
            e -= self.beta
        if r < i < c < j or i < r < j < c:
            e -= self.gamma
        if (r != i or c != j) and (r <= i and c >= j or r >= i and c <= j):
            e += self.mi / (abs(r - i) + abs(c - j))
        return e

    def node_weight(self, r, c):
        """Calculate weight of the neuron.

        Args:
            r, c: Neuron coordinates.

        Return:
            e: Bias energy of the node.
        """
        e = 0
        if self.seq[r] == rna.complementary(self.seq[c]):
            e += self.ni
        elif self.seq[r] + self.seq[c] in ['GU', 'UG']:
            e += self.ni / 4
        if c < r + 4:
            e -= self.delta
        return e

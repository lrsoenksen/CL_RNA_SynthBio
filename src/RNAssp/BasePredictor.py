import numpy as np
import rna


class BasePredictor:
    """Base Predictor abstract class."""

    def __init__(self):
        """Make Base Predictor instance and initiate variables."""
        self.X = np.zeros((0, 2))

    def load_data(self, filename, n_chains=1, capitalize=False, purify=False, repair=False):
        """Load data for training predictions.

        Args:
            filename: Name of the file that contains data in FASTA-like format.
            n_chains: Only structures with this number of chains will be loaded.
            capitalize (bool): Capitalize sequence letters.
            purify (bool): Remove all other characters than dots and brackets from structure notation.
            repair (bool): Try to repair examples that were saved incorrectly.

        Returns:
            X: Loaded data.
        """
        with open(filename) as file:
            data = file.read()
        data = data.split('\n\n')
        result = []
        for line in data:
            chains = line.split(">")
            num_chains = 0
            sequences = []
            dots = []
            for chain in chains:
                if "model:1/" in chain:
                    num_chains += 1
                    splitted = chain.split('\n')
                    dot = splitted[2].replace(
                        '[', '.').replace(']', '.').replace('<', '.').replace('>', '.').replace('{', '.').replace(
                        '}', '.').replace('-', '.') if purify else splitted[2]
                    sequence = splitted[1]
                    if repair and dot.count('(') + 1 == dot.count(')'):  # only for single-stranded?
                        dot = '(' + dot
                        pos = rna.match_parentheses(dot, 0) - 1
                        a = sequence[pos]
                        sequence = rna.complementary(a).lower() + sequence
                    if repair and dot.count('(') != dot.count(')'):
                        continue
                    sequences.append(sequence.upper() if capitalize else sequence)
                    dots.append(dot)
            if num_chains == n_chains and len(sequences) == n_chains:
                result.append(sequences + dots)
        self.X = np.mat(result)
        return self.X

    def train(self, X=None):
        """Train predictor if data is given."""
        if X is None and self.X.shape[0] == 0:
            raise Exception('There is no data to train.')
        else:
            if X is not None:
                self.X = X
            self.train_X()

    def train_X(self):
        """Virtual method which trains the predictor with already saved data."""
        raise Exception("You cannot train a base predictor.")

    def predict(self, seq):
        """Virtual method which predicts structure of RNA sequence.

        Args:
            seq: RNA sequence (or list of Molecule objects depending on the used predictor).
        """
        raise Exception("You cannot predict with a base predictor.")

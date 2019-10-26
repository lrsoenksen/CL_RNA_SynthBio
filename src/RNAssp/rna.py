import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import math


class Molecule:
    """RNA molecule class."""

    def __init__(self, seq, dot=None):
        """Make RNA molecule.

        Args:
            seq: String that contains RNA sequence.
            dot: Bracket notation of the secondary structure.
        """
        if not set(seq) <= set('AGCU'):
            raise Exception("Wrong RNA sequence - only A,G,C,U nucleobases are allowed.")
        if dot:
            if dot.count('(') != dot.count(')'):
                raise Exception("Wrong bracket notation.")
        self.__seq = seq
        self.__dot = None
        self.matrix = None
        if dot:
            self.__dot = dot

    def __repr__(self):
        return "\n{}\n{}\n".format(self.seq, self.dot)

    def __str__(self):
        return "{}\n{}".format(self.seq, self.dot)

    def __eq__(self, other):
        if isinstance(other, Molecule):
            return self.seq == other.seq and self.dot == other.dot
        else:
            return False

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __hash__(self):
        return hash(self.__repr__())

    def show(self):
        """Show molecule's secondary structure in the browser."""
        if self.__dot:
            webbrowser.open(
                "http://nibiru.tbi.univie.ac.at/forna/forna.html?id=url/name&sequence={}&structure={}".format(
                    self.__seq,
                    self.__dot))
        else:
            raise Exception('Structure notation does not exist.')

    @property
    def seq(self):
        """String containing RNA sequence."""
        return self.__seq

    @seq.setter
    def seq(self, seq):
        self.__seq = seq

    @property
    def dot(self):
        """Bracket notation of the secondary structure."""
        return self.__dot

    @dot.setter
    def dot(self, dot):
        self.__dot = dot

    def get_substrings(self, length):
        """Get all substrings having correct structure of given length.

        Args:
            length: Length of returned substrings.

        Returns:
            valid: List of valid substrings.
        """
        if self.dot is None:
            raise Exception("There is no structure given for this molecule.")
        else:
            valid = []
            for i in range(len(self.seq) - length + 1):
                substring = self.seq[i:i + length]
                subdot = self.dot[i:i + length]
                ctr = 0
                for j in range(length):
                    if subdot[j] == '(':
                        ctr += 1
                    if subdot[j] == ')':
                        ctr -= 1
                    if ctr < 0:
                        break
                if ctr == 0:
                    valid.append(Molecule(substring, subdot))
            return valid

    def repair(self):
        """Repair the secondary structure by removing invalid pairs and sharp loops.

        Returns:
            self: Molecule object (repaired in place).
        """
        self.dot = self.dot.replace('()', '..').replace('(.)', '...').replace('(..)', '....').replace('(...)', '.....')
        self.matrix = pair_matrix(self)
        length = len(self.seq)
        for x in range(length):
            for y in range(x, length):
                if self.matrix[x, y] == 1:
                    if not is_pair_allowed(self.seq[x], self.seq[y]):
                        self.dot = self.dot[:x] + '.' + self.dot[x + 1:y] + '.' + self.dot[y + 1:]
        return self

    def evaluate(self):
        """Evaluate the energy of Molecule.

        Returns:
            score: Score based on pairings and hairpin loops.
        """
        self.matrix = pair_matrix(self)
        score = 0
        for x in range(len(self.seq)):
            for y in range(x, len(self.seq)):
                if self.matrix[x, y] == 1:
                    if abs(x - y) < 5:
                        score -= 7
                    if self.seq[x] == complementary(self.seq[y]):
                        score += 2
                    elif self.seq[x] == 'U' and self.seq[y] == 'G' or self.seq[x] == 'G' and self.seq[y] == 'U':
                        score += 1
                    else:
                        score -= 5
        return score


def complementary(a):
    """Get complementary nucleobase.

    Args:
        a: One of four nucleobases ('A', 'G', 'C', 'U').

    Returns:
        b: Complementary base.
    """
    a = a.upper()
    if a == 'A':
        return 'U'
    if a == 'U':
        return 'A'
    if a == 'C':
        return 'G'
    if a == 'G':
        return 'C'
    raise Exception('The given letter is not a valid RNA base.')


def is_pair_allowed(a, b):
    """Check if the nucleobase pair is allowed.

    Args:
        a: First base.
        b: Second base.

    Returns:
        allowed (bool): Information whether the pair is allowed or not.
    """
    if a == complementary(b):
        return True
    if a == 'G' and b == 'U' or a == 'U' and b == 'G':
        return True
    return False


def encode_rna(x):
    """Encode RNA sequence as a list of integers.

    Args:
        x: RNA sequence.

    Returns:
        e: List containing encoded sequence.
    """
    return [0 if y == 'A' else 1 if y == 'U' else 2 if y == 'G' else 3 for y in x]


def match_parentheses(dot, position):
    """Find matching parenthesis in bracket notation.

    Args:
        dot: Bracket notation.
        position: Position where there is an opening parenthesis to match.

    Returns:
        i: Index of matching parenthesis (-1 if nothing was found).
    """
    stack = 0
    for i in range(position + 1, len(dot)):
        if dot[i] == '(':
            stack += 1
        elif dot[i] == ')':
            if stack == 0:
                return i
            else:
                stack -= 1
    return -1


def dot_reverse(dot):
    """Reverse bracket notation.

    Args:
        dot: Bracket notation.

    Return:
        reversed (string): Reversed bracket notation.
    """
    return dot[::-1].replace('(', '/').replace(')', '(').replace('/', ')')


def pair_matrix(m, show=False):
    """Produce pair matrix for the given molecule.

    Args:
        m: Molecule object.
        show (bool): Make a matrix plot of the result.

    Returns:
        p: Pair matrix.
    """
    l = len(m.seq)
    p = np.zeros((l, l))
    dot = m.dot
    for begin in range(l):
        if dot[begin] == '(':
            end = match_parentheses(dot, begin)
            p[begin, end] = p[end, begin] = 1

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.matshow(p, cmap=plt.cm.gray, interpolation='nearest')
        ax.set_xticks(np.arange(l))
        ax.set_yticks(np.arange(l))
        ax.set_xticklabels([i for i in m.seq])
        ax.set_yticklabels([i for i in m.seq])
        plt.show()

    return p


def complementarity_matrix(m, show=False):
    """Produce complementarity matrix for the given molecule.

    Complementary bases (according to Watson-Crick) are assigned 2, and G-U pair are assigned 1.

    Args:
        m: Molecule object.
        show (bool): Make a matrix plot of the result.

    Returns:
        p: Complementarity matrix.
    """
    l = len(m.seq)
    p = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            if m.seq[i] == complementary(m.seq[j]):
                p[i, j] = 2
            if m.seq[i] == 'G' and m.seq[j] == 'U' or m.seq[i] == 'U' and m.seq[j] == 'G':
                p[i, j] = 1

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.matshow(p, interpolation='nearest')
        ax.set_xticks(np.arange(l))
        ax.set_yticks(np.arange(l))
        ax.set_xticklabels([i for i in m.seq])
        ax.set_yticklabels([i for i in m.seq])
        plt.show()

    return p

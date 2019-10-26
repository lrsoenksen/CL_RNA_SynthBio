import random
import rna


class GeneticPredictor:
    """Predictor that uses a simple genetic algorithm."""

    def __init__(self, population_size=5, num_epoch=50):
        """Make predictor.

        Args:
            population_size: Number of units remaining in each population.
            num_epoch: Number of epochs during the prediction.
        """
        self.population_size = population_size
        self.num_epoch = num_epoch

    def predict(self, molecule):
        """Predict the secondary structure.

        Args:
            molecule: Molecule object for prediction.

        Returns:
            population: Last known population.
        """

        if not molecule.dot:
            molecule.dot = '.' * len(molecule.seq)
        population = [molecule] + [self.mutate(molecule) for _ in range(self.population_size - 1)]
        for epoch in range(self.num_epoch):
            new_population = set(population)
            for i in range(self.population_size * 20):
                mutation = self.mutate(population[random.randrange(len(population))])
                new_population.add(mutation.repair())
            population = sorted(list(new_population), key=lambda x: x.evaluate())[-self.population_size:]
        return population

    def mutate(self, molecule):
        """Mutate molecule by inserting or deleting basepairs.

        Args:
            molecule: Molecule that should be mutated.

        Returns:
            mutated: Mutated Molecule.
        """
        m = rna.pair_matrix(molecule)
        seq = molecule.seq
        dot = molecule.dot
        length = len(seq)
        x = random.randrange(length - 5)
        y = random.randrange(x + 5, length)
        if m[x, :].sum() == 0 and m[:, y].sum() == 0:
            dot = dot[:x] + '(' + dot[x + 1: y] + ')' + dot[y + 1:]
        if m[x, y] == 1:
            dot = dot[:x] + '.' + dot[x + 1: y] + '.' + dot[y + 1:]
            dot = self.mutate(rna.Molecule(seq, dot)).dot
        return rna.Molecule(seq, dot)

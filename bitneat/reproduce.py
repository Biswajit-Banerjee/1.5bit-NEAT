import math
import random
from itertools import count

import numpy as np

from neat import DefaultReproduction
from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean

class SoftmaxReproduction(DefaultClassConfig):
    """
    Implements reproduction by sampling from the current population based on
    a softmax probability distribution and applying gene mutations.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('mutation_prob', float, 0.1)])

    def __init__(self, config, reporters, stagnation):
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}
        self.mutation_prob = self.reproduction_config.mutation_prob

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    def reproduce(self, config, species, pop_size, generation):
        """
        Handles creation of genomes for the next generation by sampling from the
        current population based on a softmax probability distribution and applying
        gene mutations.
        """
        # Collect all members from non-stagnated species
        all_members = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if not stagnant:
                all_members.extend(stag_s.members.values())

        # Compute softmax probabilities
        fitnesses = [member.fitness for member in all_members]
        softmax_probs = self.compute_softmax_probs(fitnesses)

        # Create the new population by sampling and mutating
        new_population = {}
        for _ in range(pop_size):
            parent = np.random.choice(all_members, p=softmax_probs)
            gid = next(self.genome_indexer)
            child = config.genome_type(gid)
            child.configure_new(config.genome_config)
            self.mutate_genome(child, config.genome_config)
            new_population[gid] = child
            self.ancestors[gid] = (parent.key,)  # Track the parent

        return new_population

    @staticmethod
    def compute_softmax_probs(fitnesses):
        """Compute softmax probabilities from fitness values."""
        fitnesses = np.array(fitnesses)
        max_fitness = np.max(fitnesses)
        shifted_fitnesses = fitnesses - max_fitness
        exp_fitnesses = np.exp(shifted_fitnesses)
        softmax_probs = exp_fitnesses / np.sum(exp_fitnesses)
        return softmax_probs

    def mutate_genome(self, genome, genome_config):
        """Mutate the genome by applying gene mutations."""
        for cg in genome.connections.values():
            if random.random() < self.mutation_prob:
                cg.mutate(genome_config)

        for ng in genome.nodes.values():
            if random.random() < self.mutation_prob:
                ng.mutate(genome_config)
import os
import neat
import numpy as np
import gymnasium as gym


import numpy as np
import os
from pathlib import Path

from tqdm.auto import tqdm
from datetime import datetime as dt
import multiprocessing as mp
from functools import partial

# local import 
from .utils import write_genome, log

gen = 0
        
def eval_genome(genome, config, scenarios):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []

    for _ in range(scenarios):
        env = gym.make("BipedalWalker-v3", hardcore=False)
        observation, _ = env.reset()
        fitness = 0.0
        max_steps = 1600

        for i in range(max_steps):
            action = net.activate(observation)
            observation, reward, done, truncated, info = env.step(action)
            fitness += reward
            if done or truncated:
                break

        fitnesses.append(fitness)
        env.close()

    return np.mean(fitnesses)

def eval_genomes(genomes, config, storage="models", log_file_path="default_log.csv", scenarios=1):
    global gen
    best_fitness = -1e10
    most_fit = None
    avg_fitness = []
    
    # Get the number of CPU cores
    num_cores = mp.cpu_count()

    # Create a pool with the number of cores minus one
    pool = mp.Pool(processes=num_cores - 1)
    
    results = [pool.apply_async(eval_genome, args=(genome, config, scenarios)) for _, genome in genomes]
    pool.close()  # Close the pool to prevent new tasks from being submitted

    # Create a progress bar
    pbar = tqdm(total=len(genomes), desc=f"Generation {gen:3}", unit="genome")

    for genome_id, (_, genome) in enumerate(genomes):
        fitness = results[genome_id].get()  # Retrieve the result from the worker process
        genome.fitness = fitness

        avg_fitness.append(genome.fitness)

        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            most_fit = genome

        # Update the progress bar
        pbar.set_postfix(best_fitness=f"{best_fitness:.3f}", avg_fitness=f"{np.mean(avg_fitness):.3f}")
        pbar.update(1)

    pbar.close()

    # Wait for all worker processes to finish
    for result in results:
        result.wait()

    os.makedirs(storage, exist_ok=True)
    write_genome(most_fit, os.path.join(storage, f"genome_{gen}.pkl"))
    # most_fit.
    
    log([str(gen), str(best_fitness), str(np.mean(avg_fitness))], path=log_file_path)
    
    gen += 1

    pool.join() 
    
def simulate_generations(config, num_generations=150, scenarios=1, storage="models", log_file_path="default_log.csv"):
    p = neat.Population(config)
    global gen
    gen = 0
    # Run the NEAT algorithm for a smaller number of generations
    log(["Timestamp", "Generation", "Best fitness", "Avg fitness"], init=True, path=log_file_path)
    
    fitness_function = partial(eval_genomes, storage=storage, log_file_path=log_file_path, scenarios=scenarios)
    p.run(fitness_function, num_generations)
    
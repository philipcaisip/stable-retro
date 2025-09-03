"""
Train an agent using NEAT
"""

import argparse

from gymnasium.wrappers import TimeLimit
import retro

import multiprocessing
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import cv2

import neat
import visualize

MAX_EPISODE_STEPS = 10800  # roughly 3 minutes
NUM_CORES = max(1, multiprocessing.cpu_count())

# gym env has to be global to be parallelisable
env = None

def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    steps = 0
    ob, _ = env.reset()
    
    fitness = 0
    
    while True:
        steps += 1

        # format observation for neat nn
        SCALE = 1 / 8
        inx, iny, _ = env.observation_space.shape
        inx = int(SCALE * inx)
        iny = int(SCALE * iny)
        formatted_ob = cv2.resize(ob, (inx, iny))
        formatted_ob = cv2.cvtColor(formatted_ob, cv2.COLOR_BGR2GRAY)
        formatted_ob = formatted_ob.flatten()
        
        net_output = net.activate(formatted_ob)
        
        ob, rew, terminated, truncated, info = env.step(net_output)

        fitness = rew / steps
        
        if terminated or truncated:
            break
    
    return fitness



class SonicGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.discount = None

    def configure_new(self, config):
        super().configure_new(config)
        self.discount = 0.01 + 0.98 * random.random()

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self.discount = random.choice((genome1.discount, genome2.discount))

    def mutate(self, config):
        super().mutate(config)
        self.discount += random.gauss(0.0, 0.05)
        self.discount = max(0.01, min(0.99, self.discount))

    def distance(self, other, config):
        dist = super().distance(other, config)
        disc_diff = abs(self.discount - other.discount)
        return dist + disc_diff

    def __str__(self):
        return f"Reward discount: {self.discount}\n{super().__str__()}"



class PooledErrorCompute(object):
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.test_episodes = []
        self.generation = 0

        # self.min_reward = -200
        # self.max_reward = 200

        self.episode_score = []
        self.episode_length = []

    
    def evaluate_genomes(self, genomes, config):
        self.generation += 1

        t0 = time.time()
        print("Evaluating {0} test episodes".format(len(self.test_episodes)))

        if self.num_workers < 2:
            pass
        else:
            with multiprocessing.Pool(self.num_workers) as pool:
                jobs = []
                for _, genome in genomes:
                    jobs.append(pool.apply_async(evaluate_genome,
                                                  (genome, config)))

                for job, (_, genome) in zip(jobs, genomes):
                    genome.fitness = job.get(timeout=None)
                

        print("final fitness compute time {0}\n".format(time.time() - t0))



def main():
    global env

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="SonicTheHedgehog2-Genesis")
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    env = retro.make(
            game=args.game, 
            state=args.state, 
            render_mode=None, 
            scenario=args.scenario)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    
    print("action space: {0!r}".format(env.action_space))
    print("observation space: {0!r}".format(env.observation_space))
    

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-sonic-config')
    config = neat.Config(SonicGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(25, 900))

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    ec = PooledErrorCompute(NUM_CORES)
    while 1:
        try:
            pe = neat.ParallelEvaluator(NUM_CORES, ec.evaluate_genomes)
            gen_best = pop.run(ec.evaluate_genomes, n=5)

            # print(gen_best)
            
            visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")

            plt.plot(ec.episode_score, 'g-', label='score')
            plt.plot(ec.episode_length, 'b-', label='length')
            plt.grid()
            plt.legend(loc='best')
            plt.savefig("scores.svg")
            plt.close()

            mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
            print("Average mean fitness over last 5 generations: {0}".format(mfs))

            mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
            print("Average min fitness over last 5 generations: {0}".format(mfs))

            # Use the best genomes seen so far as an ensemble-ish control system.
            best_genomes = stats.best_unique_genomes(3)
            best_networks = []
            for g in best_genomes:
                best_networks.append(neat.nn.FeedForwardNetwork.create(g, config))


            if pop.generation >= 100:
                print("Solved.")

                # Save the winners.
                for n, g in enumerate(best_genomes):
                    name = 'winner-{0}'.format(n)
                    with open(name + '.pickle', 'wb') as f:
                        pickle.dump(g, f)

                    visualize.draw_net(config, g, view=False, filename=name + "-net.gv")
                    visualize.draw_net(config, g, view=False, filename=name + "-net-pruned.gv", prune_unused=True)

                break
            
            
        except KeyboardInterrupt:
            print("User break. Attempting to save model.")
            for n, g in enumerate(best_genomes):
                name = 'winner-{0}'.format(n)
                with open(name + '.pickle', 'wb') as f:
                    pickle.dump(g, f)

                visualize.draw_net(config, g, view=False, filename=name + "-net.gv")
                visualize.draw_net(config, g, view=False, filename=name + "-net-pruned.gv", prune_unused=True)
            break

    env.close()


if __name__ == "__main__":
    main()

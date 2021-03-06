import gym
import numpy as np

def relu(x):
    return np.where(x>0,x,0)

def softmax(x):
    x = np.exp(x-np.max(x))
    x[x==0] = 1e-15
    return np.array(x / x.sum())


class NeuralNet:

    def __init__(self, n_units=None, copy_network=None, var=0.02, episodes=50, max_episode_length=200):
        # Testing if we need to copy a network
        if copy_network is None:
            # Saving attributes
            self.n_units = n_units
            # Initializing empty lists to hold matrices
            weights = []
            biases = []

            # Populating the lists
            for i in range(len(n_units) - 1):
                weights.append(np.random.normal(loc=0, scale=1, size=(n_units[i], n_units[i + 1])))
                biases.append(np.zeros(n_units[i + 1]))
            # Creating dictionary of parameters
            self.params = {'weights': weights, 'biases': biases}
        else:
            # Copying over elements
            self.n_units = copy_network.n_units
            self.params = {'weights': np.copy(copy_network.params['weights']),
                           'biases': np.copy(copy_network.params['biases'])}
            # Mutating the weights
            self.params['weights'] = [x + np.random.normal(loc=0, scale=var, size=x.shape) for x in
                                      self.params['weights']]
            self.params['biases'] = [x + np.random.normal(loc=0, scale=var, size=x.shape) for x in
                                     self.params['biases']]

    def act(self, X):
        X = np.concatenate((X, self.recurr))
        # Grabbing weights and biases
        weights = self.params['weights']
        biases = self.params['biases']
        # First propagating inputs
        a = relu(np.matmul(X, weights[0]) + biases[0])
        # Now propagating through every other layer
        for i in range(1, len(weights)):
            a = relu(np.matmul(a, weights[i]) + biases[i])
        # Getting probabilities by using the softmax function
        probs = softmax(a)  # TODO:currently softmaxing all outputs: try softmaxing only the actions(a[:2])
        self.recurr = probs
        return np.argmax(probs[:2])

        # Defining the evaluation method
    def evaluate(self, episodes, max_episode_length, render_env, record):
        # Creating empty list for rewards
        rewards = []
        # First we need to set up our gym environment
        env = gym.make('CartPole-v0')
        # Recording video if we need to
        if record is True:
            env = gym.wrappers.Monitor(env, "recording")
        # Increasing max steps
        env._max_episode_steps = 1e20
        for i_episode in range(episodes):
            observation = env.reset()
            self.recurr = np.zeros(self.n_units[-1])
            for t in range(max_episode_length):
                if render_env is True:
                    env.render()
                action = self.act(np.array(observation))
                #print(recurr)
                observation, _, done, _ = env.step(action) #TODO:env.step takes an array of size 2 but self.act returns 6 but
                if done:
                    rewards.append(t)
                    break
        # Closing our environment
        env.close()
        # Getting our final reward
        if len(rewards) == 0:
            return 0
        else:
            return np.array(rewards).mean()

# Defining our class that handles populations of networks
class GeneticNetworks():
    # Defining our initialization method
    def __init__(self, architecture=(4, 16, 2), recurrent_state=4, population_size=50, generations=500, render_env=True, record=False,
                 mutation_variance=0.02, verbose=False, print_every=1, episodes=10, max_episode_length=200):
        # Creating our list of networks
        self.recurrent_state = recurrent_state
        self.architecture = (architecture[0]+architecture[2]+self.recurrent_state, architecture[1], architecture[2]+self.recurrent_state)
        self.networks = [NeuralNet(self.architecture) for _ in range(population_size)]
        self.population_size = population_size
        self.generations = generations
        self.mutation_variance = mutation_variance
        self.verbose = verbose
        self.print_every = print_every
        self.fitness = []
        self.episodes = episodes
        self.max_episode_length = max_episode_length
        self.render_env = render_env
        self.record = record

    # Defining our fiting method
    def fit(self):
        # Iterating over all generations
        for i in range(self.generations):
            # Doing our evaluations
            rewards = np.array([x.evaluate(self.episodes, self.max_episode_length, self.render_env, self.record) for x in self.networks])
            # Tracking best score per generation
            self.fitness.append(np.max(rewards))
            # Selecting the best network
            best_network = np.argmax(rewards)
            # Creating our child networks
            new_networks = [NeuralNet(copy_network=self.networks[best_network], var=self.mutation_variance,
                                      max_episode_length=self.max_episode_length) for _ in range(self.population_size - 1)]
            # Setting our new networks
            self.networks = [self.networks[best_network]] + new_networks
            # Printing output if necessary
            if self.verbose is True and (i % self.print_every == 0 or i == 0):
                print('Generation:', i + 1, '| Highest Reward:', rewards.max().round(1), '| Average Reward:',
                      rewards.mean().round(1))
            # Returning the best network
            self.best_network = self.networks[best_network]


        self.best_network.evaluate(self.episodes, self.max_episode_length, True, self.record)


# Lets train a population of networks
from time import time
start_time = time()
genetic_pop = GeneticNetworks(architecture=(4,16,2),
                                population_size=128,
                                generations=5,
                                episodes=15,
                                mutation_variance=0.1,
                                max_episode_length=10000,
                                render_env=False,
                                verbose=True)


genetic_pop.fit()
print('Finished in',round(time()-start_time,3),'seconds')






X = np.zeros(5)
X=X+3
print(X)
weights = np.ones(shape=(5,10))
#print(np.dot(X,weights))
a = relu((X @ weights))
print(a)
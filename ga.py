import gym
import numpy as np
import slimevolleygym

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

    def predict(self, X):
        # Grabbing weights and biases
        weights = self.params['weights']
        biases = self.params['biases']
        # First propagating inputs
        a = relu(np.dot(X, weights[0]) + biases[0])
        # Now propagating through every other layer
        for i in range(1, len(weights)):
            a = relu(np.dot(a, weights[i]) + biases[i])
        # Getting probabilities by using the softmax function
        probs = softmax(a)  # TODO:currently softmaxing all outputs: try softmaxing only the actions(a[:2])
        for i in range(len(probs[:3])):
            if probs[i] < 0.0001:
                probs[i] = 0
        return probs[:3], probs

        # Defining the evaluation method

# Defining our class that handles populations of networks
class GeneticNetworks():
    # Defining our initialization method
    def __init__(self, architecture=(12, 16, 3), recurrent_state=4, population_size=128, total_tournaments=50000, render_env=False,
                 mutation_variance=0.02, verbose=False, print_every=1, episodes=10, max_episode_length=200):
        # Creating our list of networks
        self.recurrent_state = recurrent_state
        self.architecture = (architecture[0]+architecture[2]+self.recurrent_state, architecture[1], architecture[2]+self.recurrent_state)
        self.networks = [NeuralNet(self.architecture) for _ in range(population_size)]
        self.population_size = population_size
        self.total_tournaments = total_tournaments
        self.winning_streak = [0] * population_size
        self.mutation_variance = mutation_variance
        self.verbose = verbose
        self.print_every = print_every
        self.fitness = []
        self.episodes = episodes
        self.max_episode_length = max_episode_length
        self.render_env = render_env

    # Defining our fitting method
    def fit(self):
        history = []
        for tournament in range(1, self.total_tournaments+1):

            m, n = np.random.choice(self.population_size, 2, replace=False)

            policy_left = self.networks[m]
            policy_right = self.networks[n]
            # Doing our evaluations
            score, length = evaluate(policy_right, policy_left)

            history.append(length)
            # Tracking best score per generation
            # if score is positive, it means policy_right won.
            if score == 0:  # if the game is tied, add noise to the left agent.
                self.networks[m] = NeuralNet(copy_network=self.networks[m], var=self.mutation_variance,
                                      max_episode_length=self.max_episode_length)
            if score > 0:
                self.networks[m] = NeuralNet(copy_network=self.networks[n], var=self.mutation_variance,
                                      max_episode_length=self.max_episode_length)
                self.winning_streak[m] = self.winning_streak[n]
                self.winning_streak[n] += 1
            if score < 0:
                self.networks[n] = NeuralNet(copy_network=self.networks[m], var=self.mutation_variance,
                                      max_episode_length=self.max_episode_length)
                self.winning_streak[n] = self.winning_streak[m]
                self.winning_streak[m] += 1

            if tournament % 100 == 0:
                record_holder = np.argmax(self.winning_streak)
                record = self.winning_streak[record_holder]
                print("tournament:", tournament,
                      "best_winning_streak:", record,
                      "mean_duration", np.mean(history),
                      "stdev:", np.std(history),
                      )
                history = []
        self.best_network = self.networks[record_holder]
        evaluate(self.best_network,self.best_network,True)


def evaluate(policy_right, policy_left, render_mode=False):
        # Creating empty list for rewards
        rewards = []
        # First we need to set up our gym environment
        env = gym.make('SlimeVolley-v0')
        obs_right = env.reset()
        obs_left = obs_right

        recurr_right = np.zeros(policy_right.n_units[-1])
        recurr_left = recurr_right

        done = False
        total_reward = 0
        t = 0

        while not done:
            input_right = np.concatenate((obs_right, recurr_right))
            input_left = np.concatenate((obs_left, recurr_left))

            action_right, recurr_right = policy_right.predict(input_right)
            action_left, recurr_left = policy_left.predict(input_left)

            obs_right, reward, done, info = env.step(action_right, action_left)
            obs_left = info['otherObs']

            total_reward += reward
            t += 1

            if render_mode:
                env.render()

        return total_reward, t

# Lets train a population of networks
from time import time
start_time = time()
genetic_pop = GeneticNetworks()


genetic_pop.fit()
print('Finished in',round(time()-start_time,3),'seconds')










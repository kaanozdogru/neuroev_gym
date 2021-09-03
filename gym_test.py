import gym
import slimevolleygym
import numpy as np
#env = gym.make('CartPole-v0')
#env2 = gym.make('SlimeVolley-v0')
#observation = env.reset()
#observation += np.zeros(6)
#recurrent = [0,0,0,0,0,0]
'''
list1 = ["a", "b", "c", 'd']
list2 = [1, 2, 3]
print(type(list2))
list3 = list1 + list2
print(list3)
print(type(observation))
recur = np.zeros(4)
print(type(recur))
print(observation.shape)
print(recur.shape)
observation = np.concatenate((observation,recur))
#observation = observation + recurrent
#print(observation)
'''


def relu(x):
    return np.where(x>0,x,0)


def softmax(x):
    x = np.exp(x-np.max(x))
    x[x==0] = 1e-15
    return np.array(x / x.sum())


class NeuralNet:

    def __init__(self, n_units=None, copy_network=None, var=0.02, episodes=50, max_episode_length=200):
        # Testing if we need to copy a network
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

    def act(self, X):
        # Grabbing weights and biases
        weights = self.params['weights']
        biases = self.params['biases']
        # First propagating inputs
        a = relu(np.matmul(X,weights[0]) + biases[0])
        # Now propagating through every other layer
        for i in range(1, len(weights)):
            a = relu(np.matmul(a,weights[i]) + biases[i])
        # Getting probabilities by using the softmax function
        probs = softmax(a) #TODO:currently softmaxing all outputs: try softmaxing only the actions(a[:2])
        print(probs[:2])
        return np.argmax(probs[:2]), probs

        # Defining the evaluation method
    def evaluate(self, recurrent_state, episodes, max_episode_length, render_env=True, record=False):
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
            recurr = np.zeros(self.n_units[-1])
            for t in range(max_episode_length):
                if render_env is True:
                    env.render()
                input = np.concatenate((observation, recurr))
                action, recurr = self.act(np.array(input))
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


recurrent = 4
shape = (4, 16, 2)
architecture = (shape[0]+shape[2]+recurrent, shape[1], shape[2]+recurrent)
net = NeuralNet(architecture)
net.evaluate(recurrent, 50, 200)


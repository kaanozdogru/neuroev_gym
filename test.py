import gym
import slimevolleygym
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
            self.actions = []
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
            if probs[i] < 0.01:
                probs[i] = 0
        self.actions.append(probs[:3])
        return probs[:3], probs

    def evaluate(self, episodes, max_episode_length, render_env, record):
        # Creating empty list for rewards
        rewards = []
        # First we need to set up our gym environment
        env = gym.make('SlimeVolley-v0')
        # Recording video if we need to
        if record is True:
            env = gym.wrappers.Monitor(env, "recording")
        # Increasing max steps
        env._max_episode_steps = 1e20
        for i_episode in range(1):
            observation = env.reset()
            recurr = np.zeros(self.n_units[-1])
            for t in range(10000):
                if render_env is True:
                    env.render()
                input = np.concatenate((observation, recurr))
                action, recurr = self.act(np.array(input))
                print(action)
                print(type(action))
                print(min(action))
                if min(action) != 0:
                    print('not zero')
                observation, _, done, _ = env.step(action)

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
shape = (12, 16, 3)
architecture = (shape[0]+shape[2]+recurrent, shape[1], shape[2]+recurrent)
for i in range(10):
    net = NeuralNet(architecture)
    net.evaluate(1,200,True,False)
prev_actions = np.array(net.actions)
env = gym.make('SlimeVolley-v0')
observation = env.reset()
total_reward = 0
i = 0
print('second env')
actions = np.array([[0.003, 0.006, 0.165], [0.005, 0.853, 0.   ], [0. ,   0.318, 0.052]
,[0., 1., 0.],[0., 1., 0.],[0., 1., 0.],[0., 1., 0.],[0., 1., 0.],[0., 1., 0.],[0., 1., 0.]])
actions = prev_actions
'''
#print(type(actions))
for action in actions:
    i+=1
    #action = np.array([1., 0., 0.])
    print(action)
    print(type(action))
    obs, reward, done, info = env.step(action)
    #print(action)
    #print(type(action[0]))
    total_reward += reward
    env.render()

'''


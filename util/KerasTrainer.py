import numpy as np
import random
import tensorflow as tf
import copy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from collections import deque
from keras import backend as K

mine = np.loadtxt('datasets/Mine.txt')

class DQN_Solver:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # list size of state
        self.action_size = action_size # list size of action
        self.memory = deque(maxlen=1000000) # memory space
        self.gamma = 0.9 # discount rate
        self.epsilon = 1.0 # randomness of choosing random action or the best one
        self.e_decay = 0.9999 # epsilon decay rate
        self.e_min = 0.01 # minimum rate of epsilon
        self.learning_rate = 0.0001 # learning rate of neural network
        self.model = self.build_model() # model
        self.model.summary() # model summary

    # model for neural network
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_shape=(2,2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer=RMSprop(learning_rate=self.learning_rate))
        return model

    # remember state, action, its reward, next state and next possible action. done means boolean for goal
    def remember_memory(self, state, action, reward, next_state, next_movables, done):
        self.memory.append((state, action, reward, next_state, next_movables, done))


    # choosing action depending on epsilon
    def choose_action(self, state, movables):
        if self.epsilon >= random.random():
            # randomly choosing action
            return random.choice(movables)
        else:
            # choosing the best action from model.predict()
            return self.choose_best_action(state, movables)
        
    # choose the best action to maximize reward expectation
    def choose_best_action(self, state, movables):
        best_actions = []
        max_act_value = -100
        for a in movables:
            np_action = np.array([[state, a]])
            act_value = self.model.predict(np_action, verbose=0)
            if act_value > max_act_value:
                best_actions = [a,]
                max_act_value = act_value
            elif act_value == max_act_value:
                best_actions.append(a)
        return random.choice(best_actions)

    # this experience replay is going to train the model from memorized states, actions and rewards
    def replay_experience(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = []
        Y = []
        for i in range(batch_size):
            state, action, reward, next_state, next_movables, done = minibatch[i]
            input_action = [state, action]
            if done:
                target_f = reward
            else:
                next_rewards = []
                for i in next_movables:
                    np_next_s_a = np.array([[next_state, i]])
                    next_rewards.append(self.model.predict(np_next_s_a, verbose=0))
                np_n_r_max = np.amax(np.array(next_rewards))
                target_f = reward + self.gamma * np_n_r_max
            X.append(input_action)
            Y.append(target_f)
        np_X = np.array(X)
        np_Y = np.array([Y]).T
        self.model.fit(np_X, np_Y, epochs=1, verbose=0)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay

    def saveModel(self):
        model_json = self.model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model/model.h5")
        print("Saved model to disk")

class Field(object):
    def __init__(self, mine, start_point, goal_point):
        self.mine = mine
        self.start_point = start_point
        self.goal_point = goal_point
        self.movable_vec = [[1,0],[-1,0],[0,1],[0,-1]]

    def get_actions(self, state):
        movables = []
        if state == self.start_point:
            y = state[0] + 1
            x = state[1]
            a = [[y, x]]
            return a
        else:
            for v in self.movable_vec:
                y = state[0] + v[0]
                x = state[1] + v[1]
                if not(0 < x < len(self.mine) and
                       0 <= y <= len(self.mine) - 1):
                    continue
                movables.append([y,x])
            if len(movables) != 0:
                return movables
            else:
                return None

    def get_val(self, state):
        y, x = state
        if state == self.start_point: return 0, False
        else:
            v = float(self.mine[y][x])
            self.mine[y][x] = 0
            if state == self.goal_point: 
                return v, True
            else: 
                return v, False

mine_field = Field(mine, start_point=[0,0], goal_point=[1023,1023])
state_size = 2
action_size = 2
dql_solver = DQN_Solver(state_size, action_size)

# number of episodes to run training
episodes = 6000

# number of times to sample the combination of state, action and reward
times = 2000000

for e in range(episodes):
    state = [0,0]
    score = 0
    mine_field = Field(mine, start_point=[0,0], goal_point=[1023,1023])
    for time in range(times):
        movables = mine_field.get_actions(state)
        action = dql_solver.choose_action(state, movables)
        reward, done = mine_field.get_val(action)
        score = score + reward - 30
        next_state = action
        next_movables = mine_field.get_actions(next_state)
        dql_solver.remember_memory(state, action, reward, next_state, next_movables, done)
        if done or time == (times - 1):
            print("episode: {}/{}, score: {}, e: {:.2} \t @ {}"
                    .format(e, episodes, score, dql_solver.epsilon, time))
            break
        state = next_state
    # run experience replay after sampling the state, action and reward for defined times
    dql_solver.replay_experience(32)
    K.clear_session()
dql_solver.saveModel()
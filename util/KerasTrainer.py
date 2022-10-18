import numpy as np
import random
import copy
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from collections import deque


mine = np.loadtxt('datasets/Mine.txt')

class DQN_Solver:
    def __init__(self, state_size, action_size):
        self.stuckpunish = 10000
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
        model.add(Dense(128, input_shape=(3,2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer=RMSprop(learning_rate=self.learning_rate))
        return model

    # remember state, action, its reward, next state and next possible action. done means boolean for goal
    def remember_memory(self, state, action, reward, next_state, next_movables, done, stuck, goal_point):
        self.memory.append((state, action, reward, next_state, next_movables, done, stuck, goal_point))


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
            np_action = np.array([[state, a, mine_field.goal_point]])
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
            state, action, reward, next_state, next_movables, done, stuck, goal_point = minibatch[i]
            input_action = [state, action, goal_point]

            if stuck == False:
                if done:
                    target_f = reward
                else:
                    next_rewards = []
                    for i in next_movables:
                        np_next_s_a = np.array([[next_state, i, goal_point]])
                        next_rewards.append(self.model.predict(np_next_s_a, verbose=0))
                    np_n_r_max = np.amax(np.array(next_rewards))
                    target_f = reward + self.gamma * np_n_r_max
                X.append(input_action)
                Y.append(target_f)
        
            if stuck == True:
                X.append(input_action)
                Y.append(self.stuckpunish)

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
        self.oldmine = copy.deepcopy(self.mine)
        self.bonus = 20000
        self.greed = 60
        self.start_point = start_point
        self.goal_point = goal_point
        self.movable_vec = [[1,0],[-1,0],[0,1],[0,-1]]

    def get_actions(self, state):
        movables = []
        if state == self.start_point:
            y = state[0] + 1
            x = state[1]
            a = [[y, x]]
            return a, False
        else:
            for v in self.movable_vec:
                y = state[0] + v[0]
                x = state[1] + v[1]
                if not(0 <= x <= len(self.mine) - 1 and
                       0 <= y <= len(self.mine) - 1 and 
                       self.oldmine[y][x] != 0):
                    continue
                movables.append([y,x])
            if len(movables) != 0:
                return movables, False
            else:
                return None, True

    def get_val(self, state):
        y, x = state
        if state == self.start_point: return 0, False
        else:
            v = float(self.oldmine[y][x])
            self.oldmine[y][x] = 0
            if state == self.goal_point: 
                return v + self.bonus, True
            else: 
                return v - self.greed, False

state_size = 2
action_size = 2
dql_solver = DQN_Solver(state_size, action_size)

# number of episodes to run training
episodes = 30000

# number of times to sample the combination of state, action and reward
times = 250

def get_random_points():
    while True:
        randstart = random.sample(range(len(mine) - 1), 2)
        randend = random.sample(range(len(mine) - 1), 2)
        if randstart != randend:
            return randstart, randend
        else:
            continue

for e in range(episodes):
    score = 0
    done = False
    stuck = False
    randpoints = get_random_points()
    mine_field = Field(mine, randpoints[0], randpoints[1])
    state = mine_field.start_point
    for time in range(times):
        movables, stuck = mine_field.get_actions(state)
        if stuck == False:
            action = dql_solver.choose_action(state, movables)
            reward, done = mine_field.get_val(action)
            score = score + reward
            next_state = action
            next_movables, stuck = mine_field.get_actions(next_state)
        dql_solver.remember_memory(state, action, reward, next_state, next_movables, done, stuck, randpoints[1])
        if done or time == (times - 1) or stuck == True:
            print("episode: {}/{}, score: {}, e: {:.2} \t @ {}, done: {}"
                    .format(e, episodes, score, dql_solver.epsilon, time, done))
            break
        state = next_state
    # run experience replay after sampling the state, action and reward for defined times
    dql_solver.replay_experience(32)
    K.clear_session()
dql_solver.saveModel()
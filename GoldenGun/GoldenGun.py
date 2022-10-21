import numpy as np
from matplotlib import pyplot as plt
import random
import tensorflow as tf
import copy
from keras import models

mine = np.loadtxt('datasets/Mine.txt')

class DQN_Solver:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # list size of state
        self.action_size = action_size # list size of action
        self.model = self.build_model() # model
        self.model.summary() # model summary

    # model for neural network
    def build_model(self):
        json_file = open('model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        mine_model = models.model_from_json(loaded_model_json)
        mine_model.load_weights("model/model.h5")
        return mine_model
    # choose the best action to maximize reward expectation
    def choose_best_action(self, state, movables, goal_point, dangers):
        best_actions = []
        max_act_value = -100
        for a in movables:
            np_action = np.array([[state, a, goal_point, dangers[0], dangers[1], dangers[2], dangers[3]]])
            act_value = self.model.predict(np_action, verbose=0)
            if act_value > max_act_value:
                best_actions = [a,]
                max_act_value = act_value
            elif act_value == max_act_value:
                best_actions.append(a)
        return random.choice(best_actions)

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
        dangers = []
        if state == self.start_point:
            y = state[0] + 1
            x = state[1]
            a = [[y, x]]
            return a, False, [[0,0], [0,0], [0,0], [0,0]]
        else:
            for v in self.movable_vec:
                y = state[0] + v[0]
                x = state[1] + v[1]
                if not(0 <= x <= len(self.mine) - 1 and
                       0 <= y <= len(self.mine) - 1 and 
                       self.oldmine[y][x] != 0):
                    dangers.append([y,x])
                    continue
                movables.append([y,x])
                dangers.append([-1,-1])
            if len(movables) != 0:
                return movables, False, dangers
            else:
                return None, True, None
    
    def display(self, point=None):
        
        if not point is None:
                y, x = point
        else:
                point = ""
        #for line in self.field_data:
                #print ("\t" + "%3s " * len(line) % tuple(line))

    def get_val(self, state):
        y, x = state
        if state == self.start_point: 
            self.oldmine[y][x] = 0
            return 0, False
        else:
            v = float(self.oldmine[y][x])
            self.oldmine[y][x] = 0
            if state == self.goal_point: 
                return v + self.bonus, True
            else: 
                return v - self.greed, False

def get_random_points():
    while True:
        randstart = random.sample(range(len(mine) - 1), 2)
        randend = random.sample(range(len(mine) - 1), 2)
        if randstart != randend:
            return randstart, randend
        else:
            continue

state_size = 2
action_size = 2
dql_solver = DQN_Solver(state_size, action_size)
randpoints = get_random_points()
mine_field = Field(mine, randpoints[0], randpoints[1])
state = mine_field.start_point

score = 0
steps = 0
while True:
    steps += 1
    movables, stuck, dangers = mine_field.get_actions(state)
    if (stuck == False):
        action = dql_solver.choose_best_action(state, movables, randpoints[1], dangers)
        print("current state: {0} -> action: {1} ".format(state, action))
        reward, done = mine_field.get_val(action)
        mine_field.display(state)
        score = score + reward
        state = action
        print("current step: {0} \t score: {1} \t goal: {2} movables: {3} \t val: {4} \t\n".format(steps, score, randpoints[1], movables, reward))
    if done or stuck:
        mine_field.display(action)
        print("goal!")
        break

fig, ax = plt.subplots()
fig, ax2 = plt.subplots()
ax.set_title("Original Mine")
ax2.set_title("AI Path")
ax.imshow(mine, cmap="cividis")
ax2.imshow(mine_field.oldmine, cmap="cividis")
plt.show()
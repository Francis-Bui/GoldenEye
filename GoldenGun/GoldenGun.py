import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras import models

dataset = np.loadtxt('datasets/Mine.txt', delimiter=' ')  # Test empty array

json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
mine_model = models.model_from_json(loaded_model_json)
mine_model.load_weights("model/model.h5")


state = start_point
score = 0
steps = 0
while True:
    steps += 1
    movables = mine_field.get_actions(state)
    action = dql_solver.choose_best_action(state, movables)
    print("current state: {0} -> action: {1} ".format(state, action))
    reward, done = mine_field.get_val(action)
    mine_field.display(state)
    score = score + reward
    state = action
    print("current step: {0} \t score: {1}\n".format(steps, score))
    if done:
        mine_field.display(action)
        print("goal!")
        break

        
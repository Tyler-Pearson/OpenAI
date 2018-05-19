import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

print "hello"

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_req = 50
initial_games = 1000 # 10000

def random_games():
   for episode in range(100):
      env.reset()
      for t in range(goal_steps):
         env.render()
         action = env.action_space.sample()
         observation, reward, done, info = env.step(action)
         if done:
            if t >= score_req:
               print("Game {}: {} steps".format(episode+1, t+1))
            break

random_games()




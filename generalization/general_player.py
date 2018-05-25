import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

print "hi"

LR = 1e-3
env = gym.make('CartPole-v0')
max_steps = 500
env._max_episode_steps = max_steps
env.reset()
score_req = 100
initial_games = 10000

def play_game(goal_steps, display, model):
   score = 0
   game_memory = []
   prev_obs = []
   env.reset()
   for step in range(goal_steps):
      if display:
         env.render()
      if (step == 0) or (model is None):
         action = env.action_space.sample()
      else:
         action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
      new_obs, reward, done, info = env.step(action)
      if len(prev_obs) > 0:
         game_memory.append([new_obs, action])
      prev_obs = new_obs
      score += reward
      if done:
         break
   return score, game_memory

def initial_population(pop_size, goal_steps, min_threshold):
   scores = []
   training_data = []
   accepted_scores = []
   for _ in range(pop_size):
      score, game_memory = play_game(goal_steps, False, None)
      if score >= min_threshold:
         accepted_scores.append(score)
         for data in game_memory:
            if data[1] == 1:
               output = [0,1]
            else:
               output = [1,0]
            training_data.append([data[0], output])
      scores.append(score)
   training_data_save = np.array(training_data)
   np.save("saved.npy", training_data_save)
   return training_data, accepted_scores, scores

data, accepted, scores = initial_population(1000, max_steps, 75)
print("Average score: ", mean(scores))
print("Average accepted: ", mean(accepted))
print("Accepted count: ", Counter(accepted))


















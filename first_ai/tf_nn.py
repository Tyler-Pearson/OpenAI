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
env._max_episode_steps = 500
env.reset()
goal_steps = 500
score_req = 100
initial_games = 100000 # 10000

def random_games():
   for episode in range(100):
      env.reset()
      for t in range(goal_steps):
         env.render()
         action = env.action_space.sample()
         observation, reward, done, info = env.step(action)
         if done:
            if t >= score_req:
               print("Game {} {}: {} steps".format(episode+1, score, t+1))
            break

def initial_population():

   training_data = []
   scores = []
   accepted_scores = []

   for _ in range(initial_games):
      score = 0
      game_memory = []
      prev_observation = [] # to get obs*action, not action*next obs
      for _ in range(goal_steps):
         action = env.action_space.sample()
         observation, reward, done, info = env.step(action)
         if len(prev_observation) > 0:
            game_memory.append([prev_observation, action])
         prev_observation = observation
         score += reward
         if done:
            break
      if score > score_req:
         accepted_scores.append(score)
         for data in game_memory:
            if data[1] == 1:
               output = [0,1]
            elif data[1] == 0:
               output = [1,0]
            training_data.append([data[0], output])
      env.reset()
      scores.append(score)
   training_data_save = np.array(training_data)
   np.save("saved.npy", training_data_save)
   print("Average accepted score:", mean(accepted_scores))
   print("Median accepted score: ", median(accepted_scores))
   print(Counter(accepted_scores))
   return training_data

def neural_network_model(input_size):
   network = input_data(shape=[None, input_size, 1], name='input')
   network = fully_connected(network, 128, activation='relu')
   network = dropout(network, 0.8)
   network = fully_connected(network, 256, activation='relu')
   network = dropout(network, 0.8)
   network = fully_connected(network, 512, activation='relu')
   network = dropout(network, 0.8)
   network = fully_connected(network, 256, activation='relu')
   network = dropout(network, 0.8)
   network = fully_connected(network, 128, activation='relu')
   network = dropout(network, 0.8)
   network = fully_connected(network, 2, activation='softmax') #output layer
   network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
   model = tflearn.DNN(network, tensorboard_dir='log')
   return model

def train_model(training_data, model=False):
   X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
   y = [i[1] for i in training_data]
   if not model:
      model = neural_network_model(input_size = len(X[0]))
   model.fit({'input':X}, {'targets':y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openaistuff')
   return model

training_data = initial_population()
model = train_model(training_data)

#model.save('myModel.model')
#model.load('myModel.model')


#
# Play game with model
#
scores = []
choices = []

for each_game in range(100):
   score = 0
   game_memory = []
   prev_obs = []
   env.reset()
   for _ in range(goal_steps):
      if each_game > 90:
         env.render()
      if len(prev_obs) == 0:
         action = env.action_space.sample()
      else:
         action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
      choices.append(action)
      new_observation, reward, done, info = env.step(action)
      prev_obs = new_observation
      game_memory.append([new_observation, action])
      score += reward
      if done:
         break
   scores.append(score)

print("Average Score: ", sum(scores)/len(scores))
print("Choice 1: {}, Choice 0: {}".format(float(choices.count(1))/float(len(choices)), float(choices.count(0))/float(len(choices))))

#if (sum(scores)/len(scores)) > 450:
#   model.save('myModel.model')


















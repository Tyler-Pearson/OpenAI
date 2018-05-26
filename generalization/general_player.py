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
initial_games = 100000

def play_game(goal_steps, display, model):
   score = 0
   game_memory = []
   prev_obs = []
   env.reset()
   for _ in range(goal_steps):
      if display:
         env.render()
      if (len(prev_obs) == 0) or (model is None):
         action = env.action_space.sample()
      else:
         action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
      new_obs, reward, done, info = env.step(action)
      if len(prev_obs) > 0:
         game_memory.append([prev_obs, action])
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
      if score > min_threshold:
         accepted_scores.append(score)
         for data in game_memory:
            if data[1] == 1:
               output = [0,1]
            elif data[1] == 0:
               output = [1,0]
            training_data.append([data[0], output])
      scores.append(score)
   training_data_save = np.array(training_data)
   np.save("saved.npy", training_data_save)
   return training_data, accepted_scores, scores

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
   network = fully_connected(network, 2, activation='softmax') #output layers
   network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
   model = tflearn.DNN(network, tensorboard_dir='log')
   return model

def train_model(training_data, model=False):
   X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
   y = [i[1] for i in training_data]
   if not model:
      model = neural_network_model(input_size = len(X[0]))
   model.fit({'input':X}, {'targets':y}, n_epoch=4, snapshot_step=max_steps, show_metric=True, run_id='openaistuff')
   return model

def test_model(model):
   test_scores = []
   for i in range(100):
      score, mem = play_game(max_steps, i < 10, model)
      test_scores.append(score)
   print("Average test score: ", mean(test_scores))
   print("Scores: ", Counter(test_scores))

training_data, accepted, train_scores = initial_population(initial_games, max_steps, score_req)
print("Average training score: ", mean(train_scores))
print("Average accepted mean: ", mean(accepted))
print("Accepted count: ", Counter(accepted))

model = train_model(training_data)

test_model(model)
















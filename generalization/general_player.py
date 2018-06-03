import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter


def play_game(env, goal_steps, display, model):
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


def get_pop(env, action_count, pop_size, goal_steps, min_threshold, model):
   scores = []
   training_data = []
   accepted_scores = []
   while len(accepted_scores) < pop_size:
      score, game_memory = play_game(env, goal_steps, False, model)
      if score > min_threshold:
         accepted_scores.append(score)
         for data in game_memory:
            output = np.zeros(action_count)
            output[data[1]] = 1
            training_data.append([data[0], output])
      scores.append(score)
   return training_data, accepted_scores, scores


def neural_network_model(input_size, action_count, LR=1e-3):
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
   network = fully_connected(network, action_count, activation='softmax') #output layers
   network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
   model = tflearn.DNN(network, tensorboard_dir='log')
   return model


def train_model(training_data, action_count, max_steps, model=False):
   X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
   y = [i[1] for i in training_data]
   if not model:
      model = neural_network_model(input_size = len(X[0]), action_count=action_count)
   #n_epoch should be determined dynamically
   model.fit({'input':X}, {'targets':y}, n_epoch=5, snapshot_step=max_steps, show_metric=True, run_id='openaistuff')
   return model


def test_model(env, model, max_steps):
   test_scores = []
   for i in range(100):
      score, mem = play_game(env, max_steps, i < 5, model)
      if (i < 5):
         print("Test {}: {}".format(i+1, score))
      test_scores.append(score)
   print("Average test score: {}".format(mean(test_scores)))
   print("Scores: {}".format(Counter(test_scores)))


def play(game_name, max_steps, score_req):
   env = gym.make(game_name)
   env._max_episode_steps = max_steps
   action_count = env.action_space.n
   pop_size = 40

   training_data, accepted, train_scores = get_pop(env, action_count, pop_size, max_steps, score_req, None)
   print("Average training score: {}".format(mean(train_scores)))
   print("Average accepted mean: {}".format(mean(accepted)))
   print("Accepted count: {}".format(Counter(accepted)))

   model = train_model(training_data, action_count, max_steps)

   test_model(env, model, max_steps)


def demo(game_name):
   max_s = 2000
   env = gym.make(game_name)
   env._max_episode_steps = max_s
   action_count = env.action_space.n
   count = 0
   print("\nDemo-ing {}\n---------\nrandom moves\ndisplay first 10 games".format(game_name))
   for i in range(1000):
      score, mem = play_game(env, max_s, i < 10, None)
      print("Score: {}".format(score))
      if score > -1*max_s:
         count += 1
   print("Wins: {}".format(count))


def main():
#   play('CartPole-v0', 500, 120)
#   play('MountainCar-v0', 2000, -1250)
   demo('Acrobat-v0')


if __name__ == "__main__":
   main()
















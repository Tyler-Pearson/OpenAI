import gym

print "Cart Pole"
env = gym.make('CartPole-v0')
env.reset()
for _ in range(150):
   env.render()
   env.step(env.action_space.sample())

print "Mountain Car"
env = gym.make('MountainCar-v0')
env.reset()
for _ in range(150):
   env.render()
   env.step(env.action_space.sample())

import gym

env = gym.make('Recomender4Students-v0')
for i_episode in range(1000):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.render()
            break
env.close()

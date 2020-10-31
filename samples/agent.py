import json
import sys
from environment import Recommender4StudentsEnv

DEFAULT_NUMBER_OPTIONS = 3

students = []
projects = []
print("Importing data....")
with open("../data/data.json") as dataFile:
    data = json.load(dataFile)
    for student in data['students']:
        students += [student]

    for project in data['projects']:
        projects += [project]

print("Data imported.")

print("Reading arguments....")
numberOptions = DEFAULT_NUMBER_OPTIONS
if len(sys.argv) == 2:
    numberOptions = int(sys.argv[1])
    print("-> Creating " + sys.argv[1] + " options per student.")
else:
    print("-> Missing number of option per student to create, default = " + str(DEFAULT_NUMBER_OPTIONS) + ".")


print("Creating environment....")
env = Recommender4StudentsEnv(students, projects, numberOptions)

print("Starting episodes....")
for i_episode in range(50):
    print("Episode " + str(i_episode) + ":")
    env.reset()
    for t in range(10000):
        # env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        # print(observation)
        # print(info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            # env.render()
            break
    # env.render()
env.close()

import json
import sys
import random
import numpy as np
from progress.bar import Bar
from environment import Recommender4StudentsEnv

DEFAULT_NUMBER_OPTIONS = 3
DEFAULT_AGENT = 1
DEFAULT_IS_TRAINING = 1
DEFAULT_TRAINING_RANGE = 20
DEFAULT_EXECUTION_RANGE = 1
agentName = ["Random", "Q Learning"]

# Q Learning ----------------
# Hyper parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []
# ---------------------------


def _qTableCreator(numberStudents, numberProjects, numberOptions):
    numberActions = numberStudents * numberProjects
    numberStates = 1
    for i in range(numberStudents * numberOptions):
        numberStates += numberProjects * ((i * numberProjects) + 1)
    print("-> QTable of " + str(numberStates) + " possible states (rows) y " + str(numberActions)
          + " actions (columns).")
    return np.zeros([numberStates, numberActions])


def _obtainAction(actionNumber, numberProjects):
    studentNumber = actionNumber / numberProjects
    projectNumber = actionNumber % numberProjects
    return [studentNumber, projectNumber]


def _actionNumber(action, numberProjects):
    studentNumber = action[0]
    projectNumber = action[1]
    return (studentNumber * numberProjects) + projectNumber


def _stateNumber(state, numberOptions, numberProjects):
    position = 0
    i = 0
    for student in range(len(state)):
        for option in range(numberOptions):
            projectNumber = state[student][option]
            position += (projectNumber + 1) * ((i * numberProjects) + 1)
            i += 1
    return position


def _accessQTable(qTable, numberOptions, numberProjects, state, action=(-1, -1)):
    if action[0] == -1 and action[1] == -1:
        return qTable[_stateNumber(state, numberOptions, numberProjects)]
    else:
        return qTable[_stateNumber(state, numberOptions, numberProjects), _actionNumber(action, numberProjects)]


def _editQTable(qTable, numberOptions, numberProjects, state, action, newValue):
    qTable[_stateNumber(state, numberOptions, numberProjects), _actionNumber(action, numberProjects)] = newValue
    return qTable


def _randomExecution(env):
    """"Function to execute random"""

    print("Starting episodes....")

    progressBar = Bar("-> Execution progress:", max=DEFAULT_EXECUTION_RANGE)
    for episode in range(DEFAULT_EXECUTION_RANGE):
        print("Episode " + str(episode) + ":")
        env.reset()

        epoch = 0
        done = False

        while not done:
            epoch += 1
            # env.render()
            observation, reward, done, info = env.step(env.action_space.sample())
            # print(observation)
            if reward != 0:
                print(info)
            if done:
                print("-> Episode finished after " + str(epoch) + " steps.")
                env.render()
                break
        # env.render()
        progressBar.next()

    progressBar.finish()
    print("Execution done.")


def _qLearningExecution(env, qTable, numberOption, numberProjects):
    """"Function to execute Q Learning algorithm"""

    print("Starting episodes....")
    totalSteps, totalEpisodes, studentTotalScore, projectTotalScore, skillsTotalScore = 0, 0, 0., 0., 0.

    progressBar = Bar("-> Execution progress:", max=DEFAULT_EXECUTION_RANGE)
    for _ in range(DEFAULT_EXECUTION_RANGE):
        state = env.reset()
        steps, reward = 0, 0

        done = False

        while not done:
            actionPosition = np.argmax(_accessQTable(qTable, numberOption, numberProjects, state))
            action = _obtainAction(actionPosition, numberProjects)
            # action = np.argmax(qTable[state])
            state, reward, done, info = env.step(action)

            steps += 1

        totalEpisodes += 1
        totalSteps += steps
        studentScore, projectScore, skillsScore = env.stepScores()
        studentTotalScore += studentScore
        projectTotalScore += projectScore
        skillsTotalScore += skillsScore
        progressBar.next()

    progressBar.finish()
    print("Execution done.")
    print("-> Results after " + str(totalEpisodes) + " episodes:")
    print("-> Average steps per episode: " + str(totalSteps / totalEpisodes) + " / 1.")
    print("-> Average final students score: " + str(studentTotalScore / totalEpisodes) + " / 1.")
    print("-> Average final projects score: " + str(projectTotalScore / totalEpisodes) + " / 1.")
    print("-> Average final skills score: " + str(skillsTotalScore / totalEpisodes) + " / 1.")
    # print("Average penalties per episode: " + str(totalPenalties / episodes))


def _qLearningTraining(env, qTable, numberOptions, numberProjects):
    """"Function to train Q Learning algorithm"""

    print("Starting training....")
    progressBar = Bar("-> Training progress:", max=DEFAULT_TRAINING_RANGE)
    for episode in range(0, DEFAULT_TRAINING_RANGE):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            # print(state)
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                actionPosition = np.argmax(_accessQTable(qTable, numberOptions, numberProjects, state))
                action = _obtainAction(actionPosition, numberProjects)
                # action = np.argmax(qTable[state])
            nextState, reward, done, info = env.step(action)

            oldValue = _accessQTable(qTable, numberOptions, numberProjects, state, action)
            # oldValue = qTable[state, action]

            nextMax = np.max(_accessQTable(qTable, numberOptions, numberProjects, nextState))
            # nextMax = np.max(qTable[nextState])

            newValue = (1 - alpha) * oldValue + alpha * (reward + gamma * nextMax)

            qTable = _editQTable(qTable, numberOptions, numberProjects, state, action, newValue)
            # qTable[state, action] = newValue

            state = nextState
            epochs += 1

        progressBar.next()

    progressBar.finish()
    print("Training finished.")
    return qTable


def main():
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
    print("-> " + str(len(students)) + " students and " + str(len(projects)) + " projects imported.")

    print("Reading arguments....")
    numberOptions = DEFAULT_NUMBER_OPTIONS
    typeAgent = DEFAULT_AGENT
    isTraining = DEFAULT_IS_TRAINING
    if len(sys.argv) > 1:
        numberOptions = int(sys.argv[1])
        print("-> Creating " + sys.argv[1] + " options per student.")
        if len(sys.argv) > 2:
            typeAgent = int(sys.argv[2])
            if len(sys.argv) == 4:
                isTraining = int(sys.argv[3])
            else:
                print("-> Missing if it has to train, default = " + str(isTraining) + ".")
        else:
            print("-> Missing if it has to train, default = " + str(isTraining) + ".")
            print("-> Missing type of agent, default = " + agentName[typeAgent] + ".")
    else:
        print("-> Missing if it has to train, default = " + str(isTraining) + ".")
        print("-> Missing type of agent, default = " + agentName[typeAgent] + ".")
        print("-> Missing number of option per student to create, default = " + str(DEFAULT_NUMBER_OPTIONS) + ".")

    print("Creating environment....")
    env = Recommender4StudentsEnv(students, projects, numberOptions)
    print("Environment created.")

    print("Starting execution with agent type " + agentName[typeAgent] + "....")
    if typeAgent == 1:
        qTable = _qTableCreator(len(students), len(projects), numberOptions)

        if isTraining:
            qTable = _qLearningTraining(env, qTable, numberOptions, len(projects))
            _qLearningExecution(env, qTable, numberOptions, len(projects))
    else:
        _randomExecution(env)

    env.close()
    print("Environment closed.")


if __name__ == '__main__':
    main()

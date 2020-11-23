import json
import sys
import random
import numpy as np
import pandas as pd
from progress.bar import Bar
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, PPO2
from environment import Recommender4StudentsEnv

DEFAULT_MODE = 1
DEFAULT_NUMBER_OPTIONS = 3
DEFAULT_AGENT = 2
DEFAULT_IS_TRAINING = 1
DEFAULT_TRAINING_RANGE = 10000
# DEFAULT_TRAINING_RANGE_MULTIPLIER = 100
DEFAULT_EXECUTION_RANGE = 1
DEFAULT_IMPORTING_DATA = 0
agentName = ["Random", "Q Learning", "A2C", "PPO2"]

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
    """"Function to create qTable"""

    numberActions = numberStudents * numberProjects
    numberStates = (numberProjects + 1) ** (numberStudents * numberOptions)
    # for i in range(numberStudents * numberOptions):
    #    numberStates += numberProjects * ((i * numberProjects) + 1)
    print("-> QTable of " + str(numberStates) + " possible states (rows) y " + str(numberActions)
          + " actions (columns).")
    return np.zeros([numberStates, numberActions])


def _obtainAction(actionNumber, numberProjects):
    """"Function to obtain action in qTable"""

    studentNumber = actionNumber / numberProjects
    projectNumber = actionNumber % numberProjects
    return [studentNumber, projectNumber]


def _actionNumber(action, numberProjects):
    """"Function to obtain action position in qTable"""

    studentNumber = action[0]
    projectNumber = action[1]
    return (studentNumber * numberProjects) + projectNumber


def _stateNumber(state, numberOptions, numberProjects):
    """"Function to obtain position of state in qTable"""

    position = 0
    i = 0
    for student in range(len(state)):
        for option in range(numberOptions):
            projectNumber = state[student][option]
            position += (projectNumber + 1) * ((numberProjects + 1) ** i)
            i += 1
    return position


def _accessQTable(qTable, numberOptions, numberProjects, state, action=(-1, -1)):
    """"Function to access qTable"""

    if action[0] == -1 and action[1] == -1:
        return qTable[_stateNumber(state, numberOptions, numberProjects)]
    else:
        return qTable[_stateNumber(state, numberOptions, numberProjects), _actionNumber(action, numberProjects)]


def _editQTable(qTable, numberOptions, numberProjects, state, action, newValue):
    """"Function to edit qTable"""

    qTable[_stateNumber(state, numberOptions, numberProjects), _actionNumber(action, numberProjects)] = newValue
    return qTable


def _randomExecution(env):
    """"Function to execute random"""

    print("Starting execution....")
    totalSteps, totalEpisodes, studentTotalScore, projectTotalScore, skillsTotalScore = 0, 0, 0., 0., 0.

    bestResult = []
    bestStudentScore = 0.0
    bestStudentAssigned = 0

    progressBar = Bar("-> Execution progress:", max=DEFAULT_EXECUTION_RANGE)
    for _ in range(DEFAULT_EXECUTION_RANGE):
        studentAssigned = 0
        # print("Episode " + str(episode) + ":")
        state = env.reset()
        steps, reward = 0, 0
        done = False
        while not done:
            # env.render()
            state, reward, done, info = env.step(env.action_space.sample())
            if reward != 0:
                studentAssigned += 1
            # print(observation)

            steps += 1

        totalEpisodes += 1
        totalSteps += steps
        studentScore, projectScore, skillsScore = env.stepScores()
        studentTotalScore += studentScore
        projectTotalScore += projectScore
        skillsTotalScore += skillsScore
        if studentAssigned >= bestStudentAssigned and studentScore > bestStudentScore:
            bestStudentAssigned = studentAssigned
            bestStudentScore = studentScore
            bestResult = env.finalState()
        progressBar.next()

    progressBar.finish()
    print("Execution done.")
    print("-> Results after " + str(totalEpisodes) + " episodes:")
    print("   -> Total student assigned " + str(bestStudentAssigned) + " episodes:")
    print("   -> Average steps per episode: " + str(totalSteps / totalEpisodes) + ".")
    print("   -> Average final students score: " + str(studentTotalScore / totalEpisodes) + " / 1.")
    print("   -> Average final projects score: " + str(projectTotalScore / totalEpisodes) + " / 1.")
    print("   -> Average final skills score: " + str(skillsTotalScore / totalEpisodes) + " / 1.")
    # print("Average penalties per episode: " + str(totalPenalties / episodes))
    return bestResult


def _qLearningExecution(env, qTable, numberOption, numberProjects):
    """"Function to execute Q Learning algorithm"""

    print("Starting execution....")
    totalSteps, totalEpisodes, studentTotalScore, projectTotalScore, skillsTotalScore = 0, 0, 0., 0., 0.

    bestResult = []
    bestStudentScore = 0.0
    bestStudentAssigned = 0
    progressBar = Bar("-> Execution progress:", max=DEFAULT_EXECUTION_RANGE)
    for _ in range(DEFAULT_EXECUTION_RANGE):
        studentAssigned = 0
        state = env.reset()
        steps, reward = 0, 0
        done = False
        while not done:
            actionPosition = np.argmax(_accessQTable(qTable, numberOption, numberProjects, state))
            action = _obtainAction(actionPosition, numberProjects)
            # action = np.argmax(qTable[state])
            state, reward, done, info = env.step(action)
            if reward != 0:
                studentAssigned += 1

            steps += 1

        totalEpisodes += 1
        totalSteps += steps
        studentScore, projectScore, skillsScore = env.stepScores()
        studentTotalScore += studentScore
        projectTotalScore += projectScore
        skillsTotalScore += skillsScore
        if studentAssigned >= bestStudentAssigned and studentScore > bestStudentScore:
            bestStudentAssigned = studentAssigned
            bestStudentScore = studentScore
            bestResult = env.finalState()
        progressBar.next()

    progressBar.finish()
    print("Execution done.")
    print("-> Results after " + str(totalEpisodes) + " episodes:")
    print("   -> Total student assigned " + str(bestStudentAssigned) + " episodes:")
    print("   -> Average steps per episode: " + str(totalSteps / totalEpisodes) + ".")
    print("   -> Average final students score: " + str(studentTotalScore / totalEpisodes) + " / 1.")
    print("   -> Average final projects score: " + str(projectTotalScore / totalEpisodes) + " / 1.")
    print("   -> Average final skills score: " + str(skillsTotalScore / totalEpisodes) + " / 1.")
    # print("Average penalties per episode: " + str(totalPenalties / episodes))
    return bestResult


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


def _stableBaselineTrainingAndExecution(env, typeAgent):
    """"Function to execute Baseline algorithms"""

    if typeAgent == 2:
        model = A2C(MlpPolicy, env, verbose=1)
    else:
        model = PPO2(MlpPolicy, env, verbose=1)

    print("Training model....")
    model.learn(total_timesteps=DEFAULT_TRAINING_RANGE)
    print("Model trained.")

    print("Starting episodes....")
    totalSteps, totalEpisodes, studentTotalScore, projectTotalScore, skillsTotalScore = 0, 0, 0., 0., 0.

    bestResult = []
    bestStudentScore = 0.0
    bestStudentAssigned = 0
    progressBar = Bar("-> Execution progress:", max=DEFAULT_EXECUTION_RANGE)
    for i in range(DEFAULT_EXECUTION_RANGE):
        studentAssigned = 0
        state = env.reset()
        steps, reward = 0, 0
        done = False
        while not done:
            action, _state = model.predict(state)
            state, reward, done, info = env.step(action)
            # env.render()
            steps += 1
            if reward != 0:
                studentAssigned += 1

        totalEpisodes += 1
        totalSteps += steps
        studentScore, projectScore, skillsScore = env.stepScores()
        studentTotalScore += studentScore
        projectTotalScore += projectScore
        skillsTotalScore += skillsScore
        if studentAssigned >= bestStudentAssigned and studentScore > bestStudentScore:
            bestStudentAssigned = studentAssigned
            bestStudentScore = studentScore
            bestResult = env.finalState()
        progressBar.next()

    progressBar.finish()
    print("Execution done.")
    print("-> Results after " + str(totalEpisodes) + " episodes:")
    print("   -> Total student assigned " + str(bestStudentAssigned) + ".")
    print("   -> Average steps per episode: " + str(totalSteps / totalEpisodes) + ".")
    print("   -> Average final students score: " + str(studentTotalScore / totalEpisodes) + " / 1.")
    print("   -> Average final projects score: " + str(projectTotalScore / totalEpisodes) + " / 1.")
    print("   -> Average final skills score: " + str(skillsTotalScore / totalEpisodes) + " / 1.")
    # print("Average penalties per episode: " + str(totalPenalties / episodes))
    return bestResult


def _finalAnalysis(finalState, studentsSelections):
    """"Function to analyze how many students are correct"""

    studentsWithOption = 0
    studentsWithCorrectOption = [0 for _ in range(len(studentsSelections[0]["options"]))]
    for student in studentsSelections:
        projectId = finalState[student["studentId"]][0]
        for option in range(len(student["options"])):
            if projectId == student["options"][option]:
                studentsWithCorrectOption[option] += 1
                studentsWithOption += 1
                break

    print("Analysis of final result:")
    for option in range(len(studentsWithCorrectOption)):
        print(" -> Number of students with option number " + str(option) +
              " => " + str(studentsWithCorrectOption[option]) + " / " + str(len(finalState)) +
              " (" + str(((studentsWithCorrectOption[option] * 1.) / (len(finalState) * 1.)) * 100) + "%)")
    print(" -> Number of unsatisfied students => " + str(len(studentsSelections) - studentsWithOption) +
          " / " + str(len(studentsSelections)) +
          " (" + str((((len(studentsSelections) - studentsWithOption) * 1.) / (len(studentsSelections) * 1.)) * 100) +
          "%)")


def _generateOptionsData(finalState, students, numberOptions, mode):
    """"Function to generate dictionary with the final results"""

    results = []
    if mode:
        for studentId in range(len(students)):
            studentAverageMark = students[studentId]['averageMark']
            optionsForStudent = []
            for option in range(numberOptions):
                projectId = finalState[studentId][option]
                optionsForStudent.append(projectId)

            results.append({"studentId": studentId,
                            "studentAverageMark": studentAverageMark,
                            "projectOptions": optionsForStudent})
    else:
        for studentId in range(len(students)):
            results.append({"studentId": studentId,
                            "project": finalState[studentId][0]})

    return results


def _assignStudentsWithSelections(numberStudents, studentsSelections, projects):
    """"Function to assign students already assigned"""

    print("Analyzing selection of students and assigning projects...")
    studentsAssignations = [[-1] for _ in range(numberStudents)]
    projectAssignations = [0 for _ in range(len(projects))]
    studentsAssigned = 0
    for student in studentsSelections:
        studentId = student['studentId']
        for projectId in student['options']:
            if projectAssignations[projectId] < projects[projectId]["nParticipants"]:
                studentsAssignations[studentId][0] = projectId
                projectAssignations[projectId] += 1
                studentsAssigned += 1
                break

    print("Assignment done. " + str(studentsAssigned) + " students assigned.")
    return studentsAssignations, studentsAssigned


def main():
    try:
        print("Reading arguments....")
        mode = DEFAULT_MODE
        numberOptions = DEFAULT_NUMBER_OPTIONS
        typeAgent = DEFAULT_AGENT
        isTraining = DEFAULT_IS_TRAINING
        importingData = DEFAULT_IMPORTING_DATA
        if len(sys.argv) > 1:
            mode = int(sys.argv[1])
            print("-> Mode of execution " + sys.argv[1] + ".")
            if len(sys.argv) > 2:
                numberOptions = int(sys.argv[2])
                print("-> Creating " + sys.argv[2] + " options per student.")
                if len(sys.argv) > 3:
                    typeAgent = int(sys.argv[3])
                    print("-> Agent type = " + sys.argv[2] + ".")
                    if len(sys.argv) > 4:
                        isTraining = int(sys.argv[4])
                        print("-> Training = " + sys.argv[2] + ".")
                        if len(sys.argv) == 6:
                            importingData = int(sys.argv[5])
                        else:
                            print("-> Missing if you want to import data, default = " + str(importingData) + ".")
                    else:
                        print("-> Missing if it has to train, default = " + str(isTraining) + ".")
                        print("-> Missing if you want to import data, default = " + str(importingData) + ".")
                else:
                    print("-> Missing if it has to train, default = " + str(isTraining) + ".")
                    print("-> Missing type of agent, default = " + agentName[typeAgent] + ".")
                    print("-> Missing if you want to import data, default = " + str(importingData) + ".")
            else:
                print("-> Missing if it has to train, default = " + str(isTraining) + ".")
                print("-> Missing type of agent, default = " + agentName[typeAgent] + ".")
                print("-> Missing number of option per student to create, default = " + str(DEFAULT_NUMBER_OPTIONS)
                      + ".")
                print("-> Missing if you want to import data, default = " + str(importingData) + ".")
        else:
            print("-> Missing what mode to use, default = " + str(mode) + ".")
            print("-> Missing if it has to train, default = " + str(isTraining) + ".")
            print("-> Missing type of agent, default = " + agentName[typeAgent] + ".")
            print("-> Missing number of option per student to create, default = " + str(DEFAULT_NUMBER_OPTIONS) + ".")
            print("-> Missing if you want to import data, default = " + str(importingData) + ".")

        students = []
        projects = []
        print("Importing data....")
        with open("../data/studentsProjectsData.json") as dataFile:
            data = json.load(dataFile)
            for student in data['students']:
                students += [student]

            for project in data['projects']:
                projects += [project]

        studentsSelections = []
        if mode == 1:
            with open("../data/studentsSelectionData.json") as dataFile:
                data = json.load(dataFile)
                for student in data['results']:
                    studentsSelections += [{"options": student['optionSelected'],
                                            "studentId": student['studentId']}]

                print("Data imported.")
                print("-> " + str(len(students)) + " students and " + str(len(projects)) + " projects imported.")
                studentsAlreadyAssigned, studentsAssigned = _assignStudentsWithSelections(len(students),
                                                                                          studentsSelections, projects)

        else:
            print("Data imported.")
            print("-> " + str(len(students)) + " students and " + str(len(projects)) + " projects imported.")

        if mode == 0 or studentsAssigned < len(students):
            print("Creating environment....")
            if typeAgent == 0 or typeAgent == 1:
                env = Recommender4StudentsEnv(students, projects, numberOptions, False, mode, studentsAlreadyAssigned)
            else:
                env = Recommender4StudentsEnv(students, projects, numberOptions, True, mode, studentsAlreadyAssigned)
            print("Environment created.")

            print("Starting execution with agent type " + agentName[typeAgent] + "....")
            if typeAgent == 0:
                bestResult = _randomExecution(env)

            elif typeAgent == 1:
                if importingData:
                    qTable = pd.read_csv('../data/qTable.csv', sep=',', header=None)
                else:
                    qTable = _qTableCreator(len(students), len(projects), numberOptions)

                if isTraining:
                    qTable = _qLearningTraining(env, qTable, numberOptions, len(projects))

                bestResult = _qLearningExecution(env, qTable, numberOptions, len(projects))

                print("Exporting QTable....")
                pd.DataFrame(data=qTable.astype(float)).to_csv('../data/qTableData.csv', sep=',', header=False,
                                                               index=False)
                print("QTable exported.")

            else:
                bestResult = _stableBaselineTrainingAndExecution(env, typeAgent)

            env.close()
            print("Environment closed.")
        else:
            bestResult = studentsAlreadyAssigned

        if mode == 0:
            print("Exporting JSON of options....")
            with open("../data/optionsData.json", "w") as file:
                json.dump({"numberOptions": numberOptions,
                           "results": _generateOptionsData(bestResult, students, numberOptions, True)}, file, indent=4)
            print("JSON of options exported.")
        else:
            _finalAnalysis(bestResult, studentsSelections)
            print("Exporting JSON with final options for students....")
            with open("../data/finalResults.json", "w") as file:
                json.dump({"results": _generateOptionsData(bestResult, students, numberOptions, False)}, file, indent=4)
            print("JSON of final options for students exported.")

    except OSError as err:
        print("Error opening file: {0}".format(err))


if __name__ == '__main__':
    main()

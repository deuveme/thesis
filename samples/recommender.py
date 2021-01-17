import json
import sys
import random
import numpy as np
import pandas as pd
import statistics as st
from time import time
from progress.bar import Bar
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C, PPO2
from samples.environment.recommender4StudentsEnv import Recommender4StudentsEnv

DEFAULT_MODE = 0
DEFAULT_NUMBER_OPTIONS = 3
DEFAULT_AGENT = 2
DEFAULT_IS_TRAINING = 1
DEFAULT_TRAINING_RANGE = 10000
DEFAULT_EXECUTION_RANGE = 1
DEFAULT_IMPORTING_DATA = 0
agentName = ["Random", "Q Learning", "A2C", "PPO2"]

# Q Learning ----------------
# Hyper parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1
# ---------------------------


def _qTableCreator(numberStudents, numberProjects, numberOptions):
    """"Function to create qTable"""

    numberActions = numberStudents * numberProjects
    numberStates = (numberProjects + 1) ** (numberStudents * numberOptions)
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
        env.reset()
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

            nextState, reward, done, info = env.step(action)
            oldValue = _accessQTable(qTable, numberOptions, numberProjects, state, action)
            nextMax = np.max(_accessQTable(qTable, numberOptions, numberProjects, nextState))
            newValue = (1 - alpha) * oldValue + alpha * (reward + gamma * nextMax)
            qTable = _editQTable(qTable, numberOptions, numberProjects, state, action, newValue)

            state = nextState
            epochs += 1

        progressBar.next()

    progressBar.finish()
    print("Training finished.")
    return qTable


def _stableBaselineTrainingAndExecution(env, typeAgent, numberOptions, mode):
    """"Function to execute Baseline algorithms"""

    if typeAgent == 2:
        model = A2C(MlpPolicy, env, verbose=1)
    else:
        model = PPO2(MlpPolicy, env, verbose=1)

    print("Training model....")
    startTime = time()
    model.learn(total_timesteps=DEFAULT_TRAINING_RANGE)
    trainingTime = time() - startTime
    print("Model trained in " + str(trainingTime) + ".")

    print("Starting episodes....")
    totalSteps, numberEpisodes, studentTotalScore, projectTotalScore, skillsTotalScore = 0, 0, 0., 0., 0.
    bestResult = []
    bestStudentScore = 0.0
    bestStudentAssigned = 0
    sumStudentAssigned = 0.0

    allStudentsAssigned = []
    allProjectAssignations = []
    allSteps = []
    allResults = []
    allAverageStudentScore = []
    allAverageProjectScore = []
    allAverageSkillsScore = []
    allStudentScores = []
    allProjectScores = []
    progressBar = Bar("-> Execution progress:", max=DEFAULT_EXECUTION_RANGE)
    for i in range(DEFAULT_EXECUTION_RANGE):
        state = env.reset(1)
        steps, reward = 0, 0
        done = False
        print("Execution " + str(i))
        while not done:
            action, _state = model.predict(state)
            state, reward, done, info = env.step(action)
            # env.render()
            steps += 1

        numberEpisodes += 1
        allSteps.append(steps)

        averageStudentScore, averageProjectScore, averageSkillsScore, studentScores, projectScores, studentsAssigned, projectAssignations = env.stepScores()
        allResults.append(env.finalState())
        allAverageStudentScore.append(averageStudentScore)
        allAverageProjectScore.append(averageProjectScore)
        allAverageSkillsScore.append(averageSkillsScore)
        allStudentScores.append(studentScores)
        allProjectScores.append(projectScores)
        allStudentsAssigned.append(studentsAssigned)
        allProjectAssignations.append(projectAssignations)
        averageStudentAssigned = sum(studentsAssigned) / numberOptions
        sumStudentAssigned += sum(studentsAssigned) / numberOptions

        if averageStudentAssigned >= bestStudentAssigned and averageStudentScore > bestStudentScore:
            bestStudentAssigned = averageStudentAssigned
            bestStudentScore = averageStudentScore
            bestResult = env.finalState()

        progressBar.next()

    progressBar.finish()

    print("Execution done.")
    print(trainingTime)
    if mode == 0:
        _executionAnalysis(numberEpisodes, allStudentScores, allProjectScores, allSteps, bestStudentAssigned,
                           numberOptions, allStudentsAssigned, allProjectAssignations, sumStudentAssigned)

    return bestResult


def _executionAnalysis(numberEpisodes, allStudentScores, allProjectScores, allSteps, bestStudentAssigned, numberOptions,
                       allStudentsAssigned, projectAssignations, sumStudentAssigned):
    """"Function to print the analysis of the execution"""

    averageMaxStudentScore = [0.0 for _ in range(numberEpisodes)]
    averageMaxProjectScore = [0.0 for _ in range(numberEpisodes)]
    averageMaxSkillsScore = [0.0 for _ in range(numberEpisodes)]

    allAverageStudentScore = [0.0 for _ in range(numberEpisodes)]
    allAverageProjectScore = [0.0 for _ in range(numberEpisodes)]
    allAverageSkillsScore = [0.0 for _ in range(numberEpisodes)]

    totalSteps = sum(allSteps)

    for episode in range(len(allStudentScores)):
        maxStudentScore = []
        maxProjectScore = []
        maxSkillsScore = []

        averageStudentScore = []
        averageProjectScore = []
        averageSkillsScore = []

        print("   Episode number " + str(episode) + ":")
        print("   Students:")
        for studentNumber in range(len(allStudentScores[episode])):
            averageStudentScore.append(st.mean(allStudentScores[episode][studentNumber]))

            maxStudentScore.append(max(allStudentScores[episode][studentNumber]))

            print("    -> " + str(studentNumber) + " (mean: "
                  + str(round(st.mean(allStudentScores[episode][studentNumber]) * 100, 2))
                  + "%, max: " + str(round(max(allStudentScores[episode][studentNumber]) * 100, 2)) + "%):")
            for option in range(len(allStudentScores[episode][studentNumber])):
                if allStudentScores[episode][studentNumber][option] != -1:
                    print("        Score Option " + str(option) + ": "
                          + str(round((allStudentScores[episode][studentNumber][option] * 100), 2)) + "%.")
                else:
                    print("        Score Option " + str(option) + ": not assigned.")

        print("   Projects:")
        for projectNumber in range(len(allProjectScores[episode])):
            averageProjectScore.append(st.mean(allProjectScores[episode][projectNumber][0]))
            averageSkillsScore.append(st.mean(allProjectScores[episode][projectNumber][1]))
            maxProjectScore.append(max(allProjectScores[episode][projectNumber][0]))
            maxSkillsScore.append(max(allProjectScores[episode][projectNumber][1]))

            print("    -> " + str(projectNumber) + ":")
            for option in range(len(allProjectScores[episode][projectNumber])):
                print("        Option " + str(option) + " (" + str(projectAssignations[episode][projectNumber][option])
                      + " students assigned):")
                if projectAssignations[episode][projectNumber][option] != 0:
                    print("         -> Score: "
                          + str(round((allProjectScores[episode][projectNumber][0][option] * 100), 2)) + "%.")
                    print("         -> Skills: "
                          + str(round((allProjectScores[episode][projectNumber][1][option] * 100), 2)) + "%.")

        print("   Total student assigned (" + str(sum(allStudentsAssigned[episode]) / numberOptions) + "): ")
        for option in range(len(allStudentsAssigned[episode])):
            print("      -> Option " + str(option) + ": " + str(allStudentsAssigned[episode][option]) +
                  " (" + str(round((allStudentsAssigned[episode][option] / len(allStudentScores[episode])) * 100, 2)) +
                  "%).")
        print("   Steps: " + str(round(allSteps[episode], 2)) + ".")
        print("   Average final students score: " + str(round(st.mean(averageStudentScore) * 100, 2)) + "%.")
        print("   Average final projects score: " + str(round(st.mean(averageProjectScore) * 100, 2)) + "%.")
        print("   Average final skills score: " + str(round(st.mean(averageSkillsScore) * 100, 2)) + "%.")
        print("   Average max students score: " + str(round(st.mean(maxStudentScore) * 100, 2)) + "%.")
        print("   Average max projects score: " + str(round(st.mean(maxProjectScore) * 100, 2)) + "%.")
        print("   Average max skills score: " + str(round(st.mean(maxSkillsScore) * 100, 2)) + "%.")
        print("")

        averageMaxStudentScore[episode] = st.mean(maxStudentScore)
        averageMaxProjectScore[episode] = st.mean(maxProjectScore)
        averageMaxSkillsScore[episode] = st.mean(maxSkillsScore)

        allAverageStudentScore[episode] = st.mean(averageStudentScore)
        allAverageProjectScore[episode] = st.mean(averageProjectScore)
        allAverageSkillsScore[episode] = st.mean(averageSkillsScore)

    print("-> Results after " + str(numberEpisodes) + " episodes:")
    print("   -> Best student assigned " + str(round(bestStudentAssigned, 2)) + ".")
    print("   -> Average student assigned " + str(round((sumStudentAssigned / numberEpisodes), 2)) + ".")
    print("   -> Average steps per episode: " + str(round((totalSteps / numberEpisodes), 2)) + ".")
    print("   -> Average final students score: " + str(round(st.mean(allAverageStudentScore) * 100, 2)) + "%.")
    print("   -> Average final projects score: " + str(round(st.mean(allAverageProjectScore) * 100, 2)) + "%.")
    print("   -> Average final skills score: " + str(round(st.mean(allAverageSkillsScore) * 100, 2)) + "%.")
    print("   -> Average max students score: " + str(round(st.mean(averageMaxStudentScore) * 100, 2)) + "%.")
    print("   -> Average max projects score: " + str(round(st.mean(averageMaxProjectScore) * 100, 2)) + "%.")
    print("   -> Average max skills score: " + str(round(st.mean(averageMaxSkillsScore) * 100, 2)) + "%.")


def _finalAnalysis(finalState, studentsSelections):
    """"Function to analyze the resignation"""

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

        if len(sys.argv) > 1:
            mode = int(sys.argv[1])
            print("-> Mode of execution " + sys.argv[1] + ".")
            if len(sys.argv) > 2:
                numberOptions = int(sys.argv[2])
                print("-> Creating " + sys.argv[2] + " options per student.")
                if len(sys.argv) == 4:
                    typeAgent = int(sys.argv[3])
                    print("-> Agent type = " + sys.argv[3] + ".")
                else:
                    print("-> Missing type of agent, default = " + agentName[typeAgent] + ".")
            else:
                print("-> Missing type of agent, default = " + agentName[typeAgent] + ".")
                print("-> Missing number of option per student to create, default = " + str(DEFAULT_NUMBER_OPTIONS)
                      + ".")
        else:
            print("-> Missing what mode to use, default = " + str(mode) + ".")
            print("-> Missing type of agent, default = " + agentName[typeAgent] + ".")
            print("-> Missing number of option per student to create, default = " + str(DEFAULT_NUMBER_OPTIONS) + ".")

        students = []
        projects = []
        print("Importing data....")
        with open("../data/studentsProjectsData.json") as dataFile:
            data = json.load(dataFile)
            projectPlaces = data['placesInAllProjects']
            for student in data['students']:
                students += [student]

            for project in data['projects']:
                projects += [project]

        studentsDeleted = []
        studentsSelections = []
        if mode == 1:
            with open("../data/studentsSelectionData.json") as dataFile:
                data = json.load(dataFile)
                for student in data['results']:
                    studentsSelections += [{"options": student['optionSelected'],
                                            "studentId": student['studentId']}]
                studentsDeleted = data['studentsWithoutAssignations']
                print("Data imported.")
                print("-> " + str(len(students)) + " students and " + str(len(projects)) + " projects with "
                      + str(projectPlaces) + " places available imported.")
                studentsAlreadyAssigned, studentsAssigned = _assignStudentsWithSelections(len(students),
                                                                                          studentsSelections, projects)

        else:
            print("Data imported.")
            print("-> " + str(len(students)) + " students and " + str(len(projects)) + " projects with "
                  + str(projectPlaces) + " places available imported.")
            studentsAlreadyAssigned = []

        if projectPlaces < len(students):
            print("-> The students with less average mark won't have assignation.")
            while projectPlaces < len(students):
                studentsDeleted.append(students.pop()["id"])
            print("-> " + str(len(students)) + " students will be assigned for the "
                  + str(projectPlaces) + " places available.")

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
                qTable = _qTableCreator(len(students), len(projects), numberOptions)

                qTable = _qLearningTraining(env, qTable, numberOptions, len(projects))

                bestResult = _qLearningExecution(env, qTable, numberOptions, len(projects))

                print("Exporting QTable....")
                pd.DataFrame(data=qTable.astype(float)).to_csv('../data/qTableData.csv', sep=',', header=False,
                                                               index=False)
                print("QTable exported.")

            else:
                bestResult = _stableBaselineTrainingAndExecution(env, typeAgent, numberOptions, mode)

            env.close()
            print("Environment closed.")
        else:
            bestResult = studentsAlreadyAssigned

        if mode == 0:
            print("Exporting JSON of options....")
            with open("../data/optionsData.json", "w") as file:
                json.dump({"numberOptions": numberOptions,
                           "results": _generateOptionsData(bestResult, students, numberOptions, True),
                           "studentsWithoutAssignations": studentsDeleted}, file, indent=4)
            print("JSON of options exported.")
        else:
            _finalAnalysis(bestResult, studentsSelections)
            print("Exporting JSON with final options for students....")
            with open("../data/finalResults.json", "w") as file:
                json.dump({"results": _generateOptionsData(bestResult, students, numberOptions, False),
                           "studentsWithoutAssignations": studentsDeleted}, file, indent=4)
            print("JSON of final options for students exported.")

    except OSError as err:
        print("Error opening file: {0}".format(err))


if __name__ == '__main__':
    main()

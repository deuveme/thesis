import numpy as np
import gym
import math
import sys
from copy import deepcopy
from geopy.distance import geodesic
from gym import spaces

ALPHA = 7.   # 7
BETA = 2.  # 2
GAMMA = 1.  # 1

BREAKING_EPISODE = 100000


class Recommender4StudentsEnv(gym.Env):
    """Environment for a recommender for students for OpenAI gym"""

    def __init__(self, students, projects, numberOptions, withImage, mode, studentsAlreadyAssigned):
        """"Environment initialization"""

        super(Recommender4StudentsEnv, self).__init__()

        self.students = students
        self.ending = 0
        self.projects = projects
        self.withImage = withImage
        self.mode = mode
        if self.mode == 1:
            self.numberOptions = 1
            self.studentsAlreadyAssigned = studentsAlreadyAssigned
            # self.studentsSelections = []
        else:
            self.numberOptions = numberOptions

        self.action_space = spaces.MultiDiscrete((len(students), len(projects)))

        if withImage:
            bytesNeeded = numberOptions * len(students) * ((len(projects) / 256) + 1)

            self.stateImageSize = int(math.pow(2, math.ceil(math.log(bytesNeeded, 4))))

            self.observation_space = spaces.Box(low=0,
                                                high=255,
                                                shape=(self.stateImageSize, self.stateImageSize, 3),
                                                dtype=np.uint8)

        else:
            self.observation_space = spaces.MultiDiscrete([len(projects) + 1] * (len(students) * numberOptions))

        self.state = [[-1 for _ in range(numberOptions)] for _ in range(len(students))]
        self.assigned = [0 for _ in range(numberOptions)]
        self.studentsAssignedToProject = [[[] for _ in range(numberOptions)] for _ in range(len(projects))]
        self._assignStudents()
        self.inSameState = 0

    def _assignStudents(self):
        """"Assign students already assigned"""

        if self.mode == 1:
            for studentId in range(len(self.studentsAlreadyAssigned)):
                projectId = self.studentsAlreadyAssigned[studentId][0]
                if projectId != -1:
                    self.state[studentId][0] = projectId
                    self.assigned[0] += 1
                    self.studentsAssignedToProject[projectId][0] += [studentId]

    @staticmethod
    def _distanceCalculation(studentLocation, projectLocation):
        """"Calculate the score of the distance preference"""

        distance = geodesic(studentLocation, projectLocation).km
        return 0.0 if distance > 50 else (50.0 - distance) / 50.0

    @staticmethod
    def _salaryCalculation(studentSalary, projectSalary):
        """"Calculate the score of the salary preference"""

        difference = abs(studentSalary - projectSalary)
        return 1.0 if studentSalary <= projectSalary else difference / 500.0 if difference <= 500 else 0.0

    def _studentPreferencesPunctuation(self, student, project):
        """"Calculate the preference score of an student"""

        punctuation = 0.0
        factorsToEvaluate = 0.0
        if student["preferredLocation"]["importance"] != 0:
            factorsToEvaluate += student["preferredLocation"]["importance"] / 5.0
            punctuation += self._distanceCalculation(student["preferredLocation"]["value"], project["location"]) * (student["preferredLocation"]["importance"] / 5.0)
        if student["preferredRemote"]["importance"] != 0:
            factorsToEvaluate += student["preferredRemote"]["importance"] / 5.0
            punctuation += student["preferredRemote"]["importance"] / 5.0 \
                if project["remote"] == student["preferredRemote"]["value"] else 0.0
        if student["preferredMinimumSalary"]["importance"] != 0:
            factorsToEvaluate += student["preferredMinimumSalary"]["importance"] / 5.0
            punctuation += self._salaryCalculation(student["preferredMinimumSalary"]["value"], project["minimumSalary"]) * (student["preferredMinimumSalary"]["importance"] / 5.0)
        if student["preferredTypeInternship"]["importance"] != 0:
            factorsToEvaluate += student["preferredTypeInternship"]["importance"] / 5.0
            punctuation += student["preferredTypeInternship"]["importance"] / 5.0 \
                if project["type"] in student["preferredTypeInternship"]["list"] else 0.0
        return punctuation / factorsToEvaluate

    @staticmethod
    def _subtractionCalculation(firstValue, secondValue, maxValue):
        """"Calculate the score for a subtraction type preference"""

        difference = abs(firstValue - secondValue)
        return difference / maxValue if difference <= maxValue else 0.0

    @staticmethod
    def _markCalculation(studentMark, projectMark):
        """"Calculate the mark preference score"""

        difference = abs(studentMark - projectMark)
        return 1.0 if studentMark <= projectMark else difference / 1.0 if difference <= 1 else 0.0

    def _projectPreferencesPunctuation(self, student, project):
        """"Calculate the preference score of a project"""

        punctuation = 0.0
        factorsToEvaluate = 0.0
        if project["preferredAgeParticipants"]["importance"] != 0:
            factorsToEvaluate += project["preferredAgeParticipants"]["importance"] / 5.0
            punctuation += (project["preferredAgeParticipants"]["importance"] / 5.0) * self._subtractionCalculation(project["preferredAgeParticipants"]["value"], student["age"], 5)

        if project["preferredDegreeParticipants"]["importance"] != 0:
            factorsToEvaluate += project["preferredDegreeParticipants"]["importance"] / 5.0
            punctuation += (project["preferredDegreeParticipants"]["importance"] / 5.0) * self._subtractionCalculation(project["preferredDegreeParticipants"]["value"], student["degree"], 2)

        if project["preferredUniversityParticipants"]["importance"] != 0:
            factorsToEvaluate += project["preferredUniversityParticipants"]["importance"] / 5.0
            punctuation += project["preferredUniversityParticipants"]["importance"] / 5.0 \
                if student["university"] in project["preferredUniversityParticipants"]["list"] else 0.0

        if project["preferredAverageMark"]["importance"] != 0:
            factorsToEvaluate += project["preferredAverageMark"]["importance"] / 5.0
            punctuation += (project["preferredAverageMark"]["importance"] / 5.0) * self._markCalculation(student["averageMark"], project["preferredAverageMark"]["value"])

        if len(project["preferredWorkExperienceParticipants"]) != 0:
            factorsToEvaluate += 1.0 if len(project["preferredWorkExperienceParticipants"]) != 0 else 0.0
            totalProjectPunctuation = 0.0
            totalStudentPunctuation = 0.0
            for workExperience in project["preferredWorkExperienceParticipants"]:
                totalProjectPunctuation += (workExperience[0] / 5.0)
                if workExperience[1] in student["workExperience"]:
                    totalStudentPunctuation += (workExperience[0] / 5.0)
            punctuation += totalStudentPunctuation / totalProjectPunctuation

        if len(project["preferredVolunteerExperienceParticipants"]) != 0:
            factorsToEvaluate += 1.0 if len(project["preferredVolunteerExperienceParticipants"]) != 0 else 0.0
            totalProjectPunctuation = 0.0
            totalStudentPunctuation = 0.0
            for volunteerExperience in project["preferredVolunteerExperienceParticipants"]:
                totalProjectPunctuation += (volunteerExperience[0] / 5.0)
                if volunteerExperience[1] in student["volunteerExperience"]:
                    totalStudentPunctuation += (volunteerExperience[0] / 5.0)
            punctuation += totalStudentPunctuation / totalProjectPunctuation

        if len(project["preferredLanguagesParticipants"]) != 0:
            factorsToEvaluate += 1.0 if len(project["preferredLanguagesParticipants"]) != 0 else 0.0
            totalProjectPunctuation = 0.0
            totalStudentPunctuation = 0.0
            for languages in project["preferredLanguagesParticipants"]:
                totalProjectPunctuation += (languages[0] / 5.0)
                if languages[1] in student["languages"]:
                    totalStudentPunctuation += (languages[0] / 5.0)
            punctuation += totalStudentPunctuation / totalProjectPunctuation

        return punctuation / factorsToEvaluate

    def _skillsPunctuation(self, studentNumber, project, option):
        if studentNumber == -1:
            final = True
        else:
            final = False
        skillsPunctuation = 0.0
        oldSkillsPunctuation = 0.0
        maximumSkillsPunctuation = 0.0
        for skill in project["preferredSkillsNeeded"]:
            # print("------")
            # print(skill[1])
            # print(self.studentsAssignedToProject[project["id"]])
            # print("--s--")
            maximumSkillsPunctuation += skill[0] / 5.0
            for studentId in self.studentsAssignedToProject[project["id"]][option]:
                # print(self.students[studentId]["skills"])
                if skill[1] in self.students[studentId]["skills"]:
                    skillsPunctuation += skill[0] / 5.0
                    if not final and studentId != studentNumber:
                        oldSkillsPunctuation += skill[0] / 5.0
                    break
            # print("--e--")

        skillsPunctuation /= maximumSkillsPunctuation
        oldSkillsPunctuation /= maximumSkillsPunctuation

        return skillsPunctuation - oldSkillsPunctuation

    def _rewardCalculation(self, studentNumber, projectNumber, option):
        """"Reward calculator (Max 10, min 0)"""

        student = self.students[studentNumber]
        project = self.projects[projectNumber]

        studentPreferencesPunctuation = self._studentPreferencesPunctuation(student, project)
        projectPreferencesPunctuation = self._projectPreferencesPunctuation(student, project)
        skillsPunctuation = self._skillsPunctuation(studentNumber, project, option)

        return (ALPHA * studentPreferencesPunctuation) + \
               (BETA * projectPreferencesPunctuation) + \
               (GAMMA * skillsPunctuation)

    def _imageStateGeneration(self):
        """"Function to convert state to image"""

        imageState = np.zeros(self.stateImageSize * self.stateImageSize, dtype=np.uint8)

        i = 0
        for student in range(len(self.students)):
            for option in range(self.numberOptions):
                projectNumber = self.state[student][option]
                base256 = (projectNumber + 1).to_bytes(((projectNumber + 1).bit_length() + 7), 'big')
                base256NormalizedProject = int.from_bytes(base256, 'big')
                size = int(sys.getsizeof(base256NormalizedProject) / 256) + 1
                imageState[i:i+size] = base256NormalizedProject
                i += size

        imageState = np.reshape(imageState, (self.stateImageSize, self.stateImageSize))
        imageState = np.stack((imageState,) * 3, axis=-1)
        return imageState

    def _isDone(self):
        """"Function to check if the state is final"""

        optionFull = 0
        projectsWithFreePlaces = [[] for _ in range(self.numberOptions)]
        '''
        Mira si quedan projectos con plazas vacias.
        '''
        for option in range(self.numberOptions):
            projectsFullInOption = 0
            for projectId in range(len(self.studentsAssignedToProject)):
                if len(self.studentsAssignedToProject[projectId][option]) == self.projects[projectId]["nParticipants"]:
                    projectsFullInOption += 1
                else:
                    projectsWithFreePlaces[option].append(projectId)
            if projectsFullInOption == len(self.projects):
                optionFull += 1
        if optionFull == self.numberOptions:
            self.ending += 1
            print("ENDING. All projects full. Finished " + str(self.ending) + " times.")
            return True
        '''
        Mira si quedan estudiantes libres.
        '''
        missingStudents = False
        for optionAssignations in self.assigned:
            if optionAssignations != len(self.students):
                # print(projectsWithFreePlaces)
                missingStudents = True
                break

        if not missingStudents:
            self.ending += 1
            print("ENDING. All students assigned. Finished " + str(self.ending) + " times.")
            return True
        '''
        Si quedan libres y llevamos tiempo en el mismo estado.
        '''
        if missingStudents and self.inSameState > BREAKING_EPISODE:
            ''' Miramos para cada opcion cada proyecto con plazas libres '''
            for optionNumber in range(self.numberOptions):
                if projectsWithFreePlaces[optionNumber]:
                    ''' Para cada estudiantes por asignar en esa opcion '''
                    for studentId in range(len(self.students)):
                        if self.state[studentId][optionNumber] == -1:
                            ''' 
                            Miramos si ha sido asignado ya en los proyectos que quedan libres en las otras opciones
                            '''
                            for optionNumber2 in range(self.numberOptions):
                                if optionNumber2 != optionNumber:
                                    freeProjectsForStudent = 0
                                    for projectWithFreePlaces in projectsWithFreePlaces[optionNumber]:
                                        if self.state[studentId][optionNumber2] != projectWithFreePlaces:
                                            freeProjectsForStudent += 1
                                    if freeProjectsForStudent == 0:
                                        # print(projectsWithFreePlaces)
                                        # print("Student with id = " + str(studentId) +
                                        #      " in option " + str(optionNumber2) +
                                        #      " => project assigned = " + str(self.state[studentId][optionNumber2]))
                                        self.ending += 1
                                        print("ENDING. Is not possible to assign more students."
                                              " Finished " + str(self.ending) + " times.")
                                        return True
        return False

    def stepScores(self):
        """"Function to compute punctuation of state"""

        studentScores = [[0.0 for _ in range(self.numberOptions)] for _ in range(len(self.students))]
        totalStudentsPunctuation = 0.0
        studentsAssigned = [0 for _ in range(self.numberOptions)]
        for studentNumber in range(len(self.students)):
            allOptionsStudentPunctuation = 0.0
            student = self.students[studentNumber]
            for option in range(self.numberOptions):
                projectNumber = self.state[studentNumber][option]
                project = self.projects[self.state[studentNumber][option]]
                studentPreferencesPunctuation = self._studentPreferencesPunctuation(student, project)

                allOptionsStudentPunctuation += studentPreferencesPunctuation

                if projectNumber != -1:
                    studentScores[studentNumber][option] = studentPreferencesPunctuation
                    studentsAssigned[option] += 1
                else:
                    studentScores[studentNumber][option] = -1

            totalStudentsPunctuation += allOptionsStudentPunctuation / self.numberOptions

        projectScores = [[[0.0 for _ in range(self.numberOptions)] for _ in range(2)] for _ in
                         range(len(self.projects))]
        projectAssignations = [[0 for _ in range(self.numberOptions)] for _ in range(len(self.projects))]
        totalProjectsPunctuation = 0.0
        totalSkillsPunctuation = 0.0
        for projectNumber in range(len(self.projects)):
            project = self.projects[projectNumber]
            allOptionsProjectPunctuation = 0.0
            allOptionsProjectSkillsPunctuation = 0.0
            for option in range(self.numberOptions):
                students = self.studentsAssignedToProject[projectNumber][option]
                for studentId in students:
                    student = self.students[studentId]
                    projectPreferencesPunctuation = self._projectPreferencesPunctuation(student, project) / len(students)
                    allOptionsProjectPunctuation += projectPreferencesPunctuation
                    projectScores[projectNumber][0][option] += projectPreferencesPunctuation

                projectSkillsPreferencesPunctuation = self._skillsPunctuation(-1, project, option)
                allOptionsProjectSkillsPunctuation += projectSkillsPreferencesPunctuation
                projectScores[projectNumber][1][option] = projectSkillsPreferencesPunctuation

                projectAssignations[projectNumber][option] = len(students)

            totalProjectsPunctuation += allOptionsProjectPunctuation / self.numberOptions
            totalSkillsPunctuation += allOptionsProjectSkillsPunctuation / self.numberOptions

        return totalStudentsPunctuation / len(self.students), \
               totalProjectsPunctuation / len(self.projects),\
               totalSkillsPunctuation / len(self.projects), \
               studentScores, projectScores, \
               studentsAssigned, projectAssignations

    def step(self, action):
        """"Environment next step generator"""

        studentNumber = action[0]
        projectNumber = action[1]
        reward = 0
        info = "[Nothing done. Trying to assign student " + str(studentNumber) + " to " + str(projectNumber) + \
               ". Reward 0.]"

        for option in range(0, self.numberOptions):
            '''
            Valida si: 
                - Tiene proyecto asignado para esa opcion.
                - Si el proyecto al cual se le quiere asignar esta lleno.
                - Si el proyecto no esta asignado en otras opciones al mismo estudiante.
            '''
            if self.state[studentNumber][option] == -1 \
                    and \
                    len(self.studentsAssignedToProject[projectNumber][option]) < self.projects[projectNumber][
                "nParticipants"] \
                    and \
                    self.projects[projectNumber]["id"] not in self.state[studentNumber]:
                self.state[studentNumber][option] = self.projects[projectNumber]["id"]
                reward = self._rewardCalculation(studentNumber, projectNumber, option)
                self.assigned[option] += 1
                self.studentsAssignedToProject[projectNumber][option] += [studentNumber]
                info = "[Option number " + str(option) + " of student " + str(studentNumber) + " assigned to project " \
                       + str(projectNumber) + ". Reward = " + str(reward) + " out of 10.]"

                # print(info)
                self.inSameState = 0
                return deepcopy(self._imageStateGeneration()) if self.withImage else deepcopy(self.state), \
                       reward, self._isDone(), {}

            '''
            Valida que se llene primero su primera opcion y asi sucesivamente.
            '''
            if self.state[studentNumber][option] == -1:
                break

        # print(info)
        self.inSameState += 1
        return deepcopy(self._imageStateGeneration()) if self.withImage else deepcopy(self.state), \
               reward, self._isDone(), {}

    def finalState(self):
        return deepcopy(self.state)

    def reset(self, execution=0):
        """"Environment reset function"""

        self.state = [[-1 for _ in range(self.numberOptions)] for _ in range(len(self.students))]
        self.assigned = [0 for _ in range(self.numberOptions)]
        self.studentsAssignedToProject = [[[] for _ in range(self.numberOptions)] for _ in range(len(self.projects))]
        self._assignStudents()
        self.inSameState = 0
        if execution:
            self.ending = 0

        return deepcopy(self._imageStateGeneration()) if self.withImage else deepcopy(self.state)

    def render(self):
        """"Environment render function"""

        print("Students assignations:")
        for student in range(len(self.students)):
            print("- Student " + str(student) + ":")
            for option in range(self.numberOptions):
                project = self.state[student][option]
                if project == -1:
                    print("   - Option " + str(option) + ": Not assigned yet.")
                else:
                    print("   - Option " + str(option) + ": Assigned to project " + str(project) + " ("
                          + str(len(self.studentsAssignedToProject[project][option])) + "/"
                          + str(self.projects[project]["nParticipants"]) + " places filled).")

        print("Projects assignations:")
        for project in range(len(self.projects)):
            print("- Project " + str(project) + ":")
            for option in range(self.numberOptions):
                students = self.studentsAssignedToProject[project][option]
                if not len(students):
                    print("   - Option " + str(option) + ": No student assigned yet.")
                else:
                    print("   - Option " + str(option) + ": Assigned to students " + str(students) + " ("
                          + str(len(students)) + "/"
                          + str(self.projects[project]["nParticipants"]) + " places filled).")

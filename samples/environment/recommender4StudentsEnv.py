import numpy as np
import gym
import math
import sys
from copy import deepcopy
from geopy.distance import geodesic
from gym import error, spaces, utils
from gym.utils import seeding

ALPHA = 5.0
GAMMA = 3.0
EPSILON = 2.0


class Recommender4StudentsEnv(gym.Env):
    """A Recommender for students environment for OpenAI gym"""

    def __init__(self, students, projects, numberOptions, withImage, mode, studentsAlreadyAssigned):
        """"Environment initialization"""

        super(Recommender4StudentsEnv, self).__init__()

        self.students = students
        self.projects = projects
        self.withImage = withImage
        self.mode = mode
        if self.mode == 1:
            self.numberOptions = 1
            self.studentsAlreadyAssigned = studentsAlreadyAssigned
            self.studentsSelections = []
        else:
            self.numberOptions = numberOptions

        self.action_space = spaces.MultiDiscrete((len(students), len(projects)))

        if withImage:
            # self.ObservationSpaceLimits = [len(projects) + 1] * (len(students) * numberOptions)
            self.bytesNeeded = numberOptions * len(students) * ((len(projects) / 256) + 1)

            # self.bytesNeeded = sum(map(lambda limit: math.ceil(math.log(limit, 256)),
            #                           self.ObservationSpaceLimits))
            self.stateImageSize = int(math.pow(2, math.ceil(math.log(self.bytesNeeded, 4))))

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

    def _assignStudents(self):
        if self.mode == 1:
            for studentId in range(len(self.studentsAlreadyAssigned)):
                projectId = self.studentsAlreadyAssigned[studentId][0]
                if projectId != -1:
                    self.state[studentId][0] = projectId
                    self.assigned[0] += 1
                    self.studentsAssignedToProject[projectId][0] += [studentId]

    @staticmethod
    def _distanceCalculation(studentLocation, projectLocation):
        distance = geodesic(studentLocation, projectLocation).km
        return 0.0 if distance > 50 else (50.0 - distance) / 50.0

    @staticmethod
    def _salaryCalculation(studentSalary, projectSalary):
        difference = abs(studentSalary - projectSalary)
        return 1.0 if studentSalary <= projectSalary else difference / 500.0 if difference <= 500 else 0.0

    def _studentPreferencesPunctuation(self, student, project):
        punctuation = 0.0
        factorsToEvaluate = 0.0
        if student["preferredLocation"]["importance"] != 0:
            factorsToEvaluate += 1.0
            punctuation += self._distanceCalculation(student["preferredLocation"]["value"], project["location"]) * (student["preferredLocation"]["importance"] / 5.0)
        if student["preferredRemote"]["importance"] != 0:
            factorsToEvaluate += 1.0
            punctuation += student["preferredRemote"]["importance"] / 5.0 \
                if project["remote"] == student["preferredRemote"]["value"] else 0.0
        if student["preferredMinimumSalary"]["importance"] != 0:
            factorsToEvaluate += 1.0
            punctuation += self._salaryCalculation(student["preferredMinimumSalary"]["value"], project["minimumSalary"]) * (student["preferredMinimumSalary"]["importance"] / 5.0)
        if student["preferredTypeInternship"]["importance"] != 0:
            factorsToEvaluate += 1.0
            punctuation += student["preferredTypeInternship"]["importance"] / 5.0 \
                if project["type"] in student["preferredTypeInternship"]["list"] else 0.0
        return punctuation / factorsToEvaluate

    @staticmethod
    def _subtractionCalculation(firstValue, secondValue, maxValue):
        difference = abs(firstValue - secondValue)
        return difference if difference <= maxValue else 0.0

    @staticmethod
    def _markCalculation(studentMark, projectMark):
        difference = abs(studentMark - projectMark)
        return 1.0 if studentMark <= projectMark else difference / 1.0 if difference <= 1 else 0.0

    def _projectPreferencesPunctuation(self, student, project):
        punctuation = 0.0
        factorsToEvaluate = 0.0
        if project["preferredAgeParticipants"]["importance"] != 0:
            factorsToEvaluate += 1.0
            punctuation += (project["preferredAgeParticipants"]["importance"] / 5.0) * self._subtractionCalculation(project["preferredAgeParticipants"]["value"], student["age"], 5)
        if project["preferredDegreeParticipants"]["importance"] != 0:
            factorsToEvaluate += 1.0
            punctuation += (project["preferredDegreeParticipants"]["importance"] / 5.0) * self._subtractionCalculation(project["preferredDegreeParticipants"]["value"], student["degree"], 2)
        if project["preferredUniversityParticipants"]["importance"] != 0:
            factorsToEvaluate += 1.0
            punctuation += project["preferredUniversityParticipants"]["importance"] / 5.0 \
                if student["university"] in project["preferredUniversityParticipants"]["list"] else 0.0
        if project["preferredAverageMark"]["importance"] != 0:
            factorsToEvaluate += 1.0
            punctuation += (project["preferredAverageMark"]["importance"] / 5.0) * self._markCalculation(student["averageMark"], project["preferredAverageMark"]["value"])

        if len(project["preferredWorkExperienceParticipants"]) != 0:
            factorsToEvaluate += 1.0
            for workExperience in project["preferredWorkExperienceParticipants"]:
                if workExperience[1] in student["workExperience"]:
                    punctuation += (workExperience[0] / 5.0) / len(project["preferredWorkExperienceParticipants"])

        if len(project["preferredVolunteerExperienceParticipants"]) != 0:
            factorsToEvaluate += 1.0
            for volunteerExperience in project["preferredVolunteerExperienceParticipants"]:
                if volunteerExperience[1] in student["volunteerExperience"]:
                    punctuation += (volunteerExperience[0] / 5.0) \
                                   / len(project["preferredVolunteerExperienceParticipants"])

        if len(project["preferredLanguagesParticipants"]) != 0:
            factorsToEvaluate += 1.0
            for languages in project["preferredLanguagesParticipants"]:
                if languages[1] in student["languages"]:
                    punctuation += (languages[0] / 5.0) / len(project["preferredLanguagesParticipants"])

        return punctuation / factorsToEvaluate

    def _rewardCalculation(self, studentNumber, projectNumber, option):
        """"Reward calculator (Max 10, min 0)"""

        student = self.students[studentNumber]
        project = self.projects[projectNumber]

        studentPreferencesPunctuation = self._studentPreferencesPunctuation(student, project)
        projectPreferencesPunctuation = self._projectPreferencesPunctuation(student, project)
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
                    if studentId != studentNumber:
                        oldSkillsPunctuation += skill[0] / 5.0
                    break
            # print("--e--")

        skillsPunctuation /= maximumSkillsPunctuation
        oldSkillsPunctuation /= maximumSkillsPunctuation

        # print("-------------------")
        # print("Student: " + str(studentPreferencesPunctuation) + ", is "
        # + str((alpha * studentPreferencesPunctuation)))
        # print("Project: " + str(projectPreferencesPunctuation) + ", is "
        # + str((gamma * projectPreferencesPunctuation)))
        # print("Skills: " + str(skillsPunctuation - oldSkillsPunctuation)
        # + ", is " + str((epsilon * (skillsPunctuation - oldSkillsPunctuation))))

        return (ALPHA * studentPreferencesPunctuation) + \
               (GAMMA * projectPreferencesPunctuation) + \
               (EPSILON * (skillsPunctuation - oldSkillsPunctuation))

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
        for option in range(len(self.studentsAssignedToProject)):
            projectsFullInOption = 0
            for projectId in range(len(self.studentsAssignedToProject)):
                if len(self.studentsAssignedToProject[projectId]) == self.projects[projectId]["nParticipants"]:
                    projectsFullInOption += 1
            if projectsFullInOption == len(self.projects):
                optionFull += 1
        if optionFull == self.numberOptions:
            return True
        for assignations in self.assigned:
            if assignations != len(self.students):
                return False
        return True

    def stepScores(self):
        """"Function to compute punctuation of state"""

        totalStudentsPunctuation = 0.0
        for studentNumber in range(len(self.students)):
            totalStudentPunctuation = 0.0
            student = self.students[studentNumber]
            for option in range(self.numberOptions):
                project = self.projects[self.state[studentNumber][option]]
                studentPreferencesPunctuation = self._studentPreferencesPunctuation(student, project)
                totalStudentPunctuation += studentPreferencesPunctuation

            totalStudentsPunctuation += totalStudentPunctuation / self.numberOptions

        totalProjectsPunctuation = 0.0
        totalSkillsPunctuation = 0.0
        for projectNumber in range(len(self.projects)):
            project = self.projects[projectNumber]
            totalProjectPunctuation = 0.0
            totalProjectSkillsPunctuation = 0.0
            for option in range(self.numberOptions):
                students = self.studentsAssignedToProject[projectNumber][option]
                for studentId in students:
                    student = self.students[studentId]
                    totalProjectPunctuation += self._projectPreferencesPunctuation(student, project)

                maximumSkillsPunctuation = 0.0
                skillsPunctuation = 0.0
                for skill in project["preferredSkillsNeeded"]:
                    maximumSkillsPunctuation += skill[0] / 5.0
                    for studentId in self.studentsAssignedToProject[project["id"]][option]:
                        if skill[1] in self.students[studentId]["skills"]:
                            skillsPunctuation += skill[0] / 5.0
                            break
                totalProjectSkillsPunctuation += skillsPunctuation / maximumSkillsPunctuation

            totalProjectsPunctuation += totalProjectPunctuation / self.numberOptions
            totalSkillsPunctuation += totalProjectSkillsPunctuation / self.numberOptions

        return totalStudentsPunctuation / len(self.students), \
               totalProjectsPunctuation / len(self.projects),\
               totalSkillsPunctuation / len(self.projects)

    def step(self, action):
        """"Environment next step generator"""

        studentNumber = action[0]
        projectNumber = action[1]
        reward = 0
        info = "[Nothing done. Trying to assign student " + str(studentNumber) + " to " + str(projectNumber) + \
               ". Reward 0.]"

        if self.mode == 2:     # Change it !!!
            # print(self.assigned)
            # print(self.studentsSelections[self.assigned[0]]['studentId'])
            reward = 0
            studentToAssign = self.studentsSelections[self.assigned[0]]['studentId']
            # print("Goal: assign student = " + str(studentToAssign))
            if self.state[studentToAssign][0] == -1:
                if studentNumber == studentToAssign:
                    '''
                    # Mirar primera opcion con plazas
                    '''
                    for option in range(0, len(self.studentsSelections[self.assigned[0]]['options'])):
                        projectId = self.studentsSelections[self.assigned[0]]['options'][option]
                        # print("option " + str(option) + " = " + str(projectId))
                        if len(self.studentsAssignedToProject[projectId][0]) < self.projects[projectId]["nParticipants"]:
                            '''
                            # Una vez encontrada, mirar si coincide con la propuesta, si coincide se asigna, sino no
                            '''
                            # print("To project = " + str(projectId) + " (option " + str(option) + ")")
                            if projectNumber == projectId:
                                self.state[studentNumber][0] = self.projects[projectNumber]["id"]
                                self.studentsAssignedToProject[projectNumber][0] += [studentNumber]
                                reward = 10
                                self.assigned[0] += 1
                                info = "[Student " + str(studentNumber) + " assigned to project " \
                                       + str(projectNumber) + ". Reward = " + str(reward) + " out of 10.]"
                                # print(info)
                                return deepcopy(self._imageStateGeneration()) if self.withImage else deepcopy(self.state), \
                                       reward, self._isDone(), {}
                            else:
                                reward = 0
                                # print(info)
                                return deepcopy(self._imageStateGeneration()) if self.withImage else deepcopy(self.state), \
                                       reward, self._isDone(), {}

            reward = 0
            # print(info)
            return deepcopy(self._imageStateGeneration()) if self.withImage else deepcopy(self.state), \
                   reward, self._isDone(), {}

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

                return deepcopy(self._imageStateGeneration()) if self.withImage else deepcopy(self.state), \
                       reward, self._isDone(), {}

            '''
            Valida que se llene primero su primera opcion y asi sucesivamente.
            '''
            if self.state[studentNumber][option] == -1:
                break

        return deepcopy(self._imageStateGeneration()) if self.withImage else deepcopy(self.state), \
               reward, self._isDone(), {}

    def finalState(self):
        return deepcopy(self.state)

    def reset(self):
        """"Environment reset function"""

        self.state = [[-1 for _ in range(self.numberOptions)] for _ in range(len(self.students))]
        self.assigned = [0 for _ in range(self.numberOptions)]
        self.studentsAssignedToProject = [[[] for _ in range(self.numberOptions)] for _ in range(len(self.projects))]
        self._assignStudents()
        # print(self.state)
        # print(self.assigned)
        # print(self.studentsAssignedToProject)

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

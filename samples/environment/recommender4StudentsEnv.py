import gym
from gym import error, spaces, utils
from gym.utils import seeding


class Recommender4StudentsEnv(gym.Env):
    """A Recommender for students environment for OpenAI gym"""

    def __init__(self, students, projects, numberOptions):
        """"Environment initialization"""

        super(Recommender4StudentsEnv, self).__init__()

        self.students = students
        self.projects = projects
        self.numberOptions = numberOptions

        self.action_space = spaces.MultiDiscrete((len(students), len(projects)))

        self.observation_space = spaces.MultiDiscrete([(1, len(students)), (1, len(projects))])

        self.state = [[-1 for _ in range(numberOptions)] for _ in range(len(students))]
        self.assigned = [0 for _ in range(numberOptions)]
        self.studentsAssignedToProject = [[[] for _ in range(numberOptions)] for _ in range(len(projects))]

    def _isDone(self):
        """"Function to check if an state is final"""

        for assignations in self.assigned:
            if assignations != len(self.students):
                return False
        return True

    def step(self, action):
        """"Environment next step generator"""

        studentNumber = action[0]
        projectNumber = action[1]
        reward = 0
        info = "[Nothing done. Reward 0.]"
        for option in range(0, self.numberOptions):
            '''
            Valida si: 
                - Tiene proyecto asignado para esa opción.
                - Si el proyecto al cual se le quiere asignar esta lleno.
                - Si el proyecto no esta asignado en otras opciones al mismo estudiante.
            '''
            if self.state[studentNumber][option] == -1 \
               and \
               len(self.studentsAssignedToProject[projectNumber][option]) < self.projects[projectNumber]["nParticipants"]\
               and \
               self.projects[projectNumber]["id"] not in self.state[studentNumber]:

                self.state[studentNumber][option] = self.projects[projectNumber]["id"]
                reward = 1
                self.assigned[option] += 1
                self.studentsAssignedToProject[projectNumber][option] += [studentNumber]
                info = "[Option number " + str(option) + " of student " + str(studentNumber) + " assigned to " \
                       + str(projectNumber) + ". Reward 1.]"
                return {}, reward, self._isDone(), info

            '''
            Valida que se llene primero su primera opción y así sucesivamente.
            '''
            if self.state[studentNumber][option] == -1:
                break

        return {}, reward, self._isDone(), info

    def reset(self):
        """"Environment reset function"""

        self.state = [[-1 for _ in range(self.numberOptions)] for _ in range(len(self.students))]
        self.assigned = [0 for _ in range(self.numberOptions)]
        self.studentsAssignedToProject = [[[] for _ in range(self.numberOptions)] for _ in range(len(self.projects))]
        return {}

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
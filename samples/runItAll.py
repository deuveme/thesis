import os
import sys

DEFAULT_NUMBER_PROJECTS = "20"
DEFAULT_NUMBER_STUDENTS = "100"
DEFAULT_NUMBER_OPTIONS = "3"
DEFAULT_AGENT = "2"


def main():
    print("Executing all scripts together....")
    print("Reading arguments....")

    numberProjects = DEFAULT_NUMBER_PROJECTS
    numberStudents = DEFAULT_NUMBER_STUDENTS
    numberOptions = DEFAULT_NUMBER_OPTIONS
    typeAgent = DEFAULT_AGENT
    if len(sys.argv) > 2:
        numberStudents = sys.argv[1]
        if len(sys.argv) > 3:
            numberProjects = sys.argv[2]
            if len(sys.argv) > 4:
                numberOptions = sys.argv[3]
                if len(sys.argv) == 5:
                    typeAgent = sys.argv[4]

    print("Executing system with:")
    print(" -> Number of students: " + numberStudents)
    print(" -> Number of projects: " + numberStudents)
    print(" -> Number of options: " + numberStudents)
    print(" -> Number of agent to use: " + numberStudents)

    print("[EXECUTING generateJSON.py]")
    os.system("python ./generateJSON.py " + numberStudents + " " + numberProjects)
    print("[DONE]")

    print("[EXECUTING recommender.py]")
    os.system("python ./recommender.py " + str(0) + " " + numberOptions + " " + typeAgent + " " + str(1) + " " + str(0))
    print("[DONE]")

    print("[EXECUTING selector.py]")
    os.system("python ./selector.py")
    print("[DONE]")

    print("[EXECUTING recommender.py]")
    os.system("python ./recommender.py " + str(1) + " " + str(1) + " " + typeAgent + " " + str(1) + " " + str(0))
    print("[DONE]")


if __name__ == '__main__':
    main()

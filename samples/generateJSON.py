import random
import sys
import json

DEFAULT_NUMBER_PROJECTS = 20
DEFAULT_NUMBER_STUDENTS = 100


def generateOptions():
    universities = ["UPC", "UB", "UPF", "UAB", "UOC", "URV"]
    abilities = ["C", "C++", "Python", "PHP", "Golang", "Docker", "Java", ".NET", "C#", "Ruby", "Node.JS",
                 "React", "Javascript", "Perl", "Rest API", "XML", "PostgreSQL", "MySQL", "MongoDB"]
    languages = ["English", "Spanish", "Catalan", "German", "French", "Greek", "Italian", "Russian",
                 "Romanian", "Hungarian", "Czech", "Polish", "Arab", "Chinese", "Japanese", "Korean"]
    workExperiences = ["Full-Stack", "BackEnd", "FrontEnd", "Human Resources", "Engineering", "Administration",
                       "Investigation"]
    volunteerExperiences = ["Events", "Helping poor people", "Public Administration"]
    companies = ["Siemens", "HP", "Glovo", "Skyscanner", "IThinkUPC", "MediaMarkt", "Uber", "CaixaBank", "Inditex",
                 "Seat", "Mango", "Adidas", "Google", "Facebook", "Typeform", "Wallapop", "Holaluz", "Housfy", "Xceed",
                 "Social Point", "Privalia", "Signaturit", "Cooltra", "MrNoow", "Goin", "GetApp", "Andjoy", "Factorial"]
    students = ["David", "Pol", "Cecilio", "Laura", "Paula", "Ricardo", "Sara", "Pau", "Maria", "Teresa", "Antoni",
                "Bernat", "Lidia", "Andrea", "Valentina", "Roc", "Anna", "Elena", "Felix", "Albert", "Ferran", "Sandra",
                "Julia", "Federico", "Jesus", "Cristiano", "Cristina", "Gerard", "Leo", "Eric", "Eva", "Hugo", "Ivan"
                "Juan", "Lara", "Mar", "Nora", "Luz", "Raul", "Hector", "Helena", "Penelope", "Alba", "Alejandro",
                "Alvaro", "Emma", "Lucas", "Lucia", "Manuel", "Mariana", "Martin", "Ester", "Gabriel", "Isabel",
                "Jorge", "Marta", "Raquel", "Samuel", "Felipe", "Margarita", "Carla", "Francisco"]
    return universities, abilities, languages, workExperiences, volunteerExperiences, companies, students


def generateListWithImportance(inputList, minValue, valueMax):
    result = []
    for _ in range(0, random.randint(minValue, valueMax)):
        findIt = False
        while not findIt:
            element = inputList[random.randint(0, len(inputList) - 1)]
            if element not in result:
                result.append((random.randint(1, 5), element))
                findIt = True
    return result


def generateList(inputList, valueMin, valueMax=-1):
    if valueMax == -1:
        valueMax = len(inputList) - 1

    result = []
    for _ in range(0, random.randint(valueMin, valueMax)):
        findIt = False
        while not findIt:
            element = inputList[random.randint(0, len(inputList) - 1)]
            if element not in result:
                result.append(element)
                findIt = True
    return result


def generateImportanceAndList(inputList, maxValue):
    importance = random.randint(0, 5)
    if importance != 0:
        return {"importance": importance, "list": generateList(inputList, 1, random.randint(1, maxValue))}
    else:
        return {"importance": 0}


def generateImportanceAndValue(value):
    importance = random.randint(0, 5)
    if importance != 0:
        return {"importance": importance, "value": value}
    else:
        return {"importance": 0}


def generateStudent(studentId, universities, allAbilities, allLanguages, allWorkExperiences, allVolunteerExperiences,
                    students):
    return {"id": studentId, "name": students[random.randint(0, len(students) - 1)],
            "age": random.randint(18, 25), "degree": random.randint(1, 5),
            "averageMark": round(random.uniform(5.00, 10.00), 2),
            "university": universities[random.randint(0, len(universities) - 1)],
            "skills": generateList(allAbilities, 1, 6),
            "languages": generateList(allLanguages, 1, 4),
            "workExperience": generateList(allWorkExperiences, 0),
            "volunteerExperience": generateList(allVolunteerExperiences, 0),
            "description": "User generated random",
            "preferredLocation": generateImportanceAndValue((random.uniform(41.43, 41.34), random.uniform(2.05, 2.21))),
            "preferredRemote": generateImportanceAndValue(random.randint(0, 1)),
            "preferredMinimumSalary": generateImportanceAndValue(random.randint(300, 1500)),
            "preferredTypeInternship":  generateImportanceAndList(allWorkExperiences, len(allWorkExperiences) - 1)}


def generateProject(projectId, allUniversities, allAbilities, allLanguages, allWorkExperiences,
                    allVolunteerExperiences, companies):
    return {"id": projectId, "companyName": companies[random.randint(0, len(companies) - 1)],
            "projectName": "Random Project number " + str(projectId),
            "nParticipants": random.randint(1, 5),
            "location": (random.uniform(41.43, 41.34), random.uniform(2.05, 2.21)),
            "remote": random.randint(0, 1),
            "type": allWorkExperiences[random.randint(0, len(allWorkExperiences) - 1)],
            "minimumSalary": random.randint(300, 1500),
            "description": "Project generated random",
            "preferredSkillsNeeded": generateListWithImportance(allAbilities, 1, 6),
            "preferredAgeParticipants": generateImportanceAndValue(random.randint(18, 25)),
            "preferredDegreeParticipants": generateImportanceAndValue(random.randint(1, 5)),
            "preferredUniversityParticipants": generateImportanceAndList(allUniversities, 3),
            "preferredWorkExperienceParticipants": generateListWithImportance(allWorkExperiences, 0, 3),
            "preferredVolunteerExperienceParticipants": generateListWithImportance(allVolunteerExperiences, 0, 3),
            "preferredLanguagesParticipants": generateListWithImportance(allLanguages, 1, 3),
            "preferredAverageMark": generateImportanceAndValue(round(random.uniform(5.00, 9.70), 2))}


def generateStudents(numberStudents, allUniversities, allAbilities, allLanguages, allWorkExperiences,
                     allVolunteerExperiences, allStudents):
    students = []
    for studentId in range(0, numberStudents):
        students.append(generateStudent(studentId, allUniversities, allAbilities, allLanguages, allWorkExperiences,
                                        allVolunteerExperiences, allStudents))

    return students


def generateProjects(numberProjects, allUniversities, allAbilities, allLanguages, allWorkExperiences,
                     allVolunteerExperiences, companies):
    projects = []
    for projectId in range(0, numberProjects):
        projects.append(generateProject(projectId, allUniversities, allAbilities, allLanguages, allWorkExperiences,
                                        allVolunteerExperiences, companies))

    return projects


def main():
    print("Reading arguments....")

    numberProjects = DEFAULT_NUMBER_PROJECTS
    numberStudents = DEFAULT_NUMBER_STUDENTS
    if len(sys.argv) > 2:
        numberStudents = int(sys.argv[1])
        print("-> Creating " + sys.argv[1] + " students.")
        if len(sys.argv) == 3:
            numberProjects = int(sys.argv[2])
            print("-> Creating " + sys.argv[2] + " projects.")
        else:
            print("-> Missing number of projects to create, default = " + str(DEFAULT_NUMBER_PROJECTS) + ".")
    else:
        print("-> Missing number of students to create, default = " + str(DEFAULT_NUMBER_STUDENTS) + ".")
        print("-> Missing number of projects to create, default = " + str(DEFAULT_NUMBER_PROJECTS) + ".")

    allUniversities, allAbilities, allLanguages, allWorkExperiences, \
        allVolunteerExperiences, companies, allStudents = generateOptions()

    print("Generating students....")
    students = generateStudents(numberStudents, allUniversities, allAbilities, allLanguages, allWorkExperiences,
                                allVolunteerExperiences, allStudents)

    print("Generating projects....")
    projects = generateProjects(numberProjects, allUniversities, allAbilities, allLanguages, allWorkExperiences,
                                allVolunteerExperiences, companies)

    print("Writing JSON in studentsProjectsData.json....")
    with open("../data/studentsProjectsData.json", "w") as file:
        json.dump({"students": students,
                   "projects": projects}, file, indent=4)

    print("Done.")


if __name__ == "__main__":
    main()

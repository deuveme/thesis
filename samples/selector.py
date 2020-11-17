import json
import random as r
from progress.bar import Bar


def main():
    print("Importing data....")
    try:
        with open("../data/optionsData.json") as dataFile:
            data = json.load(dataFile)
            numberOptions = data['numberOptions']
            students = data['results']
        print("Data imported.")

        studentWithOptionSelected = []
        progressBar = Bar("Selecting option for each student:", max=len(students))
        for student in students:
            studentWithOptionSelected.append({"studentId": student['studentId'],
                                              "studentAverageMark": student['studentAverageMark'],
                                              "optionSelected": student['projectOptions'][r.randint(0, numberOptions - 1)]})
            progressBar.next()
        progressBar.finish()

        print("Done.")

        print("Writing JSON in optionsSelectedData.json....")
        with open("../data/optionsSelectedData.json", "w") as file:
            json.dump({"results": sorted(studentWithOptionSelected,
                                         key=lambda studentOption: -studentOption['studentAverageMark'])}, file, indent=4)

        print("Done.")

    except OSError:
        print("File optionsData.json not found in ../data/")


if __name__ == "__main__":
    main()

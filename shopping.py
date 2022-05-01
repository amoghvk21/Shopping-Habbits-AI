import csv
from datetime import datetime
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(f"{filename}.csv") as f:

        evidence = []
        labels = []

        reader = csv.reader(f)
        next(reader)

        for row in f:
            r = row.split(",")
            
            if r[10] == "Feb":
                r[10] = 1
            elif r[10] == "Mar":
                r[10] = 2
            elif r[10] == "May":
                r[10] = 4
            elif r[10] == "June":
                r[10] = 5
            elif r[10] == "Jul":
                r[10] = 6
            elif r[10] == "Aug":
                r[10] = 7
            elif r[10] == "Sep":
                r[10] = 8
            elif r[10] == "Oct":
                r[10] = 9
            elif r[10] == "Nov":
                r[10] = 10
            elif r[10] == "Dec":
                r[10] = 11
            else:
                raise Exception(f"Date error: {r[10]}")

            r[17] = r[17][:len(r[17])-1]
            r[0] = int(r[0])
            r[1] = float(r[1])
            r[2] = int(r[2])
            r[3] = float(r[3])
            r[4] = int(r[4])
            r[5] = float(r[5])
            r[6] = float(r[6])
            r[7] = float(r[7])
            r[8] = float(r[8])
            r[9] = float(r[9])
            r[11] = int(r[11])
            r[12] = int(r[12])
            r[13] = int(r[13])
            r[14] = int(r[14])
            r[15] = 1 if r[15] == 'Returning_Visitor' else 0
            r[16] = True if r[16] == "TRUE" else False
            r[17] = True if r[17] == "TRUE" else False
            evidence.append(r[:17])
            labels.append(r[17])
            #print(r)
    
        return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    no_of_correct_true = 0
    no_of_correct_false = 0    
    total_no_of_false = labels.count(False)
    total_no_of_true = labels.count(True)

    for label, prediction in zip(labels, predictions):
        if (label == False) and (prediction == False):
            no_of_correct_false += 1
        elif (label == True) and (prediction == True):
            no_of_correct_true += 1

    sensitivity = no_of_correct_true/total_no_of_true
    specificity = no_of_correct_false/total_no_of_false

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()

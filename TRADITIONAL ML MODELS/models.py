from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def get_model(name):
    if name == "RF":
       return RandomForestClassifier(n_estimators=100, random_state=42)

    elif name == "NB":
        return GaussianNB()

    elif name == "SVM":
        return LinearSVC()

    elif name == "LR":
        return LogisticRegression(max_iter=1000)

    elif name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)

    else:
        raise ValueError("Invalid model name")
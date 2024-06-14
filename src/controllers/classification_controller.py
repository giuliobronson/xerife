import numpy as np
import cv2
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

pipe = Pipeline(steps=[
    ('scaler', StandardScaler()), 
    ('pca', PCA(n_components=0.95))
])

knn = KNeighborsClassifier(n_neighbors=5)

def fetch_data(directory="./src/data/faces", parent="", data=[], targets=[]):
    for element in os.listdir(directory):
        if os.path.isdir(f"{directory}/{element}"):
            data, targets = fetch_data(directory=f"{directory}/{element}", parent=element, data=data, targets=targets)
        else:
            image = cv2.imread(f"{directory}/{element}", cv2.IMREAD_GRAYSCALE)
            targets.append(parent)
            data.append(np.array(image).reshape(-1))
    return data, targets

def train():
    data, targets = fetch_data()
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.3, random_state=42)
    X_train_transformed = pipe.fit_transform(X_train, y_train) 
    X_test_transformed  = pipe.transform(X_test)
    knn.fit(X_train_transformed, y_train)
    y_pred = knn.predict(X_test_transformed)
    return {"Accuracy": accuracy_score(y_test, y_pred)}

def test(input):
    X_pipelined = pipe.transform(input)
    return knn.predict(X_pipelined)[0]

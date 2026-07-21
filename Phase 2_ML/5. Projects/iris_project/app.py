import pickle
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE, "model", "iris_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE, "model", "scaler.pkl"), "rb"))
labels = pickle.load(open(os.path.join(BASE, "model", "labels.pkl"), "rb"))

print("\n----- IRIS FLOWER CLASSIFIER -----\n")

vals = []
fields = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

for f in fields:
    vals.append(float(input(f"{f}: ")))

data = np.array([vals])
data = scaler.transform(data)

pred = model.predict(data)[0]

print("\nFlower Type:")
print(labels[pred])

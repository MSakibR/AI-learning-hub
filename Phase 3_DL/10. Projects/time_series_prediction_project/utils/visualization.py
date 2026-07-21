import matplotlib.pyplot as plt


def plot_predictions(actual, predicted, title="Time-Series Prediction"):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual", color="blue")
    plt.plot(predicted, label="Predicted", color="red")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

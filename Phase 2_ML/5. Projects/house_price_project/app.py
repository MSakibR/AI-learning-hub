import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle
import os

# -------------------------------
# Load Model & Scaler
# -------------------------------
model_path = "model/house_model.pkl"
scaler_path = "model/scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    messagebox.showerror(
        "Error", "Model or Scaler not found. Please run train.py first."
    )
    raise FileNotFoundError("Model or Scaler not found. Please run train.py first.")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# Fields & Entries
# -------------------------------
fields = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

entries = {}


# -------------------------------
# Prediction Function
# -------------------------------
def predict():
    try:
        # Get values from entry widgets
        values = []
        for f in fields:
            val = entries[f].get().strip()
            if val == "":
                messagebox.showerror("Error", f"Please enter a value for {f}")
                return
            values.append(float(val))

        # Scale & predict
        values_scaled = scaler.transform([values])
        pred = model.predict(values_scaled)[0]

        # Show prediction
        messagebox.showinfo(
            "Prediction",
            f"Estimated House Price: ${pred * 100000:.2f}",  # convert to dollars
        )

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers only")
    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{e}")


# -------------------------------
# Tkinter GUI Setup
# -------------------------------
root = tk.Tk()
root.title("🏠 House Price Prediction App")
root.geometry("400x400")  # Optional: set window size

# Input fields
for field in fields:
    row = tk.Frame(root)
    label = tk.Label(row, width=20, text=field + ": ", anchor="w")
    entry = tk.Entry(row)
    row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    label.pack(side=tk.LEFT)
    entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
    entries[field] = entry

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=10)

tk.Button(btn_frame, text="Predict", command=predict, bg="green", fg="white").pack(
    side=tk.RIGHT, padx=5
)
tk.Button(btn_frame, text="Quit", command=root.quit, bg="red", fg="white").pack(
    side=tk.RIGHT, padx=5
)

root.mainloop()

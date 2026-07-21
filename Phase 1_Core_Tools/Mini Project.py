# ---------------------------------------
# 📊 Student Performance Analysis Project
# Uses NumPy + Pandas + Matplotlib
# ---------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# 1️⃣ Generate Data (NumPy)
# --------------------------

# Create random marks for 10 students (50–100)
np.random.seed(42)
marks = np.random.randint(50, 101, 10)

# Student names
names = [
    "Saki",
    "Hasan",
    "Nusrat",
    "Rafi",
    "Tania",
    "Sakib",
    "Rahim",
    "Karim",
    "Asha",
    "Nabila",
]

# --------------------------
# 2️⃣ Create DataFrame (Pandas)
# --------------------------

df = pd.DataFrame({"Name": names, "Marks": marks})

# Add Pass/Fail column
df["Status"] = df["Marks"].apply(lambda m: "Pass" if m >= 60 else "Fail")

# Print the table
print("\n📌 Student Table:")
print(df)

# Summary statistics
print("\n📊 Summary:")
print("Average Marks:", df["Marks"].mean())
print("Max Marks:", df["Marks"].max())
print("Min Marks:", df["Marks"].min())

# --------------------------
# 3️⃣ Visualization (Matplotlib)
# --------------------------

# Line Plot of Student Marks
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(df["Name"], df["Marks"], marker="o")
plt.title("Student Marks Line Plot")
plt.xticks(rotation=45)

# Bar Chart
plt.subplot(1, 3, 2)
plt.bar(df["Name"], df["Marks"])
plt.title("Marks Bar Chart")
plt.xticks(rotation=45)

# Pie Chart: Pass vs Fail
status_counts = df["Status"].value_counts()

plt.subplot(1, 3, 3)
plt.pie(status_counts, labels=status_counts.index, autopct="%1.1f%%")
plt.title("Pass vs Fail")

plt.tight_layout()
plt.show()

# --------------------------
# 4️⃣ Save the Data
# --------------------------

df.to_csv("student_data.csv", index=False)
print("\n💾 File Saved: student_data.csv")

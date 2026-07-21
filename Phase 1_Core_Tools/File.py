try:
    username = input("Enter your name: ")
    with open("log.txt", "a") as f:
        f.write(f"User {username} logged in\n")
    print("Log saved!")
except Exception as e:
    print("Error:", e)

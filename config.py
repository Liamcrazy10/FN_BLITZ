import os
import time

# Portion of screen to be captured (This forms a square/rectangle around the center of screen)
screenShotHeight = 320
screenShotWidth = 320

# Use "left" or "right" for the mask side depending on where the interfering object is, useful for 3rd player models or large guns
useMask = False
maskSide = "left"
maskWidth = 80
maskHeight = 200

# Autoaim mouse movement amplifier
os.system('cls' if os.name == 'nt' else 'clear')

while True:
    data = input("How much movement speed would you like? (normal is 0.4): ").strip()

    try:
        speed = float(data)
    except ValueError:
        print("Invalid number. Please retry.")
        continue

    # Scale to work with ints internally (0.4 -> 4)
    speed_int = int(speed * 10)

    allowed_speeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if speed_int in allowed_speeds:
        print("Changing config...")
        break
    else:
        print("Invalid number. Please retry.")

aaMovementAmp = speed_int

# Person Class Confidence
os.system('cls' if os.name == 'nt' else 'clear')

confidence = 0.1

# What key to press to quit and shutdown the autoaim
os.system('cls' if os.name == 'nt' else 'clear')
# Change q To Change What Button To Close The Hack With
aaQuitKey = "q"

# If you want to main slightly upwards towards the head
os.system('cls' if os.name == 'nt' else 'clear')
while True:
    headshot = input("headshot mode preference (True/False): ")

    if headshot == 'true':
        headshot_mode = True
        print("Headshot mode enabled.")
        break
    elif data == 'false':
        headshot_mode = False
        print("Headshot mode disabled.")
        break
    else:
        print("Invalid input. Please enter 'True' or 'False'.")

headshot_mode = (headshot)

# Displays the Corrections per second in the terminal
cpsDisplay = True

# Set to True if you want to get the visuals
visuals = False

# Smarter selection of people
centerOfScreen = True

# ONNX ONLY - Choose 1 of the 3 below
# 1 - CPU
# 2 - AMD
# 3 - NVIDIA
onnxChoice = 3

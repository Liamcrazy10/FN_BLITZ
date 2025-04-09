# Portion of screen to be captured (This forms a square/rectangle around the center of screen)
screenShotHeight = 320
screenShotWidth = 320

# Use "left" or "right" for the mask side depending on where the interfering object is, useful for 3rd player models or large guns
useMask = False
maskSide = "left"
maskWidth = 80
maskHeight = 200

# Autoaim mouse movement amplifier

data = input("how much Movement speed would you like, (normal is 0.4): ")
print (data)

aaMovementAmp = (data)

# Person Class Confidence
confidence = 0.1

# What key to press to quit and shutdown the autoaim

quitkey = input("what key do you want to quit the cheat on: ")

aaQuitKey = (quitkey)

# If you want to main slightly upwards towards the head

headshot = input("do you want to have headshot mode on? True/False: ")

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
import mediapipe
print("MediaPipe file:", mediapipe.__file__)
try:
    print(dir(mediapipe))
    import mediapipe.python.solutions as solutions
    print("Direct import success")
except Exception as e:
    print("Error:", e)

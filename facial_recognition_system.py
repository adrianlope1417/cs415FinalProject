
import cv2
import os
import numpy as np
import pickle

# Directory to save profile data
PROFILE_DIR = "profiles"
os.makedirs(PROFILE_DIR, exist_ok=True)

# Function to load profiles
def load_profiles():
    if os.path.exists(f"{PROFILE_DIR}/profiles.pkl"):
        with open(f"{PROFILE_DIR}/profiles.pkl", "rb") as f:
            return pickle.load(f)
    return {}

# Function to save profiles
def save_profiles(profiles):
    with open(f"{PROFILE_DIR}/profiles.pkl", "wb") as f:
        pickle.dump(profiles, f)

# Load existing profiles
profiles = load_profiles()

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def capture_face(profile_name):
    """Captures a face and associates it with a profile."""
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Face")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Capture Face", frame)

        if len(faces) > 0:
            print("Face detected. Press 's' to save or 'q' to quit.")
        else:
            print("No face detected. Adjust position and lighting.")

        key = cv2.waitKey(1)
        if key == ord("s"):
            # Save the first detected face
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))
            profiles[profile_name] = face_img
            save_profiles(profiles)
            print(f"Profile {profile_name} saved.")
            break
        elif key == ord("q"):
            print("Capture cancelled.")
            break

    cam.release()
    cv2.destroyAllWindows()

def authenticate():
    """Authenticate a user by matching their face."""
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Authenticate")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Authenticate", frame)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))

            # Compare with profiles
            for name, saved_face in profiles.items():
                diff = np.linalg.norm(face_img - saved_face)
                if diff < 1000:  # Adjust threshold as needed
                    print(f"Authenticated as {name}")
                    cam.release()
                    cv2.destroyAllWindows()
                    return

            print("No match found. Try again.")
        
        key = cv2.waitKey(1)
        if key == ord("q"):
            print("Authentication cancelled.")
            break

    cam.release()
    cv2.destroyAllWindows()

# Main menu
def main_menu():
    while True:
        print("\nMain Menu")
        print("1. Add New Profile")
        print("2. Authenticate")
        print("3. Exit")

        choice = input("Enter your choice: ")
        if choice == "1":
            profile_name = input("Enter profile name: ")
            if profile_name in profiles:
                print("Profile already exists. Choose another name.")
            else:
                capture_face(profile_name)
        elif choice == "2":
            if not profiles:
                print("No profiles available. Add profiles first.")
            else:
                authenticate()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Try again.")

# Run the program
if __name__ == "__main__":
    main_menu()

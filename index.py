import cv2

# Load both frontal & profile classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

# Open webcam
video_cap = cv2.VideoCapture(0)

# Photo counter (unique file names ke liye)
photo_count = 0

while True:
    ret, frame = video_cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect frontal and profile faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Combine detections (frontal + side)
    all_faces = list(faces) + list(profiles)

    # Draw only one box per detection
    for (x, y, w, h) in all_faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Only Green Box

    # Show live video with detection
    cv2.imshow("Video_live", frame)

    # Check for key press
    key = cv2.waitKey(10)

    # Press 's' to save photo
    if key == ord('s'):
        photo_count += 1
        filename = f"face_capture_{photo_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Photo saved as {filename}")

    # Press 'a' to exit
    elif key == ord('a'):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()

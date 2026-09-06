import cv2
from deepface import DeepFace
import threading

# Load Haar cascade for fast, precise face cropping
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
current_emotion = "Show me an expression!"
analyzing = False

def get_emotion(face_img):
    global current_emotion, analyzing
    try:
        # Analyze only the cropped face region
        analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, silent=True)
        if isinstance(analysis, list):
            analysis = analysis[0]
        current_emotion = analysis['dominant_emotion'].upper()
    except Exception:
        pass
    analyzing = False

print("Supercharged emotion detector running... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a clean box around your face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 204), 2)
        
        # Extract the exact face crop
        face_crop = frame[y:y+h, x:x+w]

        # Trigger background analysis on the cropped face
        if not analyzing and face_crop.size > 0:
            analyzing = True
            threading.Thread(target=get_emotion, args=(face_crop.copy(),), daemon=True).start()

    # Display the emotion on screen
    cv2.putText(frame, f"Emotion: {current_emotion}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 204), 2)

    cv2.imshow('Selfie Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
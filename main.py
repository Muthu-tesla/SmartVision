# Smart Vision — Object Detection with Voice + Text Output
# Developed by Muthu 🚀

from ultralytics import YOLO
import cv2
import pyttsx3
import time

# -------------------- SETUP --------------------
print("🚀 Initializing Smart Vision System...")
model = YOLO("yolov8n.pt")   # small YOLOv8 model (fast)
cap = cv2.VideoCapture(0)    # open webcam (0 = default camera)

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

# Variables
last_speak_time = 0
spoken_objects = set()

# -------------------- SPEAK FUNCTION --------------------
def speak_objects(objects):
    global last_speak_time, spoken_objects
    current_time = time.time()
    # Speak only if new objects or after 5 seconds
    if (objects != spoken_objects) and (current_time - last_speak_time > 3):
        sentence = "I see " + ", ".join(objects)
        print("🗣️ Speaking:", sentence)
        engine.say(sentence)
        engine.runAndWait()
        last_speak_time = current_time
        spoken_objects = objects

# -------------------- MAIN LOOP --------------------
print("\n✅ Smart Vision started successfully!")
print("Press 'Q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not accessible. Exiting...")
        break

    # Run YOLO detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Collect detected object names
    detected_objects = set()
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        detected_objects.add(label)

    # Print detections
    if detected_objects:
        print("🧠 Detected:", ", ".join(detected_objects))
        speak_objects(detected_objects)

    # Display the camera feed
    cv2.imshow("🪄 Smart Vision - Live Object Detection", annotated_frame)

    # Exit on pressing 'Q' or 'q'
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        print("\n👋 Exiting Smart Vision...")
        break

# -------------------- CLEANUP --------------------
cap.release()
cv2.destroyAllWindows()
engine.stop()
print("✅ Camera released. Goodbye!")

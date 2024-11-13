from flask import Flask, render_template, Response, redirect, url_for
import cv2
from keras.models import model_from_json
import numpy as np

app = Flask(__name__)

# Load the model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector_CNN.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 100, 100, 1)
    return feature / 255.0

def generate_frames():
    webcam = cv2.VideoCapture(0)
    while True:
        success, frame = webcam.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (p, q, r, s) in faces:
                face = gray[q:q+s, p:p+r]
                cv2.rectangle(frame, (p, q), (p+r, q+s), (255, 0, 0), 2)
                face = cv2.resize(face, (100, 100))
                img = extract_features(face)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(frame, prediction_label, (p, q-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/analysis')
def index():
    return render_template('index.html')

@app.route('/')
def analysis():
    return render_template('analysis.html')

@app.route('/emotion_recognition')
def emotion_recognition():
    return render_template('emotion_recognition.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

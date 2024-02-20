import cv2
from flask import Flask , Response , render_template, jsonify
from flask_debug import Debug
import mediapipe as mp 
import os
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score # Accuracy metrics
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
Debug(app)

mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()



up = False 

def curl():
    landmarks = ['class']
    for val in range(1,33+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    with open('app/Barbellcurl/barbell_curl.pkl', 'rb') as f:
        model = pickle.load(f)
    cap = cv2.VideoCapture(0)
    counter = 0
    current_stage = ''

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            ret, frame = cap.read()

            #Recolor feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            #make Detections
            result = pose.process(image)

            #Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            
            try: 
                row = np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten().tolist()
                X = pd.DataFrame([row], columns = landmarks[1:]) 
                body_language_prob = model.predict_proba(X)[0]
                body_language_class = model.predict(X)[0]
                print(body_language_class, body_language_prob) 

                if body_language_class =="down" and body_language_prob[body_language_prob.argmax()] >= 0.7: 
                    current_stage = "down" 
                elif current_stage == "down" and body_language_class == "up" and body_language_prob[body_language_prob.argmax()] >= 0.7:
                    current_stage = "up" 
                    counter += 1
                    print(current_stage) 
                

                # Get status box
                cv2.rectangle(image, (0,0), (250,60), (245,117,16), -1)

                # Display Class
                cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

                # Display Probability
                cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image,str(round(body_language_prob[np.argmax(body_language_prob)],2)), (10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

                # Display Counter
                cv2.putText(image, 'COUNT', (180,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (175,40), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

            
            except Exception as e:
                pass
            imgencode = cv2.imencode('.jpg',image)[1]
            frame = imgencode.tobytes()
            yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/selection")
def selection():
    return render_template("selections.html")

@app.route("/curl")
def curl_cam():
    return Response(curl(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/barbellcurl")
def barbellcurlcurl():
    return render_template("barbellcurl.html")

if __name__ == "__main__":
    app.run(debug=True)

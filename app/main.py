import cv2
from flask import Flask , Response , render_template, jsonify
from flask_debug import Debug
import mediapipe as mp 
import os
import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score # Accuracy metrics
# from sklearn.model_selection import train_test_split
import imutils

app = Flask(__name__)
Debug(app)

mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def calculate_angle(a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle 


def squat():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    form_reported = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            
            if success == True:
                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    # Calculate angle for right knee check
                    l_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

                    # Calculate angle for left knee check
                    r_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                
                    # Curl counter logic
                    if l_knee_angle < 60 and r_knee_angle < 60:
                        stage = "up fast & breath out"
                    if l_knee_angle >= 160 and r_knee_angle >= 160 and stage =='up fast & breath out':
                        stage= "down slow & breath in"
                        counter +=1
                        print(counter)
                            
                except:
                    pass
                
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Wrong":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)         
                
                imgencode = cv2.imencode('.jpg',image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break

def lift():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    form_reported = None

    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            
            if success == True:
                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    # Calculate angle for left hip check
                    l_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)

                    # Calculate angle for right hip check
                    r_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

                    # Calculate angle for right knee check
                    l_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

                    # Calculate angle for left knee check
                    r_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    
                    # deadlift counter logic
                    if l_knee_angle < 40 and r_knee_angle < 40:
                        form_reported = "Hips too low"
                        stage = "Hips too low"
                    if 120 < l_knee_angle < 150 and 120 < r_knee_angle < 150 and l_hip_angle < 75 and r_hip_angle < 75:
                        form_reported = "Rounding back"
                        stage = "Rounding back"
                    if l_knee_angle <= 120 and r_knee_angle <= 120 and l_hip_angle < 55 and r_hip_angle < 55:
                        stage = "Up fast & Breath out"
                    if l_knee_angle >= 160 and r_knee_angle >= 160 and l_hip_angle >= 160 and r_hip_angle >= 160 and stage =='Up fast & Breath out':
                        stage= "Down slow & Breath in"
                        counter +=1
                        print(counter)
                            
                except:
                    pass
                
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Hips too low" or form_reported == "Rounding back":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)           

                imgencode = cv2.imencode('.jpg', image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break

def BSLR():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0
    stage = None
    form_reported = None
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()

            if success == True:
                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    # Calculate left shoulder angle
                    left_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)

                    # Get coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    # Calculate right shoulder angle
                    right_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)

                    # Counter logic
                    if left_shoulder_angle < 100 and right_shoulder_angle < 100:
                        correct_form = True
                        form_reported = "Good"
                    if left_shoulder_angle >= 110 or right_shoulder_angle >=110:
                        correct_form = False
                        form_reported = "Wrong"
                        stage = "Too high"
                    if left_shoulder_angle <= 20 and right_shoulder_angle <= 20 and correct_form:
                        stage = "Up fast & Breath out"
                    if left_shoulder_angle >= 90 and right_shoulder_angle >= 90 and stage =='Up fast & Breath out' and correct_form:
                        stage= "Down slow & Breath in"
                        counter +=1
                        print(counter)

                except:
                    pass

                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Wrong":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                imgencode = cv2.imencode('.jpg', image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break

def BSSP():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    form_reported = None
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if success == True:
                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    # Calculate left shoulder angle
                    left_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)

                    # Get coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    # Calculate right shoulder angle
                    right_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
                    
                     # Counter logic
                    if left_shoulder_angle > 20 and right_shoulder_angle > 20:
                        correct_form = True
                        form_reported = "Good"
                    if left_shoulder_angle <= 20 or right_shoulder_angle <=20:
                        correct_form = False
                        form_reported = "Wrong"
                        stage = "Too low"
                    if left_shoulder_angle <= 70 and right_shoulder_angle <= 70 and correct_form:
                        stage = "Up fast & Breath out"
                    if left_shoulder_angle >= 160 and right_shoulder_angle >= 160 and stage =='Up fast & Breath out' and correct_form:
                        stage= "Down slow & Breath in"
                        counter +=1
                        print(counter)
                            
                except:
                    pass
                
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Wrong":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                imgencode = cv2.imencode('.jpg', image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break     

def leftcurl():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    form_reported = None
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            
            if success == True:
                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    # Calculate angle for right arm biceps curl
                    g_r_bi_curl_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    # Calculate angle for failed right arm biceps curl (elbow forward)
                    ef_r_bi_curl_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
                    
                    # Curl counter logic
                    if ef_r_bi_curl_angle < 20:
                        correct_form = True
                        form_reported = "Good"
                    if ef_r_bi_curl_angle >= 20 :
                        correct_form = False
                        form_reported = "Wrong"
                        stage= "Keep elbows tight"
                    if g_r_bi_curl_angle > 130 and correct_form:
                        stage = "Up fast & Breath out"
                    if g_r_bi_curl_angle < 50 and stage =='Up fast & Breath out' and correct_form:
                        stage= "Down slow & Breath in"
                        counter +=1
                        print(counter)
                            
                except:
                    pass
                
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Wrong":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                imgencode = cv2.imencode('.jpg', image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break           

def rightcurl():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    form_reported = None
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            
            if success == True:
                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    # Calculate angle for right arm biceps curl
                    g_r_bi_curl_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                    # Calculate angle for failed right arm biceps curl (elbow forward)
                    ef_r_bi_curl_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
                    

                    # Curl counter logic
                    if ef_r_bi_curl_angle < 20:
                        correct_form = True
                        form_reported = "Good"
                    if ef_r_bi_curl_angle >= 20 :
                        correct_form = False
                        form_reported = "Wrong"
                        stage= "Keep elbows tight"
                    if g_r_bi_curl_angle > 130 and correct_form:
                        stage = "Up fast & Breath out"
                    if g_r_bi_curl_angle < 50 and stage =='Up fast & Breath out' and correct_form:
                        stage= "Down slow & Breath in"
                        counter +=1
                        print(counter)
                            
                except:
                    pass
                
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Wrong":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)               
                    
                imgencode = cv2.imencode('.jpg', image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break

def LCFR():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    form_reported = None
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            
            if success == True:
                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    # Calculate left elbow angle
                    left_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    # Calculate left shoulder angle
                    left_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
                    
                    # Counter logic
                    if left_elbow_angle > 135:
                        correct_form = True
                        form_reported = "Good"
                    if left_shoulder_angle > 100:
                        correct_form = False
                        form_reported = "Wrong"
                        stage = "Too high"
                    if left_shoulder_angle < 10 and correct_form:
                        stage = "Up fast & Breath out"
                    if left_shoulder_angle >= 70 and stage =='Up fast & Breath out' and correct_form:
                        stage= "Down slow & Breath in"
                        counter +=1
                        print(counter)
                            
                except:
                    pass
                
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Wrong":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                          
                imgencode = cv2.imencode('.jpg', image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break       

def LLR():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    form_reported = None
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()

            if success == True:

                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    # Calculate left shoulder angle
                    left_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)

                    # Counter logic
                    if left_shoulder_angle < 100:
                        correct_form = True
                        form_reported = "Good"
                    if left_shoulder_angle >= 110:
                        correct_form = False
                        form_reported = "Wrong"
                        stage = "Too high"
                    if left_shoulder_angle <= 20 and correct_form:
                        stage = "Up fast & Breath out"
                    if left_shoulder_angle >= 90 and  stage =='Up fast & Breath out' and correct_form:
                        stage= "Down slow & Breath in"
                        counter +=1
                        print(counter)

                except:
                    pass

                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Wrong":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                imgencode = cv2.imencode('.jpg', image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break   

def LLDR():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    form_reported = None
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()

            if success == True:

                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                    # Calculate right shoulder angle
                    left_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    # Calculate right hip angle
                    left_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

                    # Counter logic
                    if 110 < left_hip_angle < 150:
                        correct_form = True
                        form_reported = "Good"
                    if left_hip_angle > 150:
                        correct_form = False
                        form_reported = "Wrong"
                        stage = "Hip too high"
                    if left_hip_angle <110:
                        correct_form = False
                        form_reported = "Wrong"
                        stage = "Hip too low"
                    if left_elbow_angle >= 160 and correct_form:
                        stage = "Up fast & Breath out"
                    if left_elbow_angle <= 90 and stage =='Up fast & Breath out' and correct_form:
                        stage= "Down slow & Breath in"
                        counter +=1
                        print(counter)

                except:
                    pass

                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Wrong":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                imgencode = cv2.imencode('.jpg', image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break   

def LSP():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    form_reported = None
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()

            if success == True:
            
                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    # Calculate left shoulder angle
                    left_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
                    
                    # Counter logic
                    if left_shoulder_angle > 20:
                        correct_form = True
                        form_reported = "Good"
                    if left_shoulder_angle <= 20:
                        correct_form = False
                        form_reported = "Wrong"
                        stage = "Too low"
                    if left_shoulder_angle <= 70 and correct_form:
                        stage = "Up fast & Breath out"
                    if left_shoulder_angle >= 160 and stage =='Up fast & Breath out' and correct_form:
                        stage= "Down slow & Breath in"
                        counter +=1
                        print(counter)
                            
                except:
                    pass
                
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Wrong":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                imgencode = cv2.imencode('.jpg', image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break

def RCFR():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    form_reported = None
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()

            if success == True:
            
                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    # Calculate right elbow angle
                    right_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                    # Calculate right shoulder angle
                    right_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
                    
                    # Counter logic
                    if right_elbow_angle > 135:
                        correct_form = True
                        form_reported = "Good"
                    if right_shoulder_angle > 100:
                        correct_form = False
                        form_reported = "Wrong"
                        stage = "Too high"
                    if right_shoulder_angle < 10 and correct_form:
                        stage = "Up fast & Breath out"
                    if right_shoulder_angle >= 70 and stage =='Up fast & Breath out' and correct_form:
                        stage= "Down slow & Breath in"
                        counter +=1
                        print(counter)
                            
                except:
                    pass
                
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Wrong":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                imgencode = cv2.imencode('.jpg', image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break               

def RLR():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    form_reported = None
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if success == True:
                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    # Calculate right shoulder angle
                    right_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)

                    # Counter logic
                    if right_shoulder_angle < 100:
                        correct_form = True
                        form_reported = "Good"
                    if right_shoulder_angle >=110:
                        correct_form = False
                        form_reported = "Wrong"
                        stage = "Too high"
                    if right_shoulder_angle <= 20 and correct_form:
                        stage = "Up fast & Breath out"
                    if right_shoulder_angle >= 90 and stage =='Up fast & Breath out' and correct_form:
                        stage= "Down slow & Breath in"
                        counter +=1
                        print(counter)

                except:
                    pass

                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Wrong":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                imgencode = cv2.imencode('.jpg', image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break

def RLDR():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    form_reported = None
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if success == True:
                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                    # Calculate right shoulder angle
                    right_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    # Calculate right hip angle
                    right_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)

                    # Counter logic
                    if 110 < right_hip_angle < 150:
                        correct_form = True
                        form_reported = "Good"
                    if right_hip_angle > 150:
                        correct_form = False
                        form_reported = "Wrong"
                        stage = "Hip too high"
                    if right_hip_angle <110:
                        correct_form = False
                        form_reported = "Wrong"
                        stage = "Hip too low"
                    if right_elbow_angle >= 160 and correct_form:
                        stage = "Up fast & Breath out"
                    if right_elbow_angle <= 90 and stage =='Up fast & Breath out' and correct_form:
                        stage= "Down slow & Breath in"
                        counter +=1
                        print(counter)

                except:
                    pass

                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Wrong":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                imgencode = cv2.imencode('.jpg', image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break

def RSP():
    cap = cv2.VideoCapture(0)
    # Curl counter variables
    counter = 0 
    stage = None
    form_reported = None
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if success == True:
                #Flip Cam
                frame = cv2.flip(frame,1)

                #Resize Cam to 1280x720
                frame = imutils.resize(frame,width=1280)
                frame = imutils.resize(frame,height=720)
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

                    # Calculate right shoulder angle
                    right_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
                    
                     # Counter logic
                    if right_shoulder_angle > 20:
                        correct_form = True
                        form_reported = "Good"
                    if right_shoulder_angle <= 20:
                        correct_form = False
                        form_reported = "Wrong"
                        stage = "Too low"
                    if right_shoulder_angle <= 70 and correct_form:
                        stage = "Up fast & Breath out"
                    if right_shoulder_angle >= 160 and stage =='Up fast & Breath out' and correct_form:
                        stage= "Down slow & Breath in"
                        counter +=1
                        print(counter)
                            
                except:
                    pass
                
                # Render curl counter
                # Setup status box
                cv2.rectangle(image, (0,0), (1280,73), (34,177,76), -1)
                cv2.rectangle(image, (0,0), (10,1111), (34,177,76), -1)
                cv2.rectangle(image, (1270,1280), (950,0), (34,177,76), -1)
                
                if form_reported == "Wrong":
                    cv2.rectangle(image, (0,0), (1280,73), (0,0,255), -1)
                    cv2.rectangle(image, (0,0), (10,1111), (0,0,255), -1)
                    cv2.rectangle(image, (1270,1280), (950,0), (0,0,255), -1)

                # Rep data
                cv2.putText(image,str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, stage, 
                            (130,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                imgencode = cv2.imencode('.jpg', image)[1]
                frame = imgencode.tobytes()
                yield(b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+ frame +b'\r\n')
            else:
                break


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/selection")
def selection():
    return render_template("selections.html")

@app.route("/leftcurl")
def leftcurl_cam():
   return Response(leftcurl(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/rightcurl")
def rightcurl_cam():
   return Response(rightcurl(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/squat")
def squat_cam():
    return Response(squat(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/lift")
def lift_cam():
    return Response(lift(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/BSLR")
def BSLR_cam():
    return Response(BSLR(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/BSSP")
def BSSP_cam():
    return Response(BSSP(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/LCFR")
def LCFR_cam():
    return Response(LCFR(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/LLR")
def LLR_cam():
    return Response(LLR(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/LLDR")
def LLDR_cam():
    return Response(LLDR(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/LSP")
def LSP_cam():
    return Response(LSP(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/RCFR")
def RCFR_cam():
    return Response(RCFR(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/RLR")
def RLR_cam():
    return Response(RLR(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/RLDR")
def RLDR_cam():
    return Response(RLDR(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/RSP")
def RSP_cam():
    return Response(RSP(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/leftdumbbellcurl")
def leftdumbbellcurl():
    return render_template("leftdumbbellcurl.html")

@app.route("/rightdumbbellcurl")
def rightdumbbellcurl():
    return render_template("rightdumbbellcurl.html")

@app.route("/barbellsquat")
def barbellsquat():
    return render_template("barbellsquat.html")

@app.route("/deadlift")
def deadlift():
    counter = 23
    return render_template("deadlift.html",counter=counter)

@app.route("/Both_Side_Lateral_Raise")
def Both_Side_Lateral_Raise():
    return render_template("Both_Side_Lateral_Raise.html")

@app.route("/Both_Side_Shoulder_Press")
def Both_Side_Shoulder_Press():
    return render_template("Both_Side_Shoulder_Press.html")

@app.route("/Left_Chest_Front_Raise")
def Left_Chest_Front_Raise():
    return render_template("Left_Chest_Front_Raise.html")

@app.route("/Left_Lateral_Raise")
def Left_Lateral_Raise():
    return render_template("Left_Lateral_Raise.html")

@app.route("/Left_Leaning_Dumbbell_Row")
def Left_Leaning_Dumbbell_Row():
    return render_template("Left_Leaning_Dumbbell_Row.html")

@app.route("/Left_Shoulder_Press")
def Left_Shoulder_Press():
    return render_template("Left_Shoulder_Press.html")

@app.route("/Right_Chest_Front_Raise")
def Right_Chest_Front_Raise():
    return render_template("Right_Chest_Front_Raise.html")

@app.route("/Right_Lateral_Raise")
def Right_Lateral_Raise():
    return render_template("Right_Lateral_Raise.html")

@app.route("/Right_Leaning_Dumbbell_Row")
def Right_Leaning_Dumbbell_Row():
    return render_template("Right_Leaning_Dumbbell_Row.html")

@app.route("/Right_Shoulder_Press")
def Right_Shoulder_Press():
    return render_template("Right_Shoulder_Press.html")

@app.route("/AboutUs")
def AboutUs():
    return render_template("AboutUs.html")

@app.route("/WhatIsSWUFit")
def WhatIsSWUFit():
    return render_template("WhatIsSWUFit.html")

if __name__ == "__main__":
    app.run(debug=True)

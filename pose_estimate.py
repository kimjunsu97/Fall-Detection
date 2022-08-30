import cv2
import mediapipe as mp
import numpy as np
import time
import math
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]) # 각도 재기
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def Euclidean_distance(x1,x2,y1,y2):
    return (((x1-x2)**2)+((y1-y2)**2))**(1/2)

def cal_Center(x,y):
    N=len(x)
    A,Cx,Cy = 0,0,0
    
    for i in range(0,N-1):
        A += ((x[i]*y[i+1]-x[i+1]*y[i]))
    A+= ((x[N-1]*y[0]-x[0]*y[N-1]))    
    for i in range(0,N-1):
        Cx += ((x[i]+x[i+1])*(x[i]*y[i+1]-x[i+1]*y[i]))
    Cx += ((x[N-1]+x[0])*(x[N-1]*y[0]-x[0]*y[N-1]))    
    
    for i in range(0, N-1):
        Cy += ((y[i]+y[i+1])*(x[i]*y[i+1]-x[i+1]*y[i]))
    Cy += ((y[N-1]+y[0])*(x[N-1]*y[0]-x[0]*y[N-1]))    
    
    A /= 2
    Cx /= (6*A)
    Cy /= (6*A)
    
    Center = (Cx,Cy)
    return Center

cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0 
stage = None
prev_frame_time = 0
new_frame_time = 0
prev_frame_centroid = (0,0)
new_frame_centroid = (0,0)
arr_x = 0
arr_y = 0
distance = 0
Fall = 0
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
          
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        
        fps = int(fps)
        fps=str(fps)
        
        #distance
        new_frame_centroid = np.array([arr_x,arr_y])
        distance =  Euclidean_distance(new_frame_centroid[0],prev_frame_centroid[0],new_frame_centroid[1],prev_frame_centroid[1])
        prev_frame_centroid = new_frame_centroid
    
        
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, fps, (500,70), font, 3, (100,255,0),3, cv2.LINE_AA)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get BoundBox, R
            x_list = []
            y_list = []
            z_list = []
            for i in landmarks:
                x_list.append(i.x)
                y_list.append(i.y)
                z_list.append(i.z)
            x_minVal, x_maxVal, _, _ = cv2.minMaxLoc(np.array(x_list))
            y_minVal, y_maxVal, _, _ = cv2.minMaxLoc(np.array(y_list))
            height, width = y_maxVal-y_minVal, x_maxVal-x_minVal
            
            R = width/height
            
           # Get body coordinates
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle= [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate angle
            angle_right_elbow = calculate_angle(right_wrist, right_elbow, right_shoulder)
            angle_left_elbow = calculate_angle(left_wrist, left_elbow, left_shoulder)

            angle_right_shoulder1 = calculate_angle(right_elbow, right_shoulder, left_shoulder)
            angle_right_shoulder2 = calculate_angle(right_elbow, right_shoulder, right_hip)
            angle_right_shoulder3 = calculate_angle(left_shoulder, right_shoulder, right_hip)

            angle_left_shoulder1 = calculate_angle(left_elbow, left_shoulder, right_shoulder)
            angle_left_shoulder2 = calculate_angle(left_elbow, left_shoulder, left_hip)
            angle_left_shoulder3 = calculate_angle(right_shoulder, left_shoulder, left_hip)

            angle_right_hip1 = calculate_angle(right_shoulder, right_hip, left_hip)
            angle_right_hip2 = calculate_angle(right_shoulder, right_hip, right_knee)
            angle_right_hip3 = calculate_angle(left_hip, right_hip, right_knee)

            angle_left_hip1 = calculate_angle(left_shoulder, left_hip, right_hip)
            angle_left_hip2 = calculate_angle(left_shoulder, left_hip, left_knee)
            angle_left_hip3 = calculate_angle(right_hip, left_hip, left_knee)

            angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)
            angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
            
            # Get face coordinates
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            right_eye_inner = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y]
            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
            right_eye_outer = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y]

            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            left_eye_inner = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y]
            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
            left_eye_outer = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]

            left_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x,landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
            right_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
            
            # Get face centroid
            center_x = [right_ear[0],right_eye[0],left_eye[0],left_ear[0],left_shoulder[0],right_shoulder[0]]
            center_y = [right_ear[1],right_eye[1],left_eye[1],left_ear[1],left_shoulder[1],right_shoulder[1]]

            arr_x,arr_y = cal_Center(center_x,center_y) 

            cv2.circle(image, tuple(np.multiply((arr_x,arr_y), [frame_width, frame_height]).astype(int)),10, (0,0,255), -1)

            # Visualize angle
            '''
            cv2.putText(image, str(angle_right_elbow), 
                           tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            '''
            cv2.rectangle(image, (0,0), (200,500), (0,0,0), -1)
            cv2.putText(image, 'left_elbow :{:.2f}'.format(angle_left_elbow), 
                           (10,90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, 'left_shoulder1 :{:.2f}'.format(angle_left_shoulder1), 
                           (10,110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, 'left_shoulder2 :{:.2f}'.format(angle_left_shoulder2), 
                           (10,130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, 'left_hip1 :{:.2f}'.format(angle_left_hip1), 
                           (10,150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, 'left_hip2 :{:.2f}'.format(angle_left_hip2), 
                           (10,170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, 'left_knee :{:.2f}'.format(angle_left_knee), 
                           (10,190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, 'right_elbow :{:.2f}'.format(angle_right_elbow), 
                           (10,210), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, 'right_shoulder1 :{:.2f}'.format(angle_right_shoulder1), 
                           (10,230), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, 'right_shoulder2 :{:.2f}'.format(angle_right_shoulder2), 
                           (10,250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, 'right_hip1 :{:.2f}'.format(angle_right_hip1), 
                           (10,270), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, 'right_hip2 :{:.2f}'.format(angle_right_hip2), 
                           (10,290), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(image, 'right_knee :{:.2f}'.format(angle_right_knee), 
                           (10,310), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                               )
            cv2.putText(image, 'distance :{:.2f}'.format(distance), 
                           (10,330), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                               )
            cv2.putText(image, 'R :{:.2f}'.format(R), 
                           (10,350), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                               )
            cv2.rectangle(image, tuple(np.multiply((x_minVal,y_minVal), [frame_width, frame_height]).astype(int)),
                                       tuple(np.multiply((x_maxVal,y_maxVal), [frame_width, frame_height]).astype(int)), (0,255,0), 1)
            if distance >= 0.03 and R>=1:
                Fall = 1
                print(Fall)
            if Fall ==1:
                cv2.putText(image, 'Fall Down', (10,50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2, cv2.LINE_AA)
            
            '''
            cv2.putText(image, str(angle_right_elbow), 
                           (200.200),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            '''
            
            # Curl counter logic
#            if angle_left_elbow > 160:
#                stage = "down"
#            if angle_left_elbow < 30 and stage =='down':
#                stage="up"
#                counter +=1
#                print(counter)
                       
        except:
            pass
        
        # Render curl counter
        # Setup status box
#        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
#        cv2.putText(image, 'REPS', (15,12), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
#        cv2.putText(image, str(counter), 
#                    (10,60), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
#        cv2.putText(image, 'STAGE', (65,12), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
#        cv2.putText(image, stage, 
#                    (60,60), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        
    cap.release()
    cv2.destroyAllWindows()
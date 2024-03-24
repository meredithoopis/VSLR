import mediapipe as mp 
import numpy as np 
import pandas as pd 
import cv2 

def main(): 
    mp_drawing = mp.solutions.drawing_utils 
    mp_drawing_styles = mp.solutions.drawing_styles 
    mp_holistic = mp.solutions.holistic 
    
    def transform(): 
        frame_n = 0 
        frame = []
        type_ = []
        index = []
        x, y = [], []
        vid = cv2.VideoCapture('fake.MP4')
        #image = cv2.imread('cc.jpg')
        # cv2.imwrite('frame_1.jpg', vid.get(10))
        with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic: 
            while vid.isOpened(): 
                success, image = vid.read()
                if not success: 
                    break 
                frame_n +=1 

                image.flags.writeable = False 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                res = holistic.process(image) #image


                
                #Left hand 
                if (res.left_hand_landmarks is None): 
                    for i in range(21): 
                        frame.append(frame_n)
                        type_.append("left_hand")
                        index.append(i)
                        x.append(None)
                        y.append(None)
                        #z.append(None)
                else: 
                    for ind, val in enumerate(res.left_hand_landmarks.landmark):
                        frame.append(frame_n)
                        type_.append("left_hand")
                        index.append(ind)
                        x.append(val.x)
                        y.append(val.y)
                        #z.append(val.z)

                #Pose 
                if res.pose_landmarks is None: 
                    for i in range(33): 
                        frame.append(frame_n)
                        type_.append("pose")
                        index.append(i)
                        x.append(None)
                        y.append(None)
                        #z.append(None)
                else: 
                    for ind, val in enumerate(res.pose_landmarks.landmark):
                        frame.append(frame_n)
                        type_.append("pose")
                        index.append(ind)
                        x.append(val.x)
                        y.append(val.y)
                        #z.append(val.z)

                #Right hand 
                if res.right_hand_landmarks is None: 
                    for i in range(21): 
                        frame.append(frame_n)
                        type_.append("right_hand")
                        index.append(i)
                        x.append(None)
                        y.append(None)
                        #z.append(None)
                else: 
                    for ind, val in enumerate(res.right_hand_landmarks.landmark):
                        frame.append(frame_n)
                        type_.append("right_hand")
                        index.append(ind)
                        x.append(val.x)
                        y.append(val.y)
                        #z.append(val.z)

        return pd.DataFrame({
            "frame": frame, 
            "type": type_, 
            "index": index, 
            "x": x, 
            "y": y
            #"z": z
        }) 
    
    
    df = transform()
    df.to_csv('huhu.csv')


if __name__ == "__main__": 
    #import os
    #print(os.getcwd())
    main()


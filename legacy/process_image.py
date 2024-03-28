import mediapipe as mp 
import numpy as np 
import pandas as pd 
import cv2 

def main(): 
    mp_holistic = mp.solutions.holistic 
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    
    def transform(image): 
        frame = []
        type_ = []
        index = []
        x, y = [], []
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: 
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res = holistic.process(image_rgb)

            # Left hand 
            if res.left_hand_landmarks is None: 
                for i in range(21): 
                    frame.append(0)
                    type_.append("left_hand")
                    index.append(i)
                    x.append(None)
                    y.append(None)
                    #z.append(None)
            else: 
                for ind, val in enumerate(res.left_hand_landmarks.landmark):
                    frame.append(0)
                    type_.append("left_hand")
                    index.append(ind)
                    x.append(val.x)
                    y.append(val.y)
                    #z.append(val.z)

            # Pose 
            if res.pose_landmarks is None: 
                for i in range(33): 
                    frame.append(0)
                    type_.append("pose")
                    index.append(i)
                    x.append(None)
                    y.append(None)
                    #z.append(None)
            else: 
                for ind, val in enumerate(res.pose_landmarks.landmark):
                    frame.append(0)
                    type_.append("pose")
                    index.append(ind)
                    x.append(val.x)
                    y.append(val.y)
                    #z.append(val.z)

            # Right hand 
            if res.right_hand_landmarks is None: 
                for i in range(21): 
                    frame.append(0)
                    type_.append("right_hand")
                    index.append(i)
                    x.append(None)
                    y.append(None)
                    #z.append(None)
            else: 
                for ind, val in enumerate(res.right_hand_landmarks.landmark):
                    frame.append(0)
                    type_.append("right_hand")
                    index.append(ind)
                    x.append(val.x)
                    y.append(val.y)
                    #z.append(val.z)
            mp_drawing.draw_landmarks(
                image_rgb,
                res.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.
                get_default_pose_landmarks_style())
            cv2.imwrite('annotated_image.png', image_rgb)
        return pd.DataFrame({
            "frame": frame, 
            "type": type_, 
            "index": index, 
            "x": x, 
            "y": y
            #"z": z
        }) 
    
    # Load the static image
    image = cv2.imread('cc.jpg')
    
    df = transform(image)
    df.to_csv('huhu.csv')

if __name__ == "__main__": 
    main()

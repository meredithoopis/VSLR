from mediapipe import solutions
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import pandas as pd

from config import *
from functions import average_pose_to_hand_distance

# `NullPoint` is a class with 2 attribute `x`, `y` which have null value

# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# Image from top left to bottom right in Uniform(0,1)
# Dont need to care about mirror image, 'cause I will flip it eventually
# No legs 

# Only need to use function detect_video (for video) or detect_image (for image)

# All the function below only apply for 1 person. If you want to detect more,
# contact me and I will fix the code

frame = []
type_ = []
index = []
x, y = [], []


def _not_handedness(frame_n = 0, side = 'both'):
    """
    Add null value to the list
    """
    for i in range(21):
        frame.append(frame_n)
        type_.append("Left_hand")
        index.append(i)
        x.append(None)
        y.append(None)
    
    for i in range(21):
        frame.append(frame_n)
        type_.append("Right_hand")
        index.append(i)
        x.append(None)
        y.append(None)
        
def _get_single_hand_detail(landmarks, frame_n = 0, side='Left'):
    part = side + '_hand'
    for i, val in enumerate(landmarks):
        frame.append(frame_n)
        type_.append(part)
        index.append(i)
        x.append(np.float32(val.x))
        y.append(np.float32(val.y))
        


def _get_hand_detail(detect_hand, frame_n = 0, left_palm = None, right_palm = None):
    """
        There will be a lot of if-else because the hand-tracking is not correctly 
        specify left and right hand
        
        Manually match the correct palm to the hand landmarks
        The input order is Left_hand and then Right_hand
    """
    
    handedness = detect_hand.handedness
    hand_landmarks = detect_hand.hand_landmarks
    
    if not handedness:
        _not_handedness(frame_n)
    
    
            
    if left_palm == None: # If no palm is detected
        
        # If the is only one hand, there will be no condition to check, so 
        # I just put it in a correct order
        if len(handedness) == 1:
            side = handedness[0][0].category_name
            other_side = 'Right' if side == 'Left' else 'Left'
            
            if side == 'Left':
                _get_single_hand_detail(hand_landmarks[0], frame_n = frame_n, side = side)
                _get_single_hand_detail([NullPoint() for _ in range(20)], frame_n = frame_n, side = other_side)
            else:
                _get_single_hand_detail([NullPoint() for _ in range(20)], frame_n = frame_n, side = other_side)
                _get_single_hand_detail(hand_landmarks[0], frame_n = frame_n, side = side)
        
        # Checking if there is a 2 left hand and 2 right hand, since there is no 
        # other object to validate, I just mark the left-most hand is Left.
        
        if len(handedness) == 2:
            if handedness[0][0].category_name == handedness[1][0].category_name:
                first_hand = 'Left' if \
                    hand_landmarks[0][0].x < hand_landmarks[1][0].x \
                    else 'Right'
                    
                second_hand = 'Right' if first_hand == 'Left' else 'Left'
                    
            else:
                first_hand = handedness[0][0].category_name
                second_hand =  handedness[1][0].category_name
                
            if first_hand=='Left':
                _get_single_hand_detail(hand_landmarks[0], frame_n = frame_n, side = first_hand)
                _get_single_hand_detail(hand_landmarks[1], frame_n = frame_n, side = second_hand)
            else:
                _get_single_hand_detail(hand_landmarks[1], frame_n = frame_n, side = second_hand)
                _get_single_hand_detail(hand_landmarks[0], frame_n = frame_n, side = first_hand)
        return
    
    """This is where we have palm to cross-validate

        If there are 2 hands detected, the logic will detect the correct hand, regardless 
        of sides
        
        With 2 hands, I will calculate the distance of left palm to 2 other hands. 
        If the distance of the hand cross a certain threshold, it will swap hands
        
        With 1 hand, I will calculate the distance of that hand to 2 palms, and if 
        it crosses the threshold, it will be swaped from left to right and vice versa.
    """
    
    if len(hand_landmarks) == 2:  

        left_index = 0 if  handedness[1][0].category_name == 'Left' else 1
        right_index = np.abs(1-left_index)
        
        distance_l_l =  average_pose_to_hand_distance(left_palm, [hand_landmarks[left_index][0]])
        distance_l_r =  average_pose_to_hand_distance(left_palm, [hand_landmarks[right_index][0]])
        
        if distance_l_l/distance_l_r > MIN_DISTANCE_DIFF:
            _get_single_hand_detail(hand_landmarks[left_index], frame_n = frame_n, side = 'Right')
            _get_single_hand_detail(hand_landmarks[right_index], frame_n = frame_n, side = 'Left')
        else: 
            _get_single_hand_detail(hand_landmarks[left_index], frame_n = frame_n, side = 'Left')
            _get_single_hand_detail(hand_landmarks[right_index], frame_n = frame_n, side = 'Right')
    
    if len(hand_landmarks) == 1:
        side = handedness[0][0].category_name
        
        
        distance_l_h =  average_pose_to_hand_distance(left_palm, [hand_landmarks[0][0]])
        distance_r_h =  average_pose_to_hand_distance(right_palm, [hand_landmarks[0][0]])
        
        if side == 'Right':  
            if distance_r_h/distance_l_h > MIN_DISTANCE_DIFF: # Spot the different
                _get_single_hand_detail(hand_landmarks[0], frame_n = frame_n, side = 'Left')
                _get_single_hand_detail([NullPoint() for _ in range(20)], frame_n = frame_n, side = 'Right')
            else:
                _get_single_hand_detail([NullPoint() for _ in range(20)], frame_n = frame_n, side = 'Left')
                _get_single_hand_detail(hand_landmarks[0], frame_n = frame_n, side = 'Right')
                
        else:
            if distance_l_h/distance_r_h > MIN_DISTANCE_DIFF: # Spot the different
                _get_single_hand_detail([NullPoint() for _ in range(20)], frame_n = frame_n, side = 'Left')
                _get_single_hand_detail(hand_landmarks[0], frame_n = frame_n, side = 'Right')
            else:
                _get_single_hand_detail(hand_landmarks[0], frame_n = frame_n, side = 'Left')
                _get_single_hand_detail([NullPoint() for _ in range(20)], frame_n = frame_n, side = 'Right')
                

def _get_single_pose_detail(pose_landmarks, position, part, frame_n = 0):
    
    """
        Fill the data of specific body part landmarks
    """
    for pos in position:
        frame.append(frame_n)
        type_.append(part)
        index.append(pos)
        x.append(np.float32(pose_landmarks[0][pos].x))
        y.append(np.float32(pose_landmarks[0][pos].y))

def _get_pose_detail(detect_pose, frame_n = 0): # return left palm and right palm
    """
        Get coodinate of body part from left to right
        
        If body is detected, return coodinate of left palm and right palm
        else return None, None
    """
    pose_landmarks = detect_pose.pose_landmarks
    if pose_landmarks:
        
        _get_single_pose_detail(pose_landmarks, LEFT_HEAD,  'left_head',  frame_n)
        _get_single_pose_detail(pose_landmarks, RIGHT_HEAD, 'right_head', frame_n)
        _get_single_pose_detail(pose_landmarks, LEFT_ARM,  'left_arm',  frame_n)
        _get_single_pose_detail(pose_landmarks, RIGHT_ARM, 'right_arm', frame_n)
            
        left_palm  = (pose_landmarks[0][16], pose_landmarks[0][18], pose_landmarks[0][20])
        right_palm = (pose_landmarks[0][15], pose_landmarks[0][17], pose_landmarks[0][19])
        
        return left_palm, right_palm
    else:
        pose_landmarks = [[NullPoint() for _ in range(25)]]
        _get_single_pose_detail(pose_landmarks, LEFT_HEAD,  'left_head',  frame_n)
        _get_single_pose_detail(pose_landmarks, RIGHT_HEAD, 'right_head', frame_n)
        _get_single_pose_detail(pose_landmarks, LEFT_ARM,  'left_arm',  frame_n)
        _get_single_pose_detail(pose_landmarks, RIGHT_ARM, 'right_arm', frame_n)
        
        return None, None
    

def draw_landmarks_on_image(rgb_image, detection_result, mode = "pose"):
    """
        Draw the point into image to validate the result
    """
    
    if mode == "pose":
        
        pose_landmarks_list = detection_result.pose_landmarks
    else:
        pose_landmarks_list = detection_result.hand_landmarks
        
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        if mode == "pose":
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        else:
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
            
    return annotated_image


def tracking_an_image(pose_detector, hand_detector,image, image_file = None):
    """
        Track an image
    """
    if image_file:
        image = mp.Image.create_from_file(image_file)

    # STEP 4: Detect pose landmarks from the input image.
    detection_pose = pose_detector.detect(image)
    detection_hand = hand_detector.detect(image)

    left_palm, right_palm = _get_pose_detail(detection_pose)
    _get_hand_detail(detection_hand, left_palm = left_palm, right_palm = right_palm)
    
    return image, detection_hand, detection_pose

def tracking_a_video(pose_detector, hand_detector, video, video_file = None):
    """
        Track a video
    """
    
    if video_file:
        video = cv2.VideoCapture(video_file)
    
    video_data = dict()
    video_data['fps'] = video.get(cv2.CAP_PROP_FPS)
    video_data['height'] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_data['width']  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    frame_n = 0
    
    detection_poses = []
    detection_hands = []
    images = []
    
    while video.isOpened():
        success, image = video.read()
        if not success:
            break
        
        image.flags.writeable = False
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data = image)
        
        detection_pose = pose_detector.detect(image)
        detection_hand = hand_detector.detect(image)

        left_palm, right_palm = _get_pose_detail(detection_pose, frame_n=frame_n)
        _get_hand_detail(detection_hand, left_palm = left_palm, right_palm = right_palm, frame_n=frame_n)
        
        detection_hands.append(detection_hand)
        detection_poses.append(detection_pose)
        images.append(image)
        
        frame_n += 1
    
    video.release()
        
    return video_data, images, detection_hands, detection_poses


def detect_image(image = None, pose_detector = None, hand_detector = None, image_file = None, draw = False):
    """
    This function will detect hand and pose from an image
    Parameter:
    - `image`: image mp object
    - `image_file`: directory of image
    - `pose_detector`: pose_detection model. If None then load a new model
    - `hand_detector`: hand_detection model. If None then load a new model
    - `draw`: create image output with detection on it
    
    return: pd.Dataframe: the data for hand detection
    """
    
    global frame
    global type_
    global index
    global x
    global y
    
    assert image != None or image_file != None

    if not pose_detector:
        base_pose_options = python.BaseOptions('models/pose_landmarker_full.task')
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_pose_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=CONFIDENT, 
            min_tracking_confidence=CONFIDENT,
            output_segmentation_masks=False)
        pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    if not hand_detector:
        base_hand_options = python.BaseOptions('models/hand_landmarker.task')
        hand_options = vision.HandLandmarkerOptions(
            base_options=base_hand_options,
            num_hands=2,
            min_hand_detection_confidence=CONFIDENT, 
            min_tracking_confidence=CONFIDENT,
            running_mode=vision.RunningMode.IMAGE)
        hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    # STEP 3: Load the input image.
    image, detection_hand, detection_pose = tracking_an_image(pose_detector, hand_detector, image = image, image_file = image_file)
    type_ = [x.lower() for x in type_]

    df = pd.DataFrame({
                "frame": frame, 
                "type": type_, 
                "index": index, 
                "x": x, 
                "y": y
                #"z": z
            })


    # print(detection_pose.pose_landmarks)
    # print(detection_pose.pose_world_landmarks)

    print("DETECTED")
    
    if draw:
        if image_file:
            destination = "track/"+image_file
        else:
            destination = "track/random.jpg"
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_pose)
        annotated_image = draw_landmarks_on_image(annotated_image, detection_hand, mode = "hand")
        cv2.imwrite(destination, annotated_image)
    return df

def detect_video(video = None, pose_detector = None, hand_detector = None, video_file = None, vis = False):
    """
    This function will detect hand and pose from a video
    Parameter:
    - `video`: video cv2 object
    - `video_file`: directory of video
    - `pose_detector`: pose_detection model. If None then load a new model
    - `hand_detector`: hand_detection model. If None then load a new model
    - `draw`: create a video output with detection on it
    
    return: pd.Dataframe: the data for hand detection
    """
    
    global frame
    global type_
    global index
    global x
    global y
    
    assert video != None or video_file != None

    if not pose_detector:
        base_pose_options = python.BaseOptions('models/pose_landmarker_full.task')
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_pose_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=CONFIDENT, 
            min_tracking_confidence=CONFIDENT,
            output_segmentation_masks=False)
        pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    if not hand_detector:
        base_hand_options = python.BaseOptions('models/hand_landmarker.task')
        hand_options = vision.HandLandmarkerOptions(
            base_options=base_hand_options,
            num_hands=2,
            min_hand_detection_confidence=CONFIDENT, 
            min_tracking_confidence=CONFIDENT,
            running_mode=vision.RunningMode.IMAGE)
        hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    # STEP 3: Load the input image.
    video_data, images, detection_hands, detection_poses = tracking_a_video(pose_detector, hand_detector, video = video, video_file = video_file)
    type_ = [x.lower() for x in type_]
    df = pd.DataFrame({
                "frame": frame, 
                "type": type_, 
                "index": index, 
                "x": x, 
                "y": y
                #"z": z
            })


    # print(detection_pose.pose_landmarks)
    # print(detection_pose.pose_world_landmarks)

    print("DETECTED")
    if vis:
        if video_file:
            destination = "track/"+video_file
        else:
            destination = "track/random.mp4"
        
        # This gonna need more if-else for file format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 codec
        out = cv2.VideoWriter(destination, fourcc, video_data['fps'], (video_data['width'], video_data['height']))
            
        for image, detection_hand, detection_pose in zip(images, detection_hands, detection_poses):
            annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_pose)
            annotated_image = draw_landmarks_on_image(annotated_image, detection_hand, mode = "hand")
            out.write(annotated_image)
            
        out.release()
         
        
    
    return df
    

if __name__ == "__main__":
    df = detect_video(video_file="videos/cc.mp4", vis=True)
    df.to_csv('record/cc1.csv')
# 
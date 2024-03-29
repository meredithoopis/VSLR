import cv2
import mediapipe as mp 

def main(): 
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands


    image = cv2.imread('cc.jpg')


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_rgb.flags.writeable = False

    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        results = hands.process(image_rgb)


        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        hand_landmarks_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
    
                for ids, landmrk in enumerate(hand_landmarks.landmark):
 
                    cx, cy = int(landmrk.x * image_bgr.shape[1]), int(landmrk.y * image_bgr.shape[0])
                    print(cx, cy)
                    hand_landmarks_list.append((cx, cy))

                mp_drawing.draw_landmarks(
                    image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        #cv2.imshow('MediaPipe Hands', image_bgr)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


        print(hand_landmarks_list)

if __name__ == "__main__": 
    main()


'''
#Holistic 
height = 400 
width = 400 
def resize(img): 
    h,w = img.shape[:2]
    if h < w: 
        img = cv2.resize(img, (width, math.floor(h/(w/width))))
    else: 
        img = cv2.resize(img, (math.floor(w/(h/height)), height))
    cv2.imshow(img)



mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_style = mp.solutions.drawing_styles 

with mp_holistic.Holistic(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2
) as holistic: 
    image = cv2.imread('hung.jpg')
    img_h, img_w, _  = image.shape 
    res = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if res.pose_landmarks:
        print(
          f'Nose coordinates: ('
          f'{res.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * img_w}, '
          f'{res.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * img_h})'
      )
    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    # Draw pose, left and right hands, and face landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        res.face_landmarks,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_style
        .get_default_face_mesh_tesselation_style())
    mp_drawing.draw_landmarks(
        annotated_image,
        res.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_style.
        get_default_pose_landmarks_style())
    cv2.imwrite('annotated.jpg', annotated_image)
    
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        res.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
    cv2.imshow('img', annotated_image)


'''
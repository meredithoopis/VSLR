import cv2

def show(video_path): 
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise Exception("Unable to open video file.")
    else:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                raise Exception("Unable to read frame from video.")
                break
            cv2.imshow('Video', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()
    
def get_fps(video_path): 
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened(): 
        raise Exception("Not able to open the video")
        
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_cap.release()
    return fps 

def get_duration(video_path): 
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened(): 
        raise Exception("Not able to open the video")
        
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    totalNoFrames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    durationInSeconds = totalNoFrames // fps
    video_cap.release()
    return durationInSeconds
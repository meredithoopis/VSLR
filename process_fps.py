import cv2 
import ffmpegcv


def cal_fps(video_path): 
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened(): 
        raise Exception("Not able to open the video")
        return 
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_cap.release()
    return fps 

def get_video_encoding_format(video_path):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Unable to open video file.")
        return
    codec = int(video_capture.get(cv2.CAP_PROP_FOURCC))

    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

    video_capture.release()

    return codec_str

#fps = cal_fps('videos/cc1_reduced.mp4')
#print(round(fps))

def reduce_fps(input_path, output_path, target_fps): 
    video_cap = cv2.VideoCapture(input_path)
    if not video_cap.isOpened(): 
        raise Exception("Not able to open the video")
        return 
    frame_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = video_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc(*'h264')
    #(frame_h, )
    #output_video = ffmpegcv.VideoWriter(output_path, 'h264', original_fps / target_fps)
    output_video = ffmpegcv.VideoWriter(output_path, 'h264', target_fps, (frame_w, frame_h))
    frame_skip = int(original_fps / target_fps)
    frame_count = 0 
    while video_cap.isOpened(): 
        ret, frame = video_cap.read()
        if not ret: 
            break 
        frame_count += 1 
        if frame_count % frame_skip == 0: 
            output_video.write(frame)
        if frame_count >= total_frames: 
            break 
    video_cap.release()
    output_video.release()

#reduce_fps('videos/cc1.mp4', 'videos/cc1_reduced.mp4', 12)



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

#show('cc.mp4')
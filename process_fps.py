import cv2 
import ffmpegcv
from SL.ultis.video_utils import *

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

def reduce_fps(input_path, target_fps): 
    output_path = ''.join(input_path.split('.')[:-1]) + "_reduced_" \
                    + str(target_fps) + "." + input_path.split('.')[-1]
    print(output_path)
    video_cap = cv2.VideoCapture(input_path)
    if not video_cap.isOpened(): 
        raise Exception("Not able to open the video")
        
    frame_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    original_fps = video_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video = ffmpegcv.VideoWriter(output_path, 'h264', target_fps, (frame_w, frame_h))
    frame_skip = int(original_fps / target_fps)
    
    fps_diff = original_fps/target_fps
    
    frame_count = 0 
    frame_recorded = 0

    if original_fps >= target_fps:
        while video_cap.isOpened(): 
            ret, frame = video_cap.read()
            if not ret: 
                break 
            
            if int(frame_count/fps_diff) == frame_recorded: 
                output_video.write(frame)
                frame_recorded +=1
            if frame_count >= total_frames: 
                break 
            frame_count += 1 
    else:
        while video_cap.isOpened(): 
            ret, frame = video_cap.read()
            if not ret: 
                break 
            
            while frame_count*fps_diff < frame_recorded: 
                output_video.write(frame)
                frame_count +=1
            if frame_recorded >= total_frames: 
                break 
            frame_recorded += 1 
        
    video_cap.release()
    output_video.release()





if __name__ == "__main__":
    # show('videos/cc.mp4')
    # reduce_fps('videos/cc1_reduced_10.mp4', 60)
    # print(get_fps('videos/cc1.mp4'))
    print("Process_FPS")



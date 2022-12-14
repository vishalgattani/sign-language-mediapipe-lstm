import sys
import mediapipe as mp
import cv2
import numpy as np
import time
import os
from mp_utils import mediapipe_detection,draw_landmarks, mp_drawing, mp_holistic, extract_keypoints

def dataCollection(actions,num_sequences,num_frames):
    print("Need to collect data...")
    cap = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed")

    # width  = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float `width`
    # height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    # specify folder where you will store action data
    DATA_PATH = os.path.join('mp_data')

    make_directories = input("Make directories for training? [y/n]: ")
    if make_directories == 'y':
        for action in actions:
            for num_sequence in range(num_sequences):
                try:
                    os.makedirs(os.path.join(DATA_PATH,action,str(num_sequence)))
                except:
                    pass
    else:
        sys.exit()

    start_recording = input("Start Recording? [y/n]: ")
    if(start_recording=="y"):
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # loop for actions
            for action in actions:
                # loop for videos
                for sequence in range(num_sequences):
                    # loop for frames in videos
                    for frame_num in range(num_frames):
                        # read feed
                        ret, frame = cap.read()
                        # making detections
                        image, results = mediapipe_detection(frame, holistic)
                        # draw markers
                        draw_landmarks(image, results)

                        if frame_num == 0:
                            cv2.putText(image, 'STARTING COLLECTION', (120,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.waitKey(3000)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}/{}'.format(action, sequence+1,num_sequences), (15,12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        else:
                            cv2.putText(image, 'Collecting frames for {} Video Number {}/{}'.format(action, sequence+1,num_sequences), (15,12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # extract keypoints and save it to their respective folders
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                        # print(npy_path)
                        np.save(npy_path,keypoints)

                        # output
                        cv2.imshow('OpenCV Feed', image)
                        # break
                        if cv2.waitKey(1) == 27:
                            print("Closing")        # wait for ESC key to exit
                            cv2.destroyAllWindows()
                            break

        cap.release()
        cv2.destroyAllWindows()
    else:
        # path = '/home/User/Desktop/file.txt'
        # Check whether the specified path exists or not
        print(os.path.exists(DATA_PATH))
        print("Did not start recording...")


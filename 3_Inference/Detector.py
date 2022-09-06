import os
import sys

def get_parent_dir(n=1):
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")

sys.path.append(src_path)
sys.path.append(utils_path)

import argparse
from keras_yolo3.yolo import YOLO, detect_video
from timeit import default_timer as timer
from utils import detect_object
from Get_File_Paths import GetFileList
from Train_Utils import get_anchors

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(n=1), "Data")

image_folder = os.path.join(data_folder, "Source_Images")

image_test_folder = os.path.join(image_folder, "Test_Images")

detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = os.path.join(model_folder, "trained_weights_final.h5")
model_classes = os.path.join(model_folder, "data_classes.txt")

anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

if __name__ == "__main__":

    save_img = True

    # Split images and videos
    img_endings = (".jpg", ".jpeg", ".png")
    vid_endings = (".mp4", ".mpeg", ".mpg", ".avi")

    input_image_paths = []
    input_video_paths = []
    for item in input_paths:
        if item.endswith(img_endings):
            input_image_paths.append(item)
        elif item.endswith(vid_endings):
            input_video_paths.append(item)

    output_path = detection_results_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    anchors = get_anchors(anchors_path)
    # define YOLO detector
    yolo = YOLO(
        **{
            "model_path": model_weights ,
            "anchors_path": anchors_path,
            "classes_path": model_classes,
            "score": =0.25,
            "gpu_num": 1,
            "model_image_size": (416, 416),
        }
    )

    # labels to draw on images
    class_file = open(model_classes, "r")
    input_labels = [line.rstrip("\n") for line in class_file.readlines()]

    if input_image_paths:
        # This is for images
        for i, img_path in enumerate(input_image_paths):
            print(img_path)
            prediction, image = detect_object(yolo,img_path,save_img=save_img,save_img_path=detection_results_folder,postfix='.jpg')

    # This is for videos
    if input_video_paths:
        for i, vid_path in enumerate(input_video_paths):
            output_path = os.path.join(
                FLAGS.output,
                os.path.basename(vid_path).replace(".", FLAGS.postfix + "."),
            )
            detect_video(yolo, vid_path, output_path=output_path)      
    # Close the current yolo session
    yolo.close_session()

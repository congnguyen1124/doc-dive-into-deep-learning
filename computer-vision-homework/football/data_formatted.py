import json
import os
from pprint import pprint

import cv2

video_folder = "train"
output_folder = "train_formatted"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    os.makedirs(os.path.join(output_folder,"images"))
    os.makedirs(os.path.join(output_folder,"labels"))

def convert_video_to_image():
    video_path = os.path.join(video_folder, folder_name, folder_name + ".mp4")
    cap = cv2.VideoCapture(video_path)
    counter = 1
    while cap.isOpened():
        # flag nếu quá trình đọc lỗi trả về false
        flag, frame = cap.read()
        if not flag:
            break
        cv2.imwrite(os.path.join(output_folder, "images", "{}_{}.jpg".format(folder_name, counter)), frame)
        counter += 1

def convert_json_to_annotation():
    anno_path = os.path.join(video_folder, folder_name, folder_name + ".json")
    with open(anno_path, 'r') as file:
        data = json.load(file)
        pprint(data["annotations"])

for folder_name in os.listdir(video_folder):
    convert_video_to_image()
    convert_json_to_annotation()

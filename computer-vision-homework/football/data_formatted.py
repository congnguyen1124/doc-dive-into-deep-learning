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

# def convert_video_to_image():
#     video_path = os.path.join(video_folder, folder_name, folder_name + ".mp4")
#     cap = cv2.VideoCapture(video_path)
#     counter = 1
#     while cap.isOpened():
#         # flag nếu quá trình đọc lỗi trả về false
#         flag, frame = cap.read()
#         if not flag:
#             break
#         cv2.imwrite(os.path.join(output_folder, "images", "{}_{}.jpg".format(folder_name, counter)), frame)
#         counter += 1
#
# def convert_json_to_annotation():
#     anno_path = os.path.join(video_folder, folder_name, folder_name + ".json")
#     with open(anno_path, 'r') as file:
#         data = json.load(file)
#         pprint(data["annotations"])

for folder_name in os.listdir(video_folder):
    anno_path = os.path.join(video_folder, folder_name, folder_name + ".json")
    with open(anno_path, 'r') as file:
        data = json.load(file)

    video_path = os.path.join(video_folder, folder_name, folder_name + ".mp4")
    cap = cv2.VideoCapture(video_path)
    counter = 1
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        h, w, _ = frame.shape
        cv2.imwrite(os.path.join(output_folder, "images", "{}_{}.jpg".format(folder_name, counter)), frame)
        player_ball_data = [obj for obj in data["annotations"] if
                            obj["image_id"] == counter and obj["category_id"] in [3, 4]]

        cv2.imwrite(os.path.join(output_folder, "images", "{}_{}.jpg".format(folder_name, counter)), frame)
        player_data = [obj for obj in data["annotations"] if obj["image_id"] == counter and obj["category_id"] == 4]
        with open(os.path.join(output_folder, "labels", "{}_{}.txt".format(folder_name, counter)), "w") as f:
            for obj in player_data:
                bbox = obj["bbox"]
                xmin, ymin, width, height = bbox
                xcent = xmin + width / 2
                ycent = ymin + height / 2
                xcent /= w
                width /= w
                ycent /= h
                height /= h
                if obj["category_id"] == 4:
                    class_id = 0  # player
                else:
                    class_id = 1  # ball
                f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_id, xcent, ycent, width, height))
        exit(0)
        counter += 1

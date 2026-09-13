import json
import os
import cv2

video_folder = "datasets/val"
output_folder = "val"

def create_output_folders():
    """Tạo các folder output nếu chưa tồn tại."""
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)


def extract_images(video_path, folder_name):
    """
    Đọc video và lưu từng frame vào folder images.

    Returns:
        int: Số lượng frame đã lưu.
    """
    cap = cv2.VideoCapture(video_path)

    counter = 1

    while cap.isOpened():
        flag, frame = cap.read()

        if not flag:
            break

        image_name = f"{folder_name}_{counter}.jpg"
        image_path = os.path.join(
            output_folder,
            "images",
            image_name
        )

        cv2.imwrite(image_path, frame)

        counter += 1

    cap.release()

    return counter - 1


def extract_labels(annotation_data, folder_name, frame_width, frame_height):
    """
    Đọc annotation data và tạo file label cho từng frame.

    Category:
        4 -> player -> class_id = 0
        3 -> ball   -> class_id = 1
    """

    annotations = annotation_data["annotations"]

    # Lấy tất cả frame có annotation thuộc player hoặc ball
    frame_ids = sorted({
        obj["image_id"]
        for obj in annotations
        if obj["category_id"] in [3, 4]
    })

    for frame_id in frame_ids:

        player_ball_data = [
            obj
            for obj in annotations
            if obj["image_id"] == frame_id
            and obj["category_id"] in [3, 4]
        ]

        label_name = f"{folder_name}_{frame_id}.txt"

        label_path = os.path.join(
            output_folder,
            "labels",
            label_name
        )

        with open(label_path, "w") as f:

            for obj in player_ball_data:

                bbox = obj["bbox"]

                xmin, ymin, width, height = bbox

                # Convert xywh -> center xywh
                xcent = xmin + width / 2
                ycent = ymin + height / 2

                # Normalize về [0, 1]
                xcent /= frame_width
                ycent /= frame_height
                width /= frame_width
                height /= frame_height

                # Category ID -> YOLO class ID
                if obj["category_id"] == 4:
                    class_id = 0  # player
                elif obj["category_id"] == 3:
                    class_id = 1  # ball
                else:
                    continue

                f.write(
                    f"{class_id} "
                    f"{xcent:.6f} "
                    f"{ycent:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )


def get_frame_size(video_path):
    """
    Lấy width và height của video.
    """
    cap = cv2.VideoCapture(video_path)

    flag, frame = cap.read()

    if not flag:
        cap.release()
        return None, None

    height, width, _ = frame.shape

    cap.release()

    return width, height


def main():

    create_output_folders()

    for folder_name in os.listdir(video_folder):

        folder_path = os.path.join(video_folder, folder_name)

        # Bỏ qua những file không phải folder
        if not os.path.isdir(folder_path):
            continue

        # ==========================
        # 1. IMAGE
        # ==========================

        video_path = os.path.join(
            video_folder,
            folder_name,
            f"{folder_name}.mp4"
        )

        extract_images(
            video_path,
            folder_name
        )

        # ==========================
        # 2. LABEL
        # ==========================

        anno_path = os.path.join(
            video_folder,
            folder_name,
            f"{folder_name}.json"
        )

        with open(anno_path, "r") as file:
            data = json.load(file)

        frame_width, frame_height = get_frame_size(video_path)

        if frame_width is None:
            print(f"Không thể đọc video: {video_path}")
            continue

        extract_labels(
            data,
            folder_name,
            frame_width,
            frame_height
        )


if __name__ == "__main__":
    main()

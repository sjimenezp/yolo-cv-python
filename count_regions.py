import cv2
import argparse
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import torch
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon
from ultralytics.utils.files import increment_path
from shapely.geometry.point import Point
from pathlib import Path

WINDOW_WIDTH = 1024
WINDOW_NAME = "Tracking objects"
VIDEO_SOURCE = "data/people_walking.mp4"
OUTPUT_PATH = "output"
YOLO_MODEL = "yolov5mu_v3.pt"

track_history = defaultdict(list)

current_region = None
counting_regions = [
    # {
    #     "name": "YOLOv8 Polygon Region",
    #     "polygon": Polygon(
    #         [(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]
    #     ),  # Polygon points
    #     "counts": 0,
    #     "dragging": False,
    #     "region_color": (255, 42, 4),  # BGR Value
    #     "text_color": (255, 255, 255),  # Region Text Color
    # },
    {
        "name": "Counting Region",
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),
        "counts": 0,
        "dragging": False,
        "region_color": (255, 0, 0),
        "text_color": (255, 255, 255),
    },
]


def mouse_callback(event, x, y, flags, param):
    global current_region

    # Mouse left button down event
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [
                    (p[0] + dx, p[1] + dy)
                    for p in current_region["polygon"].exterior.coords
                ]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def setup_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(YOLO_MODEL)
    return model, device


def setup_video_source(webcam):
    video_source = 0 if webcam else VIDEO_SOURCE
    capture = cv2.VideoCapture(video_source)
    assert capture.isOpened(), "Error reading video file"
    return capture, video_source


def adjust_window_size(width, height):
    aspect_ratio = width / height
    new_height = int(WINDOW_WIDTH / aspect_ratio)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, new_height)


def setup_output(capture, source, webcam, exist_ok, resize, save_img):
    width, height, fps = int(capture.get(3)), int(capture.get(4)), int(capture.get(5))
    if resize:
        adjust_window_size(width, height)

    if save_img:
        save_dir = Path(OUTPUT_PATH)
        save_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if not webcam:
            file_path = increment_path(
                save_dir / f"{Path(source).stem}.mp4", exist_ok=exist_ok
            )
        else:
            file_path = increment_path(save_dir / "webcam.mp4", exist_ok=exist_ok)

        return cv2.VideoWriter(
            str(file_path),
            fourcc,
            fps,
            (width, height),
        )
    return None


def annotate_box(frame, annotator, box, names, cls, track_id, track_history, thickness):
    annotator.box_label(box, str(names[cls]), color=colors(cls, True))
    bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / thickness)

    track = track_history[track_id]
    track.append((float(bbox_center[0]), float(bbox_center[1])))
    if len(track) > 30:
        track.pop(0)
    points = np.hstack(track).reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(frame, [points], False, colors(cls, True), thickness)
    return bbox_center


def annotate_frame(frame, results, track_history, thickness, counting_regions):
    names = results[0].names
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy
        track_ids = results[0].boxes.id.tolist()
        clss = results[0].boxes.cls.tolist()

        annotator = Annotator(frame, line_width=2, example=str(names))

        for box, track_id, cls in zip(boxes, track_ids, clss):
            bbox_center = annotate_box(
                frame,
                annotator,
                box,
                names,
                cls,
                track_id,
                track_history,
                thickness,
            )

            for region in counting_regions:
                if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                    region["counts"] += 1

    for region in counting_regions:
        region_label = str(region["counts"])
        region_color = region["region_color"]
        region_text_color = region["text_color"]

        polygon_coords = np.array(region["polygon"].exterior.coords, np.int32)
        centroid_x, centroid_y = (
            int(region["polygon"].centroid.x),
            int(region["polygon"].centroid.y),
        )

        text_size, _ = cv2.getTextSize(
            region_label, cv2.FONT_HERSHEY_SIMPLEX, 1, thickness
        )
        text_x = centroid_x - text_size[0] // 2
        text_y = centroid_y + text_size[1] // 2
        cv2.rectangle(
            frame,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            region_color,
            -1,
        )
        cv2.putText(
            frame,
            region_label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            region_text_color,
            thickness,
        )
        cv2.polylines(frame, [polygon_coords], True, region_color, thickness)


def process_frame(frame, model, device, classes, track_history, thickness):
    results = model.track(
        frame, persist=True, show=False, device=device, classes=classes
    )
    annotate_frame(frame, results, track_history, thickness, counting_regions)


def cleanup(save_img, video_writer, capture, count):
    del count
    if save_img:
        video_writer.release()
    capture.release()
    cv2.destroyAllWindows()


def main(
    speed=None,
    webcam=False,
    resize=False,
    classes=[0],
    view_img=False,
    save_img=False,
    exist_ok=False,
    thickness=2,
):
    count = 0
    model, device = setup_model()
    window_name = WINDOW_NAME
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    capture, source = setup_video_source(webcam)
    video_writer = setup_output(capture, source, webcam, exist_ok, resize, save_img)

    while capture.isOpened():
        success, frame = capture.read()

        if not success:
            break

        count += 1
        if speed:
            if count % speed != 0:
                continue

        process_frame(frame, model, device, classes, track_history, thickness)

        if view_img:
            if count == 1:
                cv2.setMouseCallback(window_name, mouse_callback)
            cv2.imshow(window_name, frame)

        if save_img:
            video_writer.write(frame)

        for region in counting_regions:  # Reinitialize count for each region
            region["counts"] = 0

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cleanup(save_img, video_writer, capture, count)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--speed",
        type=int,
        nargs="?",
        default=1,
        help="Playback speed. 1 is normal speed, 2 is double speed, etc.",
    )
    parser.add_argument(
        "-w", "--webcam", action="store_true", help="Use webcam as video source."
    )
    parser.add_argument(
        "-r", "--resize", action="store_true", help="Resize the video window."
    )
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--thickness", type=int, default=2, help="bounding box thickness"
    )
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_options()
    main(**vars(options))

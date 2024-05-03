import cv2
import argparse
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import torch

WINDOW_WIDTH = 1024
WINDOW_NAME = "Tracking objects"
VIDEO_SOURCE = "data/people_walking.mp4"


def print_mouse_coordinates(event, x, y, _, __):
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_coordinates = x, y
        print(mouse_coordinates)


def main(mouse=False, speed=None, webcam=False, resize=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n-seg.pt")
    count = 0

    window_name = WINDOW_NAME
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if mouse:
        cv2.setMouseCallback(window_name, print_mouse_coordinates)

    video_source = 0 if webcam else VIDEO_SOURCE
    capture = cv2.VideoCapture(video_source)

    assert capture.isOpened(), "Error reading video file"
    original_width, original_height, fps = (
        int(capture.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )
    if resize:
        aspect_ratio = original_width / original_height
        new_height = int(WINDOW_WIDTH / aspect_ratio)
        cv2.resizeWindow(window_name, WINDOW_WIDTH, new_height)

    while capture.isOpened():
        success, frame = capture.read()

        if not success:
            break

        count += 1
        if speed:
            if count % speed != 0:
                continue

        annotator = Annotator(frame, line_width=2)

        results = model.track(frame, persist=True, show=False, device=device)
        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.tolist()

            for mask, track_id in zip(masks, track_ids):
                annotator.seg_bbox(
                    mask=mask,
                    mask_color=colors(track_id, True),
                    track_label=str(int(track_id)),
                )

        cv2.imshow(window_name, frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mouse",
        action="store_true",
        help="Print mouse coordinates when moving the mouse over the video window.",
    )
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
    args = parser.parse_args()

    main(args.mouse, args.speed, args.webcam, args.resize)

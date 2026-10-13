"""Create a small synthetic video for checking the replay pipeline."""
from pathlib import Path

import cv2
import numpy as np


def main():
    video = Path(__file__).resolve().parent / "data" / "demo" / "sample.avi"
    video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
    if not writer.isOpened():
        raise RuntimeError("Cannot create the demo video")
    try:
        for frame_index in range(60):
            frame = np.full((360, 640, 3), 235, dtype=np.uint8)
            cv2.putText(frame, "SYNTHETIC DATA - pipeline check only", (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            x = 40 + frame_index * 7
            cv2.rectangle(frame, (x, 145), (x + 50, 195), (180, 100, 30), -1)
            cv2.putText(frame, f"frame={frame_index}  time={frame_index / 10:.1f}s", (25, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            writer.write(frame)
    finally:
        writer.release()
    print("Created:", video)
    print("60 frames / 10 fps / 6 seconds. No robot or failure annotations.")


if __name__ == "__main__":
    main()

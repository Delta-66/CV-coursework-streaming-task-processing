"""Check video IO, metric calculation, bounded history and GPU execution."""
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import h5py
import matplotlib
import numpy as np
import pandas as pd
import PIL
import sklearn
import torch
import torchvision
import tqdm
import yaml
from sklearn.metrics import f1_score


def main():
    with TemporaryDirectory() as folder:
        video = Path(folder) / "sample.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
        if not writer.isOpened():
            raise RuntimeError("Video writer unavailable")
        for index in range(5):
            writer.write(np.full((64, 64, 3), index * 40, dtype=np.uint8))
        writer.release()
        reader = cv2.VideoCapture(str(video))
        history = deque(maxlen=2)
        count = 0
        while True:
            success, frame = reader.read()
            if not success:
                break
            history.append(frame)
            count += 1
        reader.release()
        assert count == 5 and len(history) == 2
        with h5py.File(Path(folder) / "sample.h5", "w") as file:
            file["timestamps"] = np.arange(count) / 5
        assert f1_score([0, 1, 1], [0, 1, 1]) == 1.0
    print("PASS: video replay, bounded history, HDF5, F1 and dependency imports")
    print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    result = torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")
    torch.cuda.synchronize()
    assert result[0, 0].item() == 32
    print("PASS: GPU matrix multiplication on", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()

"""
Simple video reader & sample-video generator.

Usage (examples):
  # create a tiny sample video to experiment with
  python src/video_reader.py --create-sample --sample-path data/sample_video.mp4

  # read a video, print metadata and preview the first few frames (no GUI)
  python src/video_reader.py --video data/sample_video.mp4 --max-frames 50

  # read and display frames with an OpenCV window (Windows/desktop only)
  python src/video_reader.py --video data/sample_video.mp4 --display
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

def create_sample_video(path, w=320, h=240, fps=20, n_frames=60):
    """Create a small synthetic video for testing (do NOT commit large videos)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # try mp4; fallback to avi if needed
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        # fallback
        path = os.path.splitext(path)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (int((i * 3) % 256), int((i * 5) % 256), int((255 - i * 4) % 256))
        cx = int(w * (0.2 + 0.6 * ((i % 20) / 19.0)))
        cy = int(h * (0.2 + 0.6 * (((i + 5) % 20) / 19.0)))
        cv2.circle(frame, (cx, cy), 18, (255, 255, 255), -1)
        cv2.putText(frame, f"Frame {i+1}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        writer.write(frame)
    writer.release()
    return path


def read_video(path, max_frames=None, display=False):
    """
    Open a video file, print metadata, and optionally preview frames.
    - max_frames: stop after this many frames (useful for testing).
    - display: if True, use cv2.imshow windows (requires desktop GUI).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else None

    print(f"Video: {path}")
    print(f" - resolution: {width} x {height}")
    print(f" - fps: {fps}")
    print(f" - frame_count (reported): {frame_count}")
    print(f" - duration (s): {duration:.2f}" if duration else " - duration: unknown")

    read = 0
    start = time.time()
    while True:
        if max_frames is not None and read >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        read += 1

        # If display requested, show a window. Press 'q' to quit.
        if display:
            cv2.imshow("video_reader preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # For learning: print a line every 50 frames to avoid flooding the screen
        if read % 50 == 0:
            print(f" - read {read} frames...")

    elapsed = time.time() - start
    cap.release()
    if display:
        cv2.destroyAllWindows()

    print(f"Finished reading. Frames read: {read}. Time: {elapsed:.2f}s")
    return {
        "path": path,
        "fps": fps,
        "frame_count_reported": frame_count,
        "frames_read": read,
        "duration_s": duration,
    }


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--video", "-v", type=str, default=None, help="Path to video file to read")
    p.add_argument("--create-sample", action="store_true", help="Create a small sample video for testing")
    p.add_argument("--sample-path", type=str, default="data/sample_video.mp4", help="Where to write sample video")
    p.add_argument("--max-frames", type=int, default=100, help="Max frames to read (for quick tests)")
    p.add_argument("--display", action="store_true", help="Show frames in an OpenCV window (desktop only)")
    return p


def main(argv):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.create_sample:
        path = create_sample_video(args.sample_path)
        print("Sample video created at:", path)
        # fall through to read the sample
        args.video = path

    if not args.video:
        print("No video path provided. Use --create-sample to make a tiny demo or pass --video PATH.")
        return

    meta = read_video(args.video, max_frames=args.max_frames, display=args.display)
    # print metadata dictionary in a readable way
    for k, v in meta.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main(sys.argv[1:])
"""
Replay visualizer: draws tracks + IDs on top of video frames.

"""

import os
import cv2
import pandas as pd
import numpy as np

def main(video_path, positions_csv, out_video='replay_out.mp4'):
    df = pd.read_csv(positions_csv)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(out_video) or '.', exist_ok=True)
    out = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        fr = df[df['frame'] == frame_idx]
        for _, r in fr.iterrows():
            x, y, w, h = r['x'], r['y'], r['w'], r['h']
            tid = int(r['track_id'])
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{tid}', (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # show live frame
        cv2.imshow('Replay', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Early stop pressed")
            break

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved replay -> {out_video}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--positions', required=True)
    parser.add_argument('--out_video', default='data/raw_videos/replay_out.mp4')
    args = parser.parse_args()
    main(args.video, args.positions, args.out_video)

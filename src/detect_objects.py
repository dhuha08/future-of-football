"""
Standalone video -> detections -> tracker -> CSV of positions & speeds
With live display and early quit while csv saving every 50 frames.
"""

import os
import cv2
import numpy as np
import pandas as pd
import time
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sort_tracker import SortTracker

# YOLO detection backend
try:
    from ultralytics import YOLO
    DET_BACKEND = 'yolov8'
except:
    DET_BACKEND = 'yolov5'
    import torch

def load_detector(conf_thresh=0.3):
    if DET_BACKEND == 'yolov8':
        model = YOLO('yolov8n.pt')
        return lambda img: model(img)[0]
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.conf = conf_thresh
        return lambda img: model(img)

def to_xywh(box):
    x1,y1,x2,y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    return [cx, cy, w, h]

def main(video_path, out_csv, conf_thresh=0.3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    detector = load_detector(conf_thresh)
    tracker = SortTracker(max_age=30, min_hits=1, iou_threshold=0.3)

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t = frame_idx / fps

        dets = []
        try:
            res = detector(frame)
            if DET_BACKEND == 'yolov8':
                for d in res.boxes.data.tolist():
                    x1,y1,x2,y2,conf = d[:5]
                    if conf >= conf_thresh:
                        dets.append([x1,y1,x2,y2,float(conf)])
            else:
                for *xyxy, conf, cls in res.xyxy[0].cpu().numpy():
                    if conf >= conf_thresh:
                        x1,y1,x2,y2 = xyxy
                        dets.append([x1,y1,x2,y2,float(conf)])
        except Exception as e:
            print(f'Detector error at frame {frame_idx}: {e}')

        tracks = tracker.update(np.array(dets))

        for tr in tracks:
            x1,y1,x2,y2,tid,conf = tr
            cx,cy,w,h = to_xywh([x1,y1,x2,y2])
            rows.append({'frame': frame_idx,
                         'time': t,
                         'track_id': int(tid),
                         'x': float(cx),
                         'y': float(cy),
                         'w': float(w),
                         'h': float(h),
                         'conf': float(conf)})

            # draw boxes on frame for live display
            x1_draw = int(cx - w/2)
            y1_draw = int(cy - h/2)
            x2_draw = int(cx + w/2)
            y2_draw = int(cy + h/2)
            cv2.rectangle(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (0,255,0), 2)
            cv2.putText(frame, f'ID:{int(tid)}', (x1_draw, max(0,y1_draw-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # show live frame
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Early stop pressed")
            break

        # save partial CSV every 50 frames
        if frame_idx % 50 == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

        if frame_idx % 250 == 0:
            print(f'Processed {frame_idx} frames...')

    # compute speeds per track (pixels/sec)
    df = pd.DataFrame(rows)
    df = df.sort_values(['track_id','frame']).reset_index(drop=True)
    df['speed_px_per_s'] = 0.0
    for tid, g in df.groupby('track_id'):
        g = g.sort_values('frame')
        frames = g['frame'].values
        xs = g['x'].values
        ys = g['y'].values
        speeds = np.zeros(len(g))
        for i in range(1,len(g)):
            dt = (frames[i]-frames[i-1]) / fps
            if dt > 0:
                dist = np.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2)
                speeds[i] = dist/dt
        df.loc[g.index,'speed_px_per_s'] = speeds

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df.to_csv(out_csv, index=False)
    print('Saved final positions CSV ->', out_csv)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--out_csv', required=True)
    args = parser.parse_args()
    main(args.video, args.out_csv)


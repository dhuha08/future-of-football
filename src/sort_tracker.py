import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bb_test, bb_gt):
    """
    Compute IoU between two boxes.
    bb = [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, bbox, track_id, conf):
        self.bbox = bbox  # [x1,y1,x2,y2]
        self.id = track_id
        self.conf = conf
        self.hits = 1
        self.age = 0
        self.time_since_update = 0


class SortTracker:
    def __init__(self, max_age=30, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self._next_id = 1

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets: Nx5 array [x1,y1,x2,y2,conf]

        Returns:
          Mx6 array [x1,y1,x2,y2,track_id,conf]
        """
        if dets.size == 0:
            # No detections, just age tracks
            outs = []
            for tr in self.tracks:
                tr.age += 1
                tr.time_since_update += 1
            # Remove expired tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return np.array(outs)

        dets = np.array(dets)
        m = len(self.tracks)
        n = len(dets)

        iou_matrix = np.zeros((m, n), dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            for j, d in enumerate(dets):
                iou_matrix[i, j] = iou(tr.bbox, d[:4])

        matched_idx = []
        if m > 0 and n > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_idx.append((r, c))

        unmatched_tr = set(range(m))
        unmatched_det = set(range(n))
        for r, c in matched_idx:
            unmatched_tr.discard(r)
            unmatched_det.discard(c)

        # update matched tracks
        for r, c in matched_idx:
            det = dets[c]
            tr = self.tracks[r]
            tr.bbox = det[:4].tolist()
            tr.conf = float(det[4])
            tr.hits += 1
            tr.time_since_update = 0
            tr.age = 0

        # new tracks for unmatched detections
        for j in unmatched_det:
            det = dets[j]
            self.tracks.append(
                Track(det[:4].tolist(), self._next_id, float(det[4]))
            )
            self._next_id += 1

        # prepare output
        outs = []
        for tr in self.tracks:
            tr.age += 1
            tr.time_since_update += 1
            if tr.time_since_update <= self.max_age and tr.hits >= self.min_hits:
                x1, y1, x2, y2 = tr.bbox
                outs.append([x1, y1, x2, y2, tr.id, tr.conf])

        # keep alive only active tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return np.array(outs)


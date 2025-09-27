import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# Helper Functions
def iou(bb_test, bb_gt):
    """
    Computes Intersection over Union between two bboxes
    bb_test, bb_gt = [x1, x2, y1, y2]
    Returns IoU value (0..1)
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    intersection = w * h
    union = ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - intersection)
    return intersection / union if union > 0 else 0
    
# Track CLass
class Track:
    count = 0 

    def __init__(self, bbok):
        """
        bbox = [x1, x2, y1, y2]
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.x[:4] = self.convert_bbox_to_z(bbok) # initial measurement

        self.id = Track.count   #assign unique ID
        Track.count += 1

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1

    def bbox_to_z(self, bbox):
        """
        Converts bbox [x1, x2, y1, y2] to [cx, cy, s, r]
        """
        x1, x2, y1, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        s = w * h  # scale is area
        r = w / float(h)  # ratio
        return np.array([cx, cy, s, r]).reshape((4, 1))
    
    def z_to_bbox(self, z):
        """
        Converts [cx, cy, s, r] back to bbox [x1, x2, y1, y2]
        """
        cx, cy, s, r = z.flatten()
        w = np.sqrt(s * r)
        h = s / w
        x1 = int(cx - w / 2)
        x2 = int(cx + w / 2)
        y1 = int(cy - h / 2)
        y2 = int(cy + h / 2)
        return [x1, x2, y1, y2]
    

# Sort Class
class Sort:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        """
        max_age: maximum frames to keep unmatched tracks alive
        min_hits: minimum hits to consider a track confirmed
        iou_threshold: minimum IoU to match detection to track
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        
    def update(self, detections):
        """
        detections: np.array(N, 5) - [x1, x2, y1, y2, confidence]
        returns: np.arrays(M, 5) - [x1, x2, y1, y2, track_id]   
        """
        tracked_objects = []

        #Step 1: Predict new locations of existing tracks
        for track in self.tracks:
            # track.kf.predict() # implement Kalman predict  
            track.time_since_update += 1

        # Step 2: Associate (IoU + Hungarian)
        matched, unmatched_dets, unmatched_trks = [], [], []
        if len(self.tracks) == 0:
            unmatched_dets = list(range(len(detections)))
        else:
            #Build IoU matrix
            cost_matrix = np.zeroes((len(self.tracks), len(detections)), dtype=np.float32)
            for t, track in enumerate(self.tracks):
                track_bbox = track.z_to_bbox(track.kf.x[:4])
                for d, det in enumerate(detections):
                    cost_matrix[t, d] = 1 - iou(track_bbox, det[:4])

            # Hungarian assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] > (1 - self.iou_threshold):
                    unmatched_trks.append(r)
                    unmatched_dets.append(c)
                else:
                    matched.append((r, c))

            unmatched_trks += [t for t in range(len(self.tracks)) if t not in row_ind]
            unmatched_dets += [d for d in range(len(detections)) if d not in col_ind]

        # Step 3: Update matched tracks
        for t, d in matched:
            track = self.tracks[t]
            track.hits += 1
            track.hit_streak += 1
            track.time_since_update = 0

        # Step 4: Create new tracks for unmatched detections
        for d in unmatched_dets:
            new_track = Track(detections[d][:4])
            self.tracks.append(new_track)

        # Step 5: Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Step 6: Prepare output
        for track in self.tracks:
            bbox = track.z_to_bbox(track.kf.x[:4])
            tracked_objects.append([*bbox + track.id])

        return np.array(tracked_objects)


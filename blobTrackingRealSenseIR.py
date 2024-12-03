import pyrealsense2 as rs
import cv2
import numpy as np
from scipy.spatial.distance import cdist

# Blob tracking data structures
class Track:
    def __init__(self, track_id, x, y):
        self.id = track_id
        self.kalman = cv2.KalmanFilter(6, 2)  # 6 state variables: x, y, vx, vy, ax, ay
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], np.float32)
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        self.kalman.statePre = np.array([x, y, 0, 0, 0, 0], np.float32)
        self.kalman.statePost = np.array([x, y, 0, 0, 0, 0], np.float32)
        self.last_position = (x, y)
        self.lost_frames = 0
        self.bounding_box = None

    def get_velocity(self):
        return np.sqrt(self.kalman.statePost[2]**2 + self.kalman.statePost[3]**2)

tracks = []
next_track_id = 0
max_lost_frames = 30

# RealSense pipeline for IR
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)  # Enable IR stream
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        ir_frame = frames.get_infrared_frame()
        if not ir_frame:
            continue

        # Convert IR frame to numpy array
        ir_image = np.asanyarray(ir_frame.get_data())

        # Preprocessing: Thresholding to isolate reflective regions
        _, thresholded = cv2.threshold(ir_image, 200, 255, cv2.THRESH_BINARY)  # Adjust threshold as needed

        # Find contours for blob detection
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        bounding_boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Adjust minimum blob size threshold
                x, y, w, h = cv2.boundingRect(contour)
                detections.append((x + w // 2, y + h // 2))  # Store centroid
                bounding_boxes.append((x, y, w, h))

        # Track update logic
        if len(detections) > 0:
            detection_array = np.array(detections, np.float32)
            if len(tracks) > 0:
                track_positions = np.array([track.last_position for track in tracks], np.float32)
                distance_matrix = cdist(track_positions, detection_array)

                # Hungarian Algorithm for optimal matching
                row_indices, col_indices = [], []
                for i, row in enumerate(distance_matrix):
                    min_index = np.argmin(row)
                    if row[min_index] < 50:  # Max distance threshold
                        row_indices.append(i)
                        col_indices.append(min_index)

                # Update matched tracks
                used_detections = set()
                for track_idx, det_idx in zip(row_indices, col_indices):
                    if det_idx not in used_detections:
                        track = tracks[track_idx]
                        track.kalman.correct(np.array([detections[det_idx][0], detections[det_idx][1]], np.float32))
                        track.last_position = detections[det_idx]
                        track.bounding_box = bounding_boxes[det_idx]
                        track.lost_frames = 0
                        used_detections.add(det_idx)

                # Create new tracks for unmatched detections
                for i, detection in enumerate(detections):
                    if i not in used_detections:
                        track = Track(next_track_id, detection[0], detection[1])
                        track.bounding_box = bounding_boxes[i]
                        tracks.append(track)
                        next_track_id += 1

            else:
                # No tracks, initialize new ones
                for i, detection in enumerate(detections):
                    track = Track(next_track_id, detection[0], detection[1])
                    track.bounding_box = bounding_boxes[i]
                    tracks.append(track)
                    next_track_id += 1

        # Predict next positions for all tracks
        for track in tracks:
            prediction = track.kalman.predict()
            track.last_position = (int(prediction[0]), int(prediction[1]))
            track.lost_frames += 1

        # Remove stale tracks
        tracks = [track for track in tracks if track.lost_frames <= max_lost_frames]

        # Draw tracked blobs
        display_frame = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
        for track in tracks:
            if track.bounding_box:
                x, y, w, h = track.bounding_box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box
            x, y = track.last_position
            velocity = track.get_velocity()
            cv2.putText(display_frame, f'ID {track.id}', (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(display_frame, f'V: {velocity:.2f}', (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Display the results
        cv2.imshow("Tracked Blobs", display_frame)
        cv2.imshow("Thresholded IR", thresholded)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

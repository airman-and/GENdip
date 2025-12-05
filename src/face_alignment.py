import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

def get_landmarks(image_np):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image_np)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

def align_face(image_np, output_size=1024):
    """
    Aligns face to FFHQ standard using MediaPipe landmarks.
    """
    landmarks = get_landmarks(image_np)
    if landmarks is None:
        print("No face detected for alignment.")
        return None

    # Eye indices (approximate center of eyes)
    # MediaPipe indices: 
    # Left Eye: 33 (inner), 133 (outer)
    # Right Eye: 362 (inner), 263 (outer)
    LEFT_EYE = [33, 133]
    RIGHT_EYE = [362, 263]
    
    h, w, _ = image_np.shape
    
    # Get average point for left and right eye
    left_eye_pts = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in LEFT_EYE])
    right_eye_pts = np.array([(landmarks.landmark[i].x * w, landmarks.landmark[i].y * h) for i in RIGHT_EYE])
    
    left_eye_center = left_eye_pts.mean(axis=0)
    right_eye_center = right_eye_pts.mean(axis=0)
    
    # Compute angle
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Compute scale
    # FFHQ: Inter-eye distance is roughly 0.35 * image_size
    # But we want to be careful not to zoom in TOO much if the face is close.
    # Standard FFHQ alignment usually places eyes at specific coordinates.
    # Let's target inter-eye distance of ~35% of output width.
    desired_dist = 0.35 * output_size 
    dist = np.sqrt(dx**2 + dy**2)
    scale = desired_dist / dist
    
    # Compute center of eyes in source image
    eye_center = ((left_eye_center[0] + right_eye_center[0]) / 2, (left_eye_center[1] + right_eye_center[1]) / 2)
    
    # Affine transformation matrix
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)
    
    # Translation to center eyes in output image
    # FFHQ eyes are roughly centered horizontally (0.5) and slightly above vertical center (0.4)
    t_x = output_size * 0.5
    t_y = output_size * 0.40 
    M[0, 2] += (t_x - eye_center[0])
    M[1, 2] += (t_y - eye_center[1])
    
    aligned = cv2.warpAffine(image_np, M, (output_size, output_size), flags=cv2.INTER_CUBIC)
    return Image.fromarray(aligned)

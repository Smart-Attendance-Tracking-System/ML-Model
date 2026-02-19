import cv2
import pickle
import numpy as np
from face_engine import FaceEngine

EMBEDDINGS_PATH = r"D:\Graduation Project 2026\Python\Files\ML Model\team_embeddings.pkl"
THRESHOLD = 0.6

# Load embeddings
with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings_db = pickle.load(f)

face_engine = FaceEngine()

def cosine_similarity(e1, e2):
    # Convert to numpy arrays and flatten to ensure they are 1D vectors
    e1 = np.asarray(e1).flatten()
    e2 = np.asarray(e2).flatten()
    
    if e1.shape != e2.shape:
        # This will now print the actual shapes to help you debug
        print(f"Warning: Vector mismatch! {e1.shape} vs {e2.shape}")
        return 0.0
        
    denominator = (np.linalg.norm(e1) * np.linalg.norm(e2))
    if denominator == 0:
        return 0.0
    return np.dot(e1, e2) / denominator

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # UNPACKING FIX: face_engine returns (vector, box)
    result = face_engine.extract_from_frame(frame)

    label = "No Face Detected"
    color = (0, 0, 255) # Red for no face/unknown

    if result is not None:
        # Assuming the engine returns: (embedding_vector, bounding_box)
        embedding, face_box = result 
        
        best_match = "Unknown"
        max_sim = -1

        for name, db_embedding in embeddings_db.items():
            sim = cosine_similarity(embedding, db_embedding)

            if sim > max_sim:
                max_sim = sim
                best_match = name

        if max_sim >= THRESHOLD:
            label = f"{best_match} ({max_sim:.2f})"
            color = (0, 255, 0) # Green for match
        else:
            label = f"Unknown ({max_sim:.2f})"

        # OPTIONAL: Draw a box around the face using the unpacked face_box
        if face_box is not None:
            x, y, w, h = face_box
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(frame, label, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)

    cv2.imshow("Team Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27: # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
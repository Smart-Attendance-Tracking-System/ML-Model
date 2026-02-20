import os
import pickle
import numpy as np
from face_engine import FaceEngine

DATASET_PATH = r"D:\Graduation Project 2026\Python\Files\ML Model\teamFaces"
OUTPUT_PATH = r"D:\Graduation Project 2026\Python\Files\ML Model\team_embeddings.pkl"

face_engine = FaceEngine()

embeddings_db = {}

for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    person_embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        result = face_engine.extract_from_path(img_path)

        if result is not None and len(result) > 0:
            embedding = result[0]
            person_embeddings.append(embedding)

    if len(person_embeddings) > 0:
        mean_embedding = np.mean(person_embeddings, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        embeddings_db[person_name] = {
            "embedding": mean_embedding,
            "count": len(person_embeddings)
        }
    
        print(f"{person_name} embeddings generated ({len(person_embeddings)} images)")

        print(type(person_embeddings[0]))
        print(np.array(person_embeddings).shape)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(embeddings_db, f)

print("\nEmbeddings saved successfully.")
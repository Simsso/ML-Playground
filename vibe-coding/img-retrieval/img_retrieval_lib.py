"""Library for image retrieval with CLIP."""

import os
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def load_image_paths(folder_path: str) -> List[str]:
    return [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]


def embed_images_with_clip(image_paths: List[str]) -> np.ndarray:
    embeddings = []
    for path in image_paths:
        image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
        embeddings.append(embedding.cpu().numpy())
    return np.vstack(embeddings)


def load_text_prompts(csv_path: str) -> Dict[int, str]:
    df = pd.read_csv(csv_path)
    return dict(zip(df['id'], df['text']))


def embed_texts_with_clip(texts: List[str]) -> np.ndarray:
    text_input = clip.tokenize(texts).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(text_input)
    return embeddings.cpu().numpy()


def batch_retrieval(text_embeddings: np.ndarray, image_embeddings: np.ndarray) -> List[Tuple[int, int]]:
    similarities = cosine_similarity(text_embeddings, image_embeddings)
    closest_images = np.argmax(similarities, axis=1)
    return list(enumerate(closest_images))


def write_results_to_csv(results: List[Tuple[int, int]], texts: Dict[int, str], image_paths: List[str], output_csv: str):
    output_data = []
    for text_id, image_idx in results:
        output_data.append({
            'id': text_id,
            'text': texts[text_id],
            'image_file_name': os.path.basename(image_paths[image_idx])
        })
    pd.DataFrame(output_data).to_csv(output_csv, index=False)

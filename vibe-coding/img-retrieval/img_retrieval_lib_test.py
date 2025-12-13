import numpy as np
import os
import tempfile
import csv
import pandas as pd
from img_retrieval_lib import *


def test_load_image_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "image1.jpg"), "w").close()
        open(os.path.join(tmpdir, "image2.png"), "w").close()
        open(os.path.join(tmpdir, "not_an_image.txt"), "w").close()

        image_paths = load_image_paths(tmpdir)
        assert len(image_paths) == 2


def test_embed_images_with_clip():
    test_images = ["path/to/image1.jpg", "path/to/image2.jpg"]
    embeddings = embed_images_with_clip(test_images)
    assert embeddings.shape[0] == len(test_images)


def test_load_text_prompts():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv") as tmpfile:
        writer = csv.writer(tmpfile)
        writer.writerow(["id", "text"])
        writer.writerow([1, "a dog"])
        writer.writerow([2, "a cat"])
        tmpfile.seek(0)

        text_prompts = load_text_prompts(tmpfile.name)
        assert len(text_prompts) == 2


def test_embed_texts_with_clip():
    test_texts = ["a dog", "a cat"]
    embeddings = embed_texts_with_clip(test_texts)
    assert embeddings.shape[0] == len(test_texts)


def test_batch_retrieval():
    image_embeddings = np.array([[1, 0], [0, 1]])
    text_embeddings = np.array([[0.9, 0], [0, 0.9]])
    results = batch_retrieval(text_embeddings, image_embeddings)
    assert results == [(0, 0), (1, 1)]


def test_write_results_to_csv():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv") as tmpfile:
        results = [(1, 0), (2, 1)]
        texts = {1: "a dog", 2: "a cat"}
        image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]

        write_results_to_csv(results, texts, image_paths, tmpfile.name)

        tmpfile.seek(0)
        df = pd.read_csv(tmpfile.name)
        assert len(df) == 2

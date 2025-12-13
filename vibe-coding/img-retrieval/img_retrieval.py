"""Script for image retrieval with CLIP."""

import argparse
from img_retrieval_lib import *


def main(folder_path: str, text_csv: str, output_csv: str):
    image_paths = load_image_paths(folder_path)
    image_embeddings = embed_images_with_clip(image_paths)

    text_prompts = load_text_prompts(text_csv)
    text_embeddings = embed_texts_with_clip(list(text_prompts.values()))

    retrieval_results = batch_retrieval(text_embeddings, image_embeddings)

    write_results_to_csv(retrieval_results, text_prompts,
                         image_paths, output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image-Text Retrieval using CLIP")
    parser.add_argument('--image_folder', type=str, required=True,
                        help="Path to the folder containing images")
    parser.add_argument('--text_csv', type=str, required=True,
                        help="Path to the CSV file containing text prompts")
    parser.add_argument('--output_csv', type=str,
                        required=True, help="Path to the output CSV file")

    args = parser.parse_args()
    main(args.image_folder, args.text_csv, args.output_csv)

# Image Retrieval with CLIP

Input:

 - A folder containing several images
 - A CSV containing two columns: id and text description


Output: A CSV file with three columns: id, text description, and the file name of the image that's best matching the text description according to the CLIP model.

## Installing Dependencies

Setup instructions from https://github.com/openai/CLIP

Run either of the two commands below:

```bash
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install torch torchvision
```

Install all other dependencies with:

```bash
pip install ftfy regex tqdm scikit-learn pandas
pip install git+https://github.com/openai/CLIP.git
```

## Run the Retrieval

```bash
python img_retrieval.py \
    --image_folder test_data/img \
    --text_csv test_data/descriptions.csv \
    --output_csv test_data/output.csv
```

## Run the Tests

```bash
pip install pytest
```

## Background

I created the first working version of this script in just 20min using ChatGPT (GPT-4).
It just struck me how well these models can code.
Here's the conversation: https://chat.openai.com/share/392ccac5-a7a0-4433-8a94-443e59a8e798

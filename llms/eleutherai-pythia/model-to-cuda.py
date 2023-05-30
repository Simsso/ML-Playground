"""Loading a model and moving it to the GPU.

Debugging issues around GPU utilization.
The 70M param model is using 2GB of GPU memory.
"""

import argparse
import time
import torch
from transformers import GPTNeoXForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=str, choices=[
                    "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"],
                    required=True)


def main():
    args = parser.parse_args()
    print(args)

    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{args.model_size}-deduped",
        revision="step3000",
        cache_dir=f"./pythia-{args.model_size}-deduped/step3000",
    )
    print("Model loaded.")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        print("Model moved to GPU.")
        gpu_utilization = torch.cuda.max_memory_allocated(
            device) / 1024**3  # Convert bytes to gigabytes

        print(f"GPU utilization of device {device}: {gpu_utilization} GB")
    else:
        print("No GPU available.")

    time.sleep(20)


if __name__ == "__main__":
    main()

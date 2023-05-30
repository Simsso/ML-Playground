"""Fine-tune Pythia LLM on a simple dataset.

Example invocation:

MODEL_SIZE=1b

# Write hparams into a file.
HPARAM_PATH=hparams/test-$MODEL_SIZE-fine-tuning-full.yaml
mkdir -p hparams
echo "learning_rate: 5e-4" > $HPARAM_PATH
echo -e "num_steps: 1000" >> $HPARAM_PATH
echo -e "sgd_momentum: 0.1" >> $HPARAM_PATH

# Generate tiny fine-tuning dataset.
DATA_PATH=data/test.txt
mkdir -p data
echo "This is a test.\nHere is example number two." > $DATA_PATH

# Copy hparams into output path.
OUTPUT_PATH=output/test-$MODEL_SIZE-fine-tuning-full
mkdir -p $OUTPUT_PATH
cp $HPARAM_PATH $OUTPUT_PATH/hparams.yaml

# Start fine-tuning the model.
python -m pip install -r requirements.txt
python pythia-finetune.py \
    --model_size $MODEL_SIZE \
    --fine_tuning_mode full \
    --hparam_path $HPARAM_PATH \
    --dataset_path $DATA_PATH \
    --output_path $OUTPUT_PATH
"""

import argparse
import torch.nn
import torch.optim
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=str, choices=[
                    "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"],
                    required=True)
parser.add_argument("--fine_tuning_mode", type=str,
                    choices=["full", "lora", "prefix"], required=True)
parser.add_argument("--hparam_path", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)


def get_model(model_size: str) -> GPTNeoXForCausalLM:
    return GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/pythia-{model_size}-deduped",
        revision="step3000",
        cache_dir=f"./pythia-{model_size}-deduped/step3000",
    )


def get_tokenizer(model_size: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{model_size}-deduped",
        revision="step3000",
        cache_dir=f"./pythia-{model_size}-deduped/step3000",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_dataset(dataset_path: str, tokenizer) -> dict[str, torch.Tensor]:
    """Load the dataset and return a dictionary of tensors.

    The dataset is a text file with one example per line.
    The returned dict contains 'input_ids', 'attention_mask', and 'target_ids'.
    """
    with open(dataset_path) as file:
        lines = list(file.readlines())
    # We train full batch, i.e., the dataset is one batch.
    result = tokenizer(
        lines, return_tensors="pt", padding='longest', return_attention_mask=True)
    batch_size = result["input_ids"].shape[0]
    # No "bos_token" is prepended here.
    target_tokens = torch.concat([
        result['input_ids'][:, 1:],
        torch.tensor(tokenizer.eos_token_id).expand(batch_size, 1)],
        axis=-1
    )
    result['target_ids'] = target_tokens
    return result


def fine_tune(fine_tuning_mode: str, model: GPTNeoXForCausalLM, tokenizer: AutoTokenizer,
              hparams: dict, dataset_path: str, output_path: str):
    learning_rate = float(hparams["learning_rate"])
    num_steps = int(hparams["num_steps"])
    sgd_momentum = float(hparams["sgd_momentum"])

    model.train()
    opt = torch.optim.SGD(model.parameters(),
                          lr=learning_rate, momentum=sgd_momentum)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    data = load_dataset(dataset_path, tokenizer)

    assert fine_tuning_mode == "full", "Only full fine-tuning is supported."

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    target_ids = data["target_ids"]
    print_gpu_utilization('before-gpu')
    if torch.cuda.is_available():
        device = torch.device("cuda")

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target_ids = target_ids.to(device)
        model.to(device)
        print_gpu_utilization('model-on-gpu')

    for step in range(num_steps):
        opt.zero_grad()
        print_gpu_utilization('post-zero-grad')
        model_output = model(input_ids=input_ids,
                             attention_mask=attention_mask)
        print_gpu_utilization('post-forward-pass')
        logits = model_output.logits.permute([0, 2, 1])
        loss = ce_loss_fn(logits, target_ids)
        loss.backward()
        print_gpu_utilization('post-backward')
        opt.step()
        print_gpu_utilization('post-opt-step')
        print(f"Step {step}: loss={loss.item()}")

        del logits
        del loss
        print_gpu_utilization('post-clean-up')

    model.save_pretrained(output_path)


def print_gpu_utilization(prefix: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_allocation = torch.cuda.max_memory_allocated(
            device)
        gpu_gb = gpu_allocation / 1024**3  # Convert bytes to gigabytes

        print(f"[{prefix}] GPU utilization of device {device}: {gpu_gb} GB")
    else:
        print("No GPU available.")


def main():
    args = parser.parse_args()
    print(args)

    with open(args.hparam_path) as file:
        hparams = yaml.safe_load(file)

    model_size = args.model_size
    model = get_model(model_size)
    tokenizer = get_tokenizer(model_size)

    fine_tune(args.fine_tuning_mode, model, tokenizer, hparams,
              args.dataset_path, args.output_path)


if __name__ == "__main__":
    print("Running finetuning script.")
    main()

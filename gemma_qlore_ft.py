from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from sklearn.metrics import log_loss, accuracy_score
from utils import CustomTokenizer
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICE"] = (
    "0"  # setting the visible computing resources 设置环境可见的运算资源
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parameters for the training script")
    parser.add_argument("--model_path", type=str, default="../gemma-2-9b-it-bnb-4bit")
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--n_splits", type=int, default=100)
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--optim_type", type=str, default="adamw_8bit")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--freeze_layers",type=int,default=16)
    args = parser.parse_args()

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=[i for i in range(42) if i >= args.freeze_layers],
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=TaskType.SEQ_CLS,
        )
    
    tokenzier = GemmaTokenizerFast.fro

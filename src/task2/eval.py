import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import scipy.io as io
import torch
from torch.utils.data import Dataset, DataLoader

from .llm_model import ECGQwenForAF, DEFAULT_QWEN_NAME, apply_lora_to_llm
from .train import _ecg_preprocess

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class EvalDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        mat_dir: str,
        ecg_len: int,
        downsample: int,
        preprocessed_npz: Optional[str] = None,
    ):
        self.items: List[Dict[str, Any]] = []
        self.mat_dir = Path(mat_dir)
        self.ecg_len = ecg_len
        self.downsample = downsample
        self._preprocessed_cache: Optional[Dict[str, np.ndarray]] = None

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                instr = obj["instruction"]
                prompt = instr + "\n答："
                ans = obj["answer"].strip()
                label = 1 if "有房颤" in ans else 0
                self.items.append(
                    {
                        "file_name": obj["file_name"],
                        "prompt": prompt,
                        "gt_answer": ans,
                        "label": label,
                    }
                )
        if preprocessed_npz is not None:
            self._load_preprocessed(preprocessed_npz)

    def __len__(self) -> int:
        return len(self.items)

    def _load_mat(self, file_name: str) -> np.ndarray:
        mat_path = self.mat_dir / f"{file_name}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"ECG mat not found: {mat_path}")
        data = io.loadmat(str(mat_path))["val"]
        sig = np.asarray(data).reshape(-1)
        return sig

    def _load_preprocessed(self, path: str) -> None:
        cache = np.load(path, allow_pickle=True)
        ecg = cache["ecg"]
        names = cache["file_name"]
        if ecg.shape[0] != len(self.items):
            raise ValueError(
                f"Preprocessed NPZ len {ecg.shape[0]} != JSON len {len(self.items)}"
            )
        for idx, (item, cached_name) in enumerate(zip(self.items, names)):
            if item["file_name"] != str(cached_name):
                raise ValueError(
                    f"Mismatch at index {idx}: JSON {item['file_name']} != NPZ {cached_name}"
                )
        if ecg.shape[1] != self.ecg_len:
            raise ValueError(
                f"Preprocessed ECG length {ecg.shape[1]} != expected {self.ecg_len}"
            )
        cache.close()
        self._preprocessed_cache = {"ecg": ecg}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        if self._preprocessed_cache is not None:
            sig = self._preprocessed_cache["ecg"][idx]
        else:
            sig = self._load_mat(it["file_name"])
            sig = _ecg_preprocess(
                sig,
                out_len=self.ecg_len,
                downsample=self.downsample,
                random_crop=False,
            )
        ecg = torch.from_numpy(sig).unsqueeze(0)
        return {"ecg": ecg, "prompt": it["prompt"], "gt_answer": it["gt_answer"], "label": it["label"]}

def collate_fn(batch, tokenizer, max_length: int = 512):
    ecg = torch.stack([b["ecg"] for b in batch], dim=0)
    prompts = [b["prompt"] for b in batch]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

    return ecg, enc["input_ids"], enc["attention_mask"], [b["gt_answer"] for b in batch], prompts, labels

def load_checkpoint(model: ECGQwenForAF, ckpt_path: str) -> ECGQwenForAF:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "trainable_state_dict" in state:
        model.load_state_dict(state["trainable_state_dict"], strict=False)
    elif "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        raise KeyError("Checkpoint missing 'trainable_state_dict' or 'model_state_dict'.")

    return model

def label_to_text(label: int) -> str:
    return "有房颤。" if label == 1 else "无房颤。"

def evaluate(
    ckpt_path: str,
    val_path: str,
    mat_dir: str,
    llm_name: str = DEFAULT_QWEN_NAME,
    encoder_ckpt: Optional[str] = None,
    val_ecg_npz: Optional[str] = None,
    ecg_len: int = 2400,
    downsample: int = 3,
    max_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    ecg_token_count: int = 16,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECGQwenForAF(
        llm_name=llm_name,
        ecg_ch_in=1,
        ecg_len=ecg_len,
        ecg_encoder_ckpt=encoder_ckpt,
        freeze_encoder=True,
        ecg_token_count=ecg_token_count,
    )
    model = apply_lora_to_llm(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    model = load_checkpoint(model, ckpt_path)
    model.to(device)
    model.eval()

    tokenizer = model.tokenizer

    dataset = EvalDataset(
        val_path,
        mat_dir=mat_dir,
        ecg_len=ecg_len,
        downsample=downsample,
        preprocessed_npz=val_ecg_npz,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length),
    )

    preds: List[int] = []
    gts: List[int] = []
    probs: List[float] = []

    for idx, (ecg, input_ids, attn_mask, gt_answers, prompts, labels) in enumerate(loader):
        with torch.no_grad():
            ecg = ecg.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)

            logits = model.classify(
                ecg=ecg,
                input_ids=input_ids,
                attention_mask=attn_mask,
            )
            prob = torch.sigmoid(logits).squeeze(0).item()
            pred_label = int(prob >= 0.5)
            gt_label = int(labels.item())

        preds.append(pred_label)
        gts.append(gt_label)
        probs.append(prob)

        if idx % 20 == 0:
            print(f"[{idx}/{len(dataset)}]")
            print("Prompt:", prompts[0])
            print("Predicted:", label_to_text(pred_label))
            print("GT answer:", gt_answers[0])
            print("-" * 60, flush=True)

    preds_tensor = torch.tensor(preds)
    gts_tensor = torch.tensor(gts)

    acc = (preds_tensor == gts_tensor).float().mean().item()

    tp = ((preds_tensor == 1) & (gts_tensor == 1)).sum().item()
    fp = ((preds_tensor == 1) & (gts_tensor == 0)).sum().item()
    fn = ((preds_tensor == 0) & (gts_tensor == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("Accuracy:", acc) 
    print("Precision:", precision) # TP / (TP + FP)
    print("Recall:", recall)       # TP / (TP + FN)
    print("F1:", f1)               # 2 * (Precision * Recall) / (Precision + Recall)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--mat-dir", type=str, required=True)
    parser.add_argument(
        "--val-ecg-npz",
        type=str,
        default=None,
        help="Optional preprocessed ECG cache aligned with --val JSONL.",
    )
    parser.add_argument("--encoder-ckpt", type=str, default=None)
    parser.add_argument("--llm-name", type=str, default=DEFAULT_QWEN_NAME)
    parser.add_argument("--ecg-len", type=int, default=2400)
    parser.add_argument("--downsample", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--ecg_token_count", type=int, default=16)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(
        ckpt_path=args.ckpt,
        val_path=args.val,
        mat_dir=args.mat_dir,
        llm_name=args.llm_name,
        encoder_ckpt=args.encoder_ckpt,
        val_ecg_npz=args.val_ecg_npz,
        ecg_len=args.ecg_len,
        downsample=args.downsample,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        ecg_token_count=args.ecg_token_count,
    )

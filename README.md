# ECG-AF-LLM-Classification

install:

```shell
git clone https://github.com/qwq-y/ECG-AF-LLM-Classification
conda env create -f environment.yml
conda activate ecg
```

generate and balance the dataset:

```shell
python -m src.task2.build_llm_dataset --cv 0

python -m data.llm_cv0.cut_jsonl

# python -m src.task2.balance_llm_dataset \
#   --input data/llm_cv0/mm_instructions_train_cv0.jsonl \
#   --output data/llm_cv0/mm_instructions_train_cv0_posx10.jsonl \
#   --factor 10

python src/task2/preprocess_llm_data.py \
  --train-jsonl data/llm_cv0/mm_instructions_train_cv0.jsonl \
  --val-jsonl data/llm_cv0/mm_instructions_val_cv0_100.jsonl \
  --mat-dir data/training2017 \
  --cv-dir data/cv \
  --output-dir data/preprocessed_cv0 \
  --augment-types noise scale
```

fine-tune:

```shell
export CUDA_VISIBLE_DEVICES=5

python -m src.task2.train \
  --data-root /home/WangQingyang/Documents/ECG-AF-LLM-Classification/data/llm_cv0 \
  --mat-dir /home/WangQingyang/Documents/ECG-AF-LLM-Classification/data/training2017 \
  --encoder-ckpt /home/WangQingyang/Documents/ECG-AF-LLM-Classification/outputs/mscnn/model.pth \
  --output-dir /home/WangQingyang/Documents/ECG-AF-LLM-Classification/outputs/llm_cv0 \
  --train-ecg-npz /home/WangQingyang/Documents/ECG-AF-LLM-Classification/data/preprocessed_cv0/mm_instructions_train_cv0_ecg.npz \
  --val-ecg-npz /home/WangQingyang/Documents/ECG-AF-LLM-Classification/data/preprocessed_cv0/mm_instructions_val_cv0_ecg.npz \
  --batch-size 64 \
  --stage1-epochs 2 \
  --stage2-epochs 4 \
  --cv 0 \
  --lr 1e-4 \
  --ecg_token_count 16 \
  --accuracy-weight 1 \
  --metric-weight 5
  # --resume outputs/llm_cv0/checkpoint-epoch140-20251214-1957.pt
```

evaluate:

```shell
python -m src.task2.eval \
  --ckpt outputs/llm_cv0/checkpoint-stage2-epoch5-16token-20251218-1901.pt \
  --val data/llm_cv0/mm_instructions_val_cv0_100.jsonl \
  --val-ecg-npz data/preprocessed_cv0/mm_instructions_val_cv0_100_ecg.npz \
  --mat-dir data/training2017 \
  --encoder-ckpt outputs/mscnn/model.pth \
  --ecg_token_count 16
```

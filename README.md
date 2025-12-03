# ECG-AF-LLM-Classification

install:

```shell
git clone https://github.com/qwq-y/ECG-AF-LLM-Classification
conda env create -f environment.yml
conda activate ecg
```

generate the dataset:

```shell
python -m src.task2.build_llm_dataset --cv 0
```

fine-tune:


evaluate:


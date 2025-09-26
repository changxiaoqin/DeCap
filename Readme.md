# Diversity-Enhanced and Classification-Aware Prompt Learning for Few-Shot Learning via Stable Diffusion

This repository contains a **PyTorch implementation** of  
**Diversity-Enhanced and Classification-Aware Prompt Learning for Few-Shot Learning via Stable Diffusion**.  

---

## Installation
```
conda create -n DeCap python=3.10
conda activate DeCap
pip install -r requirements.txt
```
##  Dataset Construction

To allow flexible customization and avoid class-order bias in classification, please organize datasets in the following format:

```
dataset/
└── cifar10/
    └── train/
        ├── {class_name1,e.g. dog}/
        │   ├── img1.png
        │   └── ...
        ├── {class_name2,e.g. cat}/
        │   ├── img1.png
        │   └── ...
        └── ...
    └── val/
    └── test/
```

---

##  Training

1. **Prepare prompt pool**

```bash
bash prepare/prepare.sh
```
2. **Generate full pool**

To accelerate training, we need to first generate synthetic dataset about the full prompt pool.

```bash
python generate.py full cifar10 4 10 None
```
3. **Start training**

```bash
python GA_train.py
```

---

##  Dataset Generation

After training, generate a dataset with:

```bash
python generate.py ours cifar10 20 10 /path/to/prompt_dict.pkl
```

---


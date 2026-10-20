# ICMR2026-StickerED
This repository contains the benchmark and code for our ICMR 2026 paper：Sticker-Enriched Empathetic Response Generation: A Role-Aware Pairing Benchmark and A Novel Multimodal Framework

> Wei Zhang, Changhong Jiang, Lulu Wang, Mingting Yu, Siyuan Zhao, and Ronghan Li. 2026. *Sticker-Enriched Empathetic Response Generation: A Role-Aware Pairing Benchmark and A Novel Multimodal Framework.* In International Conference on Multimedia Retrieval (ICMR ’26), June 16–19, 2026, Amsterdam, Netherlands. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3805622.3810708

## Dataset Overview
StickerED is a role-aware multimodal benchmark built on EmpatheticDialogues. Each dialogue turn is paired with emotionally aligned reaction stickers. The dataset provides paired stickers for dialogue history and target listener responses, alongside sampled negative stickers for multimodal ranking evaluation.

## Repository Layout
- StickerED_train.pkl
- StickerED_valid.pkl
- StickerED_test.pkl
- datasets/SER_Dataset/Images/`<pack>`/`<sticker>`.jpg

**SER_Dataset**: The sticker images are sourced from *SER30K: A Large-Scale Dataset for Sticker Emotion Recognition*. Please contact the authors of SER30K to obtain permission before using the sticker images.

## Dataset Fields
Each sample stored in `StickerED_*.pkl` is a dictionary with the following fields:

| Field              | Type            | Description                                           |
| ------------------ | --------------- | ----------------------------------------------------- |
| `convid_new`       | str             | Unique conversation ID, e.g., `conv_0`                |
| `emotion`          | str             | Emotion label of the dialogue context, e.g., `sentimental` |
| `role`             | list[str]       | Speaker role for each turn (`speaker` / `listener`)    |
| `history`          | list[str]       | Full dialogue history                                 |
| `target`           | str             | Ground-truth empathetic listener response             |
| `situation`        | str             | Contextual situation description from EmpatheticDialogues |
| `history_sticker`  | list[dict/None] | Stickers paired with each history turn; entries can be `None` |
| `target_sticker`   | dict/None       | Sticker paired with the target response; may be `None` |
| `negative_sticker` | list[str]       | Paths of negative stickers for ranking tasks          |

A sticker dictionary contains four keys:
- `filename`: sticker image file name
- `source_path`: absolute path to sticker image
- `score`: matching score between sticker and utterance
- `target_emotion`: integer emotion category label

### Load and Inspect One Sample
```python
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

path = "./MERG/StickerED/StickerED_train.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

# 取一条样本（与上面输出的 conv_0 一致；也可改成你要的索引）
item = data[0] if isinstance(data, (list, tuple)) else data
if isinstance(item, dict) and "convid_new" in item and "history" not in item:
    item = data[next(i for i, d in enumerate(data) if d["convid_new"] == "conv_0")]

convid = item["convid_new"]
roles  = item["role"]
hist   = item["history"]
hst    = item["history_sticker"]   # 与 history 一一对应，元素可能为 None 或 dict
tgt    = item["target_sticker"]
tgt_txt= item["target"]
neg    = item["negative_sticker"]

def load_img(p):
    try:
        return Image.open(p).convert("RGB")
    except Exception as e:
        print(f"[加载失败] {p}: {e}")
        return None

# ---- 主图：对话 + 各轮贴纸 ----
n_turns = len(hist) + 1  # +1 为目标回复轮
fig, axes = plt.subplots(n_turns, 2, figsize=(12, 1.6 * n_turns))
for ax in axes.flatten():
    ax.axis("off")

for i, (role, txt) in enumerate(zip(roles, hist)):
    r, c = (i, 0), (i, 1)
    axes[r[0], r[1]].set_title(f"[{i}] {role}")
    axes[r[0], r[1]].text(0.02, 0.6, txt, fontsize=10, wrap=True,
                          verticalalignment="center")
    st = hst[i]
    if isinstance(st, dict):  # 该轮有贴纸
        img = load_img(str(st["source_path"]))
        if img:
            axes[c[0], c[1]].imshow(img)
            axes[c[0], c[1]].set_title(
                f"score={st['score']:.3f} | emo={st['target_emotion']}")
    else:
        axes[c[0], c[1]].text(0.5, 0.5, "no sticker", ha="center",
                              va="center", color="gray")

# 目标回复轮
r, c = (n_turns - 1, 0), (n_turns - 1, 1)
axes[r[0], r[1]].set_title(f"[target] listener")
axes[r[0], r[1]].text(0.02, 0.6, tgt_txt, fontsize=10, wrap=True,
                      verticalalignment="center")
img = load_img(str(tgt["source_path"]))
if img:
    axes[c[0], c[1]].imshow(img)
    axes[c[0], c[1]].set_title(f"target | score={tgt['score']:.3f} | emo={tgt['target_emotion']}")

fig.suptitle(f"convid: {convid} | emotion: {item['emotion']}")
plt.tight_layout()
plt.show()

# ---- 负例贴纸（一屏展示）----
n_neg = len(neg)
fig2, axs2 = plt.subplots(1, n_neg, figsize=(2.2 * n_neg, 2.2))
if n_neg == 1:
    axs2 = [axs2]
for j, p in enumerate(neg):
    img = load_img(str(p))
    axs2[j].axis("off")
    if img:
        axs2[j].imshow(img)
        axs2[j].set_title(f"neg[{j}]")
fig2.suptitle(f"{convid} — negative stickers")
plt.tight_layout()
plt.show()
```
<img width="1189" height="945" alt="image" src="https://github.com/user-attachments/assets/9044b0b5-d20e-4f86-b344-a11a78114b95" />
<img width="1913" height="220" alt="image" src="https://github.com/user-attachments/assets/0787a3e6-4f8d-4206-8ec5-4568138ce3da" />


## Citation

@inproceedings{zhang2026sticker,
  author    = {Zhang, Wei and Jiang, Changhong and Wang, Lulu and Yu, Mingting and Zhao, Siyuan and Li, Ronghan},
  title     = {Sticker-Enriched Empathetic Response Generation: A Role-Aware Pairing Benchmark and A Novel Multimodal Framework},
  booktitle = {Proceedings of the International Conference on Multimedia Retrieval (ICMR '26)},
  year      = {2026},
  address   = {Amsterdam, Netherlands},
  publisher = {ACM},
  pages     = {10},
  doi       = {10.1145/3805622.3810708}
}


## Acknowledgements

Our dataset is built upon the EmpatheticDialogues dataset and SER sticker image collection. We thank the authors for releasing these resources.

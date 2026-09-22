"""Leakage-safe synthetic and NPZ interaction datasets."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PreparedData:
    arrays: dict
    metadata: dict


class SequenceDataset(Dataset):
    def __init__(self, arrays, indices):
        self.arrays, self.indices = arrays, np.asarray(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, position):
        row = int(self.indices[position])
        return tuple(torch.as_tensor(self.arrays[key][row], dtype=torch.long)
                     for key in ("target_item", "target_category", "history_item",
                                 "history_category", "time_gap", "label"))


def _remap(values):
    _, inverse = np.unique(values, return_inverse=True)
    return inverse.astype(np.int64) + 1


def synthetic_events(config):
    rng = np.random.default_rng(config.get("seed", 2026))
    users = int(config.get("users", 500)); interactions = int(config.get("interactions_per_user", 120))
    categories = int(config.get("categories", 20)); items_per_category = int(config.get("items_per_category", 20))
    rows = []
    for user in range(users):
        stable = rng.choice(categories, size=min(3, categories), replace=False)
        short = int(rng.choice(stable))
        for step in range(interactions):
            if step and step % 25 == 0:
                short = int(rng.choice(stable if rng.random() < .75 else categories))
            if rng.random() < .55: category = short
            elif rng.random() < .82: category = int(rng.choice(stable))
            else: category = int(rng.integers(categories))
            item = category * items_per_category + int(rng.integers(items_per_category))
            rows.append((user, item, category, step, 1))
    return np.asarray(rows, dtype=np.int64)


def load_events(config):
    if config.get("kind", "synthetic") == "synthetic":
        return synthetic_events(config)
    source = np.load(Path(config["path"]))
    required = ("user_id", "item_id", "category_id", "timestamp")
    missing = [key for key in required if key not in source]
    if missing: raise ValueError(f"missing NPZ arrays: {missing}")
    label = source["label"] if "label" in source else np.ones(len(source["user_id"]), np.int64)
    return np.column_stack([source[key] for key in required] + [label])


def prepare(config):
    events = load_events(config).copy()
    events[:, 0] = _remap(events[:, 0]); events[:, 1] = _remap(events[:, 1]); events[:, 2] = _remap(events[:, 2])
    order = np.lexsort((events[:, 3], events[:, 0])); events = events[order]
    history_length = int(config.get("history_length", 200)); negatives = int(config.get("negatives", 1))
    rng = np.random.default_rng(config.get("seed", 2026) + 17)
    num_items, num_categories = int(events[:,1].max()), int(events[:,2].max())
    item_category = np.zeros(num_items + 1, np.int64)
    for item, category in events[:,1:3]: item_category[item] = category
    category_items = {category:np.flatnonzero(item_category == category) for category in range(1,num_categories+1)}
    hard_ratio = float(config.get("hard_negative_ratio", 0.0))
    output = {key: [] for key in ("target_item", "target_category", "history_item", "history_category", "time_gap", "label", "split")}
    for user in np.unique(events[:,0]):
        group = events[events[:,0] == user]
        positives = []
        for position, (_, item, category, timestamp, label) in enumerate(group):
            prior = positives[-history_length:]
            if prior:
                hitem = np.zeros(history_length, np.int64); hcat = np.zeros(history_length, np.int64); gap = np.zeros(history_length, np.int64)
                start = history_length-len(prior)
                hitem[start:] = [x[0] for x in prior]; hcat[start:] = [x[1] for x in prior]
                raw_gap = np.asarray([max(0, timestamp-x[2]) for x in prior])
                gap[start:] = np.minimum(31, np.floor(np.log2(raw_gap+1)).astype(np.int64)+1)
                split = 0 if position < .8*len(group) else 1 if position < .9*len(group) else 2
                candidates = [(int(item), int(category), int(label))]
                if label == 1:
                    seen = {x[0] for x in positives}; seen.add(int(item))
                    for _ in range(negatives):
                        pool = category_items[int(category)] if rng.random() < hard_ratio else None
                        negative = int(rng.choice(pool)) if pool is not None and len(pool) > 1 else int(rng.integers(1, num_items+1))
                        while negative in seen: negative = int(rng.integers(1, num_items+1))
                        candidates.append((negative, int(item_category[negative]), 0))
                for target, target_category, target_label in candidates:
                    for key, value in (("target_item",target),("target_category",target_category),("history_item",hitem.copy()),("history_category",hcat.copy()),("time_gap",gap.copy()),("label",target_label),("split",split)):
                        output[key].append(value)
            if label == 1: positives.append((int(item), int(category), int(timestamp)))
    arrays = {key: np.asarray(value) for key,value in output.items()}
    metadata = {"num_items":num_items, "num_categories":num_categories,
                "history_length":history_length, "examples":len(arrays["label"]),
                "positive_rate":float(arrays["label"].mean()),
                "split_examples":{name:int(np.sum(arrays["split"]==value)) for name,value in (("train",0),("valid",1),("test",2))},
                "average_history_length":float(np.mean(np.count_nonzero(arrays["history_item"],axis=1)))}
    return PreparedData(arrays, metadata)

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


class IndexedSequenceDataset(Dataset):
    """Materialize only the current batch history from a shared positive log."""
    def __init__(self,store,indices):self.store,self.indices=store,np.asarray(indices,dtype=np.int64)
    def __len__(self):return len(self.indices)
    def __getitem__(self,position):
        row=int(self.indices[position]);store=self.store;user=int(store["user"][row]);end=int(store["positive_offset"][user]+store["prior_count"][row]);begin=max(int(store["positive_offset"][user]),end-store["history_length"]);history_rows=store["positive_rows"][begin:end]
        length=store["history_length"];items=np.zeros(length,np.int64);categories=np.zeros(length,np.int64);gaps=np.zeros(length,np.int64);start=length-len(history_rows)
        items[start:]=store["item"][history_rows];categories[start:]=store["category"][history_rows]
        delta=np.maximum(0,store["timestamp"][row]-store["timestamp"][history_rows]);gaps[start:]=np.minimum(31,np.floor(np.log2(delta+1)).astype(np.int64)+1)
        return (torch.tensor(store["item"][row],dtype=torch.long),torch.tensor(store["category"][row],dtype=torch.long),torch.from_numpy(items).long(),torch.from_numpy(categories).long(),torch.from_numpy(gaps).long(),torch.tensor(store["label"][row],dtype=torch.long))


@dataclass
class IndexedPreparedData:
    store:dict
    splits:dict
    metadata:dict
    def dataset(self,part):return IndexedSequenceDataset(self.store,self.splits[part])


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


def prepare_indexed(config):
    """Build O(events) storage for full-scale labeled impression logs."""
    source=np.load(Path(config["path"]));required=("user_id","item_id","category_id","timestamp","label")
    missing=[key for key in required if key not in source]
    if missing:raise ValueError(f"indexed data requires labeled arrays: {missing}")
    user=_remap(source["user_id"]).astype(np.int32);item=_remap(source["item_id"]).astype(np.int32);category=_remap(source["category_id"]).astype(np.int16);timestamp=np.asarray(source["timestamp"],np.int64);label=np.asarray(source["label"],np.int8)
    order=np.lexsort((timestamp,user));user,item,category,timestamp,label=[value[order] for value in (user,item,category,timestamp,label)]
    users=int(user.max());prior_count=np.zeros(len(user),np.int32);positive_offset=np.zeros(users+2,np.int64);positive_parts=[];splits={name:[] for name in ("train","valid","test")}
    boundaries=np.r_[0,np.flatnonzero(np.diff(user))+1,len(user)]
    for begin,end in zip(boundaries[:-1],boundaries[1:]):
        local_positive=np.flatnonzero(label[begin:end]==1)+begin;positive_parts.append(local_positive);positive_offset[int(user[begin])]=sum(len(part) for part in positive_parts[:-1])
        prior_count[begin:end]=np.searchsorted(local_positive,np.arange(begin,end),side="left")
        n=end-begin;split1=begin+int(.8*n);split2=begin+int(.9*n);splits["train"].append(np.arange(begin,split1));splits["valid"].append(np.arange(split1,split2));splits["test"].append(np.arange(split2,end))
    positive_rows=np.concatenate(positive_parts).astype(np.int64);positive_offset[users+1]=len(positive_rows)
    # Fill offset gaps defensively; remapping normally makes users contiguous.
    for uid in range(users,0,-1):
        if positive_offset[uid]==0 and uid!=1:positive_offset[uid]=positive_offset[uid+1]
    split_arrays={name:np.concatenate(parts).astype(np.int64) for name,parts in splits.items()}
    minimum=int(config.get("minimum_history",1));eligible=prior_count>=minimum
    split_arrays={name:rows[eligible[rows]] for name,rows in split_arrays.items()}
    store=dict(user=user,item=item,category=category,timestamp=timestamp,label=label,prior_count=prior_count,positive_rows=positive_rows,positive_offset=positive_offset,history_length=int(config.get("history_length",200)))
    metadata={"num_items":int(item.max()),"num_categories":int(category.max()),"history_length":store["history_length"],"examples":int(sum(map(len,split_arrays.values()))),"positive_rate":float(label.mean()),"split_examples":{k:len(v) for k,v in split_arrays.items()},"average_history_length":float(np.minimum(prior_count,store["history_length"])[eligible].mean()),"storage":"indexed_dynamic_history"}
    return IndexedPreparedData(store,split_arrays,metadata)


def prepare(config):
    if config.get("kind")=="indexed_npz":return prepare_indexed(config)
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

import copy
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import SequenceDataset
from .metrics import binary_metrics
from .models import build_model


def resolve_device(name):
    if name != "auto": return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); labels=[]; logits=[]
    for batch in loader:
        if isinstance(batch,dict):
            batch={key:value.to(device) for key,value in batch.items()};label=batch.pop("label");labels.extend(label.cpu().numpy().tolist());logits.extend(model(**batch).float().cpu().numpy().tolist())
        else:
            batch=[value.to(device) for value in batch];labels.extend(batch[-1].cpu().numpy().tolist());logits.extend(model(*batch[:-1]).float().cpu().numpy().tolist())
    return binary_metrics(labels, logits)


def train_one(name, prepared, config, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device = resolve_device(config.get("device", "auto"));batch_size=int(config.get("batch_size",128))
    if hasattr(prepared,"dataset"):
        datasets={part:prepared.dataset(part) for part in ("train","valid","test")}
    else:
        arrays=prepared.arrays;indices={part:np.flatnonzero(arrays["split"]==value) for part,value in (("train",0),("valid",1),("test",2))};datasets={part:SequenceDataset(arrays,rows) for part,rows in indices.items()}
    loaders={part:DataLoader(dataset,batch_size=batch_size,shuffle=part=="train",num_workers=0) for part,dataset in datasets.items()}
    model=build_model(name,prepared.metadata,config["model"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get("lr",1e-3),weight_decay=config.get("weight_decay",1e-5))
    best_auc=-1.; best=None; curve=[]; started=time.perf_counter()
    for epoch in range(1,int(config.get("epochs",5))+1):
        model.train(); total=0.; seen=0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            if isinstance(batch,dict):
                batch={key:value.to(device) for key,value in batch.items()};labels=batch.pop("label");logits=model(**batch)
            else:
                batch=[value.to(device) for value in batch];labels=batch[-1];logits=model(*batch[:-1])
            loss=model.loss(logits,labels)
            loss.backward(); optimizer.step(); total += float(loss)*len(logits); seen += len(logits)
        valid=evaluate(model,loaders["valid"],device); curve.append({"epoch":epoch,"train_loss":total/seen,**valid})
        if valid["auc"] > best_auc: best_auc=valid["auc"]; best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best); test=evaluate(model,loaders["test"],device)
    return {"model":name,"seed":seed,"device":str(device),"parameters":sum(p.numel() for p in model.parameters()),"seconds":time.perf_counter()-started,"test":test,"curve":curve}

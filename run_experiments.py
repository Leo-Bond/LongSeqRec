import argparse
import csv
import json
from pathlib import Path
import numpy as np
from longseqrec.data import prepare
from longseqrec.models import MODEL_NAMES
from longseqrec.train import train_one


def main():
    parser=argparse.ArgumentParser(description="Run controlled long-sequence recommendation ablations")
    parser.add_argument("--config",default="configs/local.json"); parser.add_argument("--device")
    args=parser.parse_args(); config=json.loads(Path(args.config).read_text())
    if args.device: config["training"]["device"]=args.device
    prepared=prepare(config["data"]); output=Path(config["output"]); output.mkdir(parents=True,exist_ok=True)
    (output/"data_audit.json").write_text(json.dumps(prepared.metadata,indent=2)+"\n")
    models=config.get("models",list(MODEL_NAMES)); records=[]
    for seed in config.get("seeds",[1,2,3]):
        for model in models:
            print(f"training model={model} seed={seed}",flush=True)
            record=train_one(model,prepared,config["training"],seed); records.append(record)
            (output/f"{model}_seed{seed}.json").write_text(json.dumps(record,indent=2)+"\n")
            print(f"  AUC={record['test']['auc']:.4f} LogLoss={record['test']['logloss']:.4f}",flush=True)
    summary=[]
    for model in models:
        group=[r for r in records if r["model"]==model]
        row={"model":model,"parameters":group[0]["parameters"]}
        for metric in ("auc","logloss","accuracy"):
            values=[r["test"][metric] for r in group]; row[f"{metric}_mean"]=float(np.mean(values)); row[f"{metric}_std"]=float(np.std(values))
        row["seconds_mean"]=float(np.mean([r["seconds"] for r in group])); summary.append(row)
    with (output/"summary.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=summary[0].keys());writer.writeheader();writer.writerows(summary)
    lines=["# LongSeqRec ablation report","",f"Examples: {prepared.metadata['examples']:,}; average observed history: {prepared.metadata['average_history_length']:.1f}; maximum history: {prepared.metadata['history_length']}; seed: {config.get('seeds',[1])}.","","| Model | Parameters | AUC | LogLoss | Accuracy | Seconds |","| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(summary,key=lambda x:x["auc_mean"],reverse=True):
        lines.append(f"| {row['model']} | {row['parameters']:,} | {row['auc_mean']:.4f} ± {row['auc_std']:.4f} | {row['logloss_mean']:.4f} ± {row['logloss_std']:.4f} | {row['accuracy_mean']:.4f} | {row['seconds_mean']:.1f} |")
    lines += ["","Single-seed results are engineering comparisons, not statistical significance claims."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n"); (output/"records.json").write_text(json.dumps(records,indent=2)+"\n")
    print(output/"REPORT.md")

if __name__=="__main__": main()

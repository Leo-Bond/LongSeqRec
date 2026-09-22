"""Convert KuaiRand-1K CSV files into LongSeqRec's compact NPZ format."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def first_tag(value):
    value = (value or "").strip().strip("[]")
    if not value: return "__unknown__"
    return value.replace("'", "").replace('"', "").split(",")[0].strip()


def convert(root, output, max_users=1000, max_events=None, positive_field="long_view"):
    root, output = Path(root), Path(output)
    logs = [root/"data"/"log_standard_4_08_to_4_21_1k.csv",
            root/"data"/"log_standard_4_22_to_5_08_1k.csv"]
    features=root/"data"/"video_features_basic_1k.csv"
    missing=[str(path) for path in logs+[features] if not path.exists()]
    if missing: raise FileNotFoundError(f"missing KuaiRand files: {missing}")
    selected_users=set(); rows=[]; videos=set()
    for log in logs:
        with log.open(newline="") as handle:
            for row in csv.DictReader(handle):
                user=int(row["user_id"])
                if user not in selected_users:
                    if len(selected_users) >= max_users: continue
                    selected_users.add(user)
                video=int(row["video_id"]);videos.add(video)
                rows.append((user,video,int(row["time_ms"]),int(float(row[positive_field]))))
                if max_events is not None and len(rows)>=max_events:break
        if max_events is not None and len(rows)>=max_events:break
    video_category={};category_ids={"__unknown__":0}
    with features.open(newline="") as handle:
        for row in csv.DictReader(handle):
            video=int(row["video_id"])
            if video not in videos:continue
            tag=first_tag(row.get("tag"))
            if tag not in category_ids:category_ids[tag]=len(category_ids)
            video_category[video]=category_ids[tag]
    arrays={"user_id":np.asarray([r[0] for r in rows],np.int64),
            "item_id":np.asarray([r[1] for r in rows],np.int64),
            "category_id":np.asarray([video_category.get(r[1],0) for r in rows],np.int64),
            "timestamp":np.asarray([r[2] for r in rows],np.int64),
            "label":np.asarray([r[3] for r in rows],np.int8)}
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    audit={"source":str(root),"logs":[path.name for path in logs],"positive_field":positive_field,
           "users":len(selected_users),"events":len(rows),"items":len(videos),"categories":len(category_ids),
           "positive_rate":float(arrays["label"].mean()),"unknown_category_rate":float(np.mean(arrays["category_id"]==0))}
    output.with_suffix(".audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2));return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",required=True)
    parser.add_argument("--output",default="data/kuairand_1k.npz");parser.add_argument("--max-users",type=int,default=1000)
    parser.add_argument("--max-events",type=int);parser.add_argument("--positive-field",choices=("is_click","long_view"),default="long_view")
    args=parser.parse_args();convert(args.root,args.output,args.max_users,args.max_events,args.positive_field)

if __name__=="__main__":main()


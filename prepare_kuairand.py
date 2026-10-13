"""Convert KuaiRand-1K into a leakage-safe rich-feature NPZ file."""
import argparse, csv, json, math
from array import array
from datetime import datetime
from pathlib import Path
import numpy as np

MAX_TAGS = 3
USER_FEATURE_COLUMNS = (
    "user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
    "follow_user_num_range", "fans_user_num_range", "friend_user_num_range", "register_days_range",
    *(f"onehot_feat{i}" for i in range(18)),
)

def integer(value, default=0):
    try:return int(float(value))
    except (TypeError, ValueError):return default

def duration_bucket(value):
    value=max(0,integer(value));return min(31,int(math.log2(value+1))+1) if value else 0

def play_ratio_bucket(play,duration):
    play,duration=max(0,integer(play)),max(0,integer(duration))
    return min(15,int(min(play/duration,2.)*8)+1) if duration else 0

def category(value,mapping):
    value=str(value or "__unknown__").strip() or "__unknown__"
    if value not in mapping:mapping[value]=len(mapping)+1
    return mapping[value]

def behavior_code(row):
    names=("is_click","is_like","is_follow","is_comment","is_forward","is_hate","is_profile_enter")
    return 1+sum((integer(row.get(name))>0)<<bit for bit,name in enumerate(names))

def read_users(path,selected):
    maps=[{} for _ in USER_FEATURE_COLUMNS];ids=[];values=[]
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                uid=integer(row.get("user_id"),-1)
                if uid in selected:
                    ids.append(uid);values.append([category(row.get(name),maps[i]) for i,name in enumerate(USER_FEATURE_COLUMNS)])
    return (np.asarray(ids,np.int64),np.asarray(values,np.int32).reshape(-1,len(USER_FEATURE_COLUMNS)),
            np.asarray([len(x) for x in maps],np.int32))

def convert(root,output,max_users=1000,max_events=None,positive_field="long_view"):
    root,output=Path(root),Path(output);data=root/"data"
    logs=[data/"log_standard_4_08_to_4_21_1k.csv",data/"log_standard_4_22_to_5_08_1k.csv"]
    basic=data/"video_features_basic_1k.csv";users_file=data/"user_features_1k.csv"
    missing=[str(x) for x in logs+[basic] if not x.exists()]
    if missing:raise FileNotFoundError(f"missing KuaiRand files: {missing}")
    selected,videos=set(),set();weekday_cache={};names=("user_id","item_id","timestamp","label","hour","weekday","tab","is_rand","behavior_code","play_ratio_bucket")
    event={name:array("q" if name in ("user_id","item_id","timestamp") else "B") for name in names}
    for log in logs:
        with log.open(newline="") as handle:
            for row in csv.DictReader(handle):
                uid=integer(row.get("user_id"),-1)
                if uid not in selected:
                    if len(selected)>=max_users:continue
                    selected.add(uid)
                vid=integer(row.get("video_id"),-1);videos.add(vid);date=str(row.get("date", ""));hourmin=integer(row.get("hourmin"),-1)
                if date not in weekday_cache:
                    try:weekday_cache[date]=datetime.strptime(date,"%Y%m%d").weekday()+1
                    except ValueError:weekday_cache[date]=0
                weekday=weekday_cache[date]
                values=(uid,vid,integer(row.get("time_ms")),integer(row.get(positive_field)),hourmin//100 if hourmin>=0 else 0,weekday,
                        max(0,integer(row.get("tab")))+1,max(0,integer(row.get("is_rand")))+1,behavior_code(row),play_ratio_bucket(row.get("play_time_ms"),row.get("duration_ms")))
                for name,value in zip(names,values):event[name].append(value)
                if max_events is not None and len(event["user_id"])>=max_events:break
        if max_events is not None and len(event["user_id"])>=max_events:break

    maps={name:{} for name in ("author","video_type","upload_type","music_type","tag")}
    video={name:array("i") for name in ("video_feature_item_id","video_author","video_type","video_upload_type","video_duration","video_music_type")};tags=array("i")
    with basic.open(newline="") as handle:
        for row in csv.DictReader(handle):
            vid=integer(row.get("video_id"),-1)
            if vid not in videos:continue
            values=(vid,category(row.get("author_id"),maps["author"]),category(row.get("video_type"),maps["video_type"]),
                    category(row.get("upload_type"),maps["upload_type"]),duration_bucket(row.get("video_duration")),category(row.get("music_type"),maps["music_type"]))
            for name,value in zip(video,values):video[name].append(value)
            raw=[x.strip() for x in str(row.get("tag") or "").strip("[]").replace("'","").replace('"',"").split(",") if x.strip()]
            encoded=[category(x,maps["tag"]) for x in raw[:MAX_TAGS]];tags.extend(encoded+[0]*(MAX_TAGS-len(encoded)))

    profile_ids,user_features,user_cards=read_users(users_file,selected)
    dtypes={"user_id":np.int64,"item_id":np.int64,"timestamp":np.int64,"label":np.int8,"hour":np.uint8,"weekday":np.uint8,"tab":np.uint8,"is_rand":np.uint8,"behavior_code":np.uint8,"play_ratio_bucket":np.uint8}
    arrays={name:np.asarray(values,dtypes[name]) for name,values in event.items()}
    arrays.update({name:np.asarray(values,np.int32) for name,values in video.items()})
    arrays.update(video_tags=np.asarray(tags,np.int32).reshape(-1,MAX_TAGS),profile_user_id=profile_ids,user_features=user_features,user_feature_cardinalities=user_cards)
    # Preserve a first-tag category for old configs and simple baselines.
    category_id=np.zeros(len(arrays["item_id"]),np.int32)
    if len(arrays["video_feature_item_id"]):
        order=np.argsort(arrays["video_feature_item_id"]);ids=arrays["video_feature_item_id"][order];pos=np.searchsorted(ids,arrays["item_id"]);found=pos<len(ids);found[found]&=ids[pos[found]]==arrays["item_id"];category_id[found]=arrays["video_tags"][order[pos[found]],0]
    arrays["category_id"]=category_id
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    audit={"source":str(root),"logs":[x.name for x in logs],"positive_field":positive_field,"users":len(selected),"events":len(arrays["user_id"]),"items":len(videos),"categories":len(maps["tag"])+1,"authors":len(maps["author"]),"video_types":len(maps["video_type"]),"music_types":len(maps["music_type"]),"user_feature_fields":len(USER_FEATURE_COLUMNS),"positive_rate":float(arrays["label"].mean()),"unknown_category_rate":float(np.mean(category_id==0)),"leakage_policy":"post-exposure feedback is used only when the event is in prior history"}
    output.with_suffix(".audit.json").write_text(json.dumps(audit,indent=2)+"\n");print(json.dumps(audit,indent=2));return audit

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",required=True);parser.add_argument("--output",default="data/kuairand_1k_rich.npz");parser.add_argument("--max-users",type=int,default=1000);parser.add_argument("--max-events",type=int);parser.add_argument("--positive-field",choices=("is_click","long_view"),default="long_view")
    args=parser.parse_args();convert(args.root,args.output,args.max_users,args.max_events,args.positive_field)
if __name__=="__main__":main()

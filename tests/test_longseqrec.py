import unittest
import numpy as np
import torch
import csv
import tempfile
from pathlib import Path
from longseqrec.data import prepare
from longseqrec.models import MODEL_NAMES,build_model
from prepare_kuairand import convert


class LongSeqRecTests(unittest.TestCase):
    def setUp(self):
        self.data=prepare({"kind":"synthetic","seed":3,"users":5,"interactions_per_user":12,"categories":4,"items_per_category":3,"history_length":8,"negatives":1})

    def test_history_is_padded_and_nonempty(self):
        history=self.data.arrays["history_item"]
        self.assertEqual(history.shape[1],8); self.assertTrue((history!=0).any(1).all())

    def test_negative_samples_share_history(self):
        arrays=self.data.arrays
        np.testing.assert_array_equal(arrays["history_item"][0],arrays["history_item"][1])
        self.assertEqual(set(arrays["label"][:2]),{0,1})

    def test_all_models_forward_backward(self):
        rows=slice(0,4); a=self.data.arrays
        inputs=[torch.as_tensor(a[key][rows],dtype=torch.long) for key in ("target_item","target_category","history_item","history_category","time_gap")]
        for name in MODEL_NAMES:
            model=build_model(name,self.data.metadata,{"dim":8,"topk":3,"recent_length":3,"time_buckets":32})
            output=model(*inputs);self.assertEqual(tuple(output.shape),(4,));self.assertTrue(torch.isfinite(output).all());output.sum().backward()

    def test_dare_has_extra_match_table(self):
        shared=build_model("dare_shared",self.data.metadata,{"dim":8})
        dare=build_model("dare",self.data.metadata,{"dim":8})
        self.assertIs(shared.item_match,shared.item_repr);self.assertIsNot(dare.item_match,dare.item_repr)
        self.assertGreater(sum(p.numel() for p in dare.parameters()),sum(p.numel() for p in shared.parameters()))

    def test_dare_match_embedding_receives_gradient(self):
        a=self.data.arrays;rows=slice(0,4)
        inputs=[torch.as_tensor(a[key][rows],dtype=torch.long) for key in ("target_item","target_category","history_item","history_category","time_gap")]
        model=build_model("dare",self.data.metadata,{"dim":8,"topk":3,"recent_length":3})
        model(*inputs).sum().backward()
        self.assertIsNotNone(model.item_match.weight.grad)
        self.assertGreater(float(model.item_match.weight.grad.abs().sum()),0.)

    def test_dare_shared_parameters_have_matched_initialization(self):
        config={"dim":8,"topk":3,"recent_length":3}
        torch.manual_seed(11);shared=build_model("dare_shared",self.data.metadata,config)
        torch.manual_seed(11);dare=build_model("dare",self.data.metadata,config)
        for key,value in shared.state_dict().items():
            self.assertTrue(torch.equal(value,dare.state_dict()[key]),key)

    def test_twin_cp_gsu_and_esu_share_relevance_scores(self):
        a=self.data.arrays;rows=slice(0,4);model=build_model("twin",self.data.metadata,{"dim":8,"topk":3,"twin_heads":4})
        item=torch.as_tensor(a["target_item"][rows]);category=torch.as_tensor(a["target_category"][rows]);history=torch.as_tensor(a["history_item"][rows]);history_category=torch.as_tensor(a["history_category"][rows]);gap=torch.as_tensor(a["time_gap"][rows]);mask=history.ne(0)
        query=model.encoder(item,category);inherent=model.encoder(history,history_category);cross=model.encoder.time(gap);values=model.encoder(history,history_category,gap)
        score=model.interest.relevance_scores(query,inherent,cross,mask);index=model.interest.search(score,mask);vector,weight=model.interest.exact(score,values,mask,index)
        expected=score.mean(-1).masked_fill(~mask,-1e9).topk(3,1).indices
        self.assertTrue(torch.equal(index,expected));self.assertEqual(tuple(score.shape),(4,8,4));self.assertEqual(tuple(vector.shape),(4,8));self.assertEqual(tuple(weight.shape),(4,3,4))

    def test_twin_all_projections_receive_gradient(self):
        a=self.data.arrays;rows=slice(0,4);inputs=[torch.as_tensor(a[key][rows],dtype=torch.long) for key in ("target_item","target_category","history_item","history_category","time_gap")]
        model=build_model("twin",self.data.metadata,{"dim":8,"topk":3,"twin_heads":4});model(*inputs).sum().backward()
        for name in ("query_projection","inherent_key_projection","cross_bias_projection","value_projection","output_projection"):
            gradient=getattr(model.interest,name).weight.grad;self.assertIsNotNone(gradient,name);self.assertGreater(float(gradient.abs().sum()),0.,name)

    def test_indexed_dataset_builds_strict_prior_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"events.npz";np.savez(path,user_id=[1,1,1,2,2],item_id=[10,11,12,20,21],category_id=[1,1,2,2,2],timestamp=[1,2,3,1,2],label=[1,0,1,1,0])
            indexed=prepare({"kind":"indexed_npz","path":str(path),"history_length":3,"minimum_history":1})
            dataset=indexed.dataset("test");sample=dataset[0]
            self.assertEqual(sample["history_item"].shape[0],3);self.assertNotIn(int(sample["target_item"]),sample["history_item"].tolist());self.assertTrue(all(value.dtype==torch.long for value in sample.values()))

    def test_kuairand_converter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data";data.mkdir();headers=["user_id","video_id","date","hourmin","time_ms","is_click","is_like","is_follow","is_comment","is_forward","is_hate","long_view","play_time_ms","duration_ms","is_profile_enter","is_rand","tab"]
            for name,offset in (("log_standard_4_08_to_4_21_1k.csv",0),("log_standard_4_22_to_5_08_1k.csv",2)):
                with (data/name).open("w",newline="") as handle:
                    writer=csv.DictWriter(handle,fieldnames=headers);writer.writeheader()
                    writer.writerow({"user_id":0,"video_id":10+offset,"date":20220409,"hourmin":930,"time_ms":100+offset,"is_click":1,"is_like":1,"long_view":1,"play_time_ms":8000,"duration_ms":10000,"is_rand":0,"tab":1})
            with (data/"video_features_basic_1k.csv").open("w",newline="") as handle:
                fields=["video_id","author_id","video_type","upload_type","video_duration","music_type","tag"];writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
                writer.writerows([{"video_id":10,"author_id":7,"video_type":"NORMAL","upload_type":"ShortImport","video_duration":10000,"music_type":2,"tag":"39,40"},{"video_id":12,"author_id":8,"video_type":"NORMAL","upload_type":"Kmovie","video_duration":20000,"music_type":3,"tag":"41"}])
            with (data/"user_features_1k.csv").open("w",newline="") as handle:
                fields=["user_id","user_active_degree","is_lowactive_period","is_live_streamer","is_video_author","follow_user_num_range","fans_user_num_range","friend_user_num_range","register_days_range"]+[f"onehot_feat{i}" for i in range(18)];writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerow({name:(0 if name!="user_active_degree" else "full_active") for name in fields})
            output=root/"out.npz";audit=convert(root,output)
            loaded=np.load(output);self.assertEqual(len(loaded["user_id"]),2);self.assertEqual(audit["categories"],4);self.assertEqual(loaded["video_tags"].shape,(2,3));self.assertEqual(loaded["user_features"].shape,(1,26));self.assertGreater(int(loaded["behavior_code"][0]),1)
            indexed=prepare({"kind":"indexed_npz","path":str(output),"history_length":2,"minimum_history":1});sample=indexed.dataset("test")[0];model=build_model("twin",indexed.metadata,{"dim":8,"topk":2,"twin_heads":4}).eval();features={key:value.unsqueeze(0) for key,value in sample.items() if key!="label"};prediction=model(**features);self.assertEqual(tuple(prediction.shape),(1,))

if __name__=="__main__":unittest.main()

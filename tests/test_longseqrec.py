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
            if key != "item_match.weight": self.assertTrue(torch.equal(value,dare.state_dict()[key]),key)

    def test_kuairand_converter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data";data.mkdir();headers=["user_id","video_id","time_ms","is_click","long_view"]
            for name,offset in (("log_standard_4_08_to_4_21_1k.csv",0),("log_standard_4_22_to_5_08_1k.csv",2)):
                with (data/name).open("w",newline="") as handle:
                    writer=csv.DictWriter(handle,fieldnames=headers);writer.writeheader()
                    writer.writerow({"user_id":0,"video_id":10+offset,"time_ms":100+offset,"is_click":1,"long_view":1})
            with (data/"video_features_basic_1k.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=["video_id","tag"]);writer.writeheader()
                writer.writerows([{"video_id":10,"tag":"39,40"},{"video_id":12,"tag":"41"}])
            output=root/"out.npz";audit=convert(root,output)
            loaded=np.load(output);self.assertEqual(len(loaded["user_id"]),2);self.assertEqual(audit["categories"],3)

if __name__=="__main__":unittest.main()

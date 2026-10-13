"""Modular long-sequence CTR models inspired by DIN/SASRec/SIM/TWIN/DARE."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES=("mean","din","sasrec","sim","twin","dare","dare_shared","dare_no_long","dare_no_time")

def masked_softmax(score,mask):
    score=score.masked_fill(~mask,-1e9);weight=torch.softmax(score,1)*mask
    return weight/weight.sum(1,keepdim=True).clamp_min(1e-12)

def batched_gather(values,index):return values.gather(1,index.unsqueeze(-1).expand(-1,-1,values.shape[-1]))

class EventEncoder(nn.Module):
    """Compose ID, content, time and prior-feedback fields into an event vector."""
    def __init__(self,num_items,num_categories,dim,time_buckets=32,**sizes):
        super().__init__();self.item=nn.Embedding(num_items+1,dim,padding_idx=0);self.category=nn.Embedding(num_categories+1,dim,padding_idx=0);self.time=nn.Embedding(time_buckets+1,dim,padding_idx=0)
        self.side=nn.ModuleDict({name:nn.Embedding(max(0,int(size))+1,dim,padding_idx=0) for name,size in {
            "author":sizes.get("num_authors",0),"video_type":sizes.get("num_video_types",0),"upload_type":sizes.get("num_upload_types",0),"duration":sizes.get("num_durations",0),"music_type":sizes.get("num_music_types",0),"tag":sizes.get("num_tags",0),"behavior":sizes.get("num_behaviors",0),"play_ratio":sizes.get("num_play_ratios",0)}.items()})
        self.norm=nn.LayerNorm(dim)
    def forward(self,item,category,gap=None,use_time=True,author=None,video_type=None,upload_type=None,duration=None,music_type=None,tags=None,behavior=None,play_ratio=None,use_feedback=True):
        value=self.item(item)+self.category(category)
        for name,feature in (("author",author),("video_type",video_type),("upload_type",upload_type),("duration",duration),("music_type",music_type)):
            if feature is not None:value=value+self.side[name](feature)
        if tags is not None:
            tag_value=self.side["tag"](tags);tag_mask=tags.ne(0).unsqueeze(-1);value=value+(tag_value*tag_mask).sum(-2)/tag_mask.sum(-2).clamp_min(1)
        if gap is not None and use_time:value=value+self.time(gap)
        if use_feedback:
            if behavior is not None:value=value+self.side["behavior"](behavior)
            if play_ratio is not None:value=value+self.side["play_ratio"](play_ratio)
        return self.norm(value)

class FeatureGroupEncoder(nn.Module):
    """Sum field-specific categorical embeddings for user or request features."""
    def __init__(self,cardinalities,dim):
        super().__init__();self.tables=nn.ModuleList([nn.Embedding(max(0,int(size))+1,dim,padding_idx=0) for size in cardinalities]);self.norm=nn.LayerNorm(dim)
    def forward(self,features,reference):
        if features is None or not self.tables:return torch.zeros_like(reference)
        value=torch.zeros_like(reference)
        for index,table in enumerate(self.tables):value=value+table(features[:,index])
        return self.norm(value)

class Dice(nn.Module):
    def __init__(self,width):super().__init__();self.norm=nn.BatchNorm1d(width);self.alpha=nn.Parameter(torch.zeros(width))
    def forward(self,x):
        shape=x.shape;x=x.reshape(-1,shape[-1]);p=torch.sigmoid(self.norm(x));return (p*x+(1-p)*self.alpha*x).reshape(shape)

class LocalActivationUnit(nn.Module):
    """DIN target-aware nonlinear relevance function."""
    def __init__(self,dim):super().__init__();self.net=nn.Sequential(nn.Linear(4*dim,4*dim),Dice(4*dim),nn.Linear(4*dim,2*dim),Dice(2*dim),nn.Linear(2*dim,1))
    def forward(self,query,keys):
        query=query[:,None].expand_as(keys);return self.net(torch.cat((query,keys,query-keys,query*keys),-1)).squeeze(-1)

class ExactSearchUnit(nn.Module):
    def __init__(self,dim):super().__init__();self.activation=LocalActivationUnit(dim)
    def forward(self,query,keys,values,mask,prior=None):
        score=self.activation(query,keys)+(prior if prior is not None else 0);weight=masked_softmax(score,mask)
        return (values*weight.unsqueeze(-1)).sum(1),score,weight

class SASRecEncoder(nn.Module):
    def __init__(self,dim,length,layers=2,heads=2):
        super().__init__();self.position=nn.Embedding(length,dim);layer=nn.TransformerEncoderLayer(dim,heads,4*dim,.1,"gelu",batch_first=True,norm_first=True);self.transformer=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False);self.norm=nn.LayerNorm(dim)
    def forward(self,events,mask):
        order=torch.argsort((~mask).to(torch.int8),dim=1,stable=True);events=batched_gather(events,order);mask=mask.gather(1,order);safe=mask.clone();safe[~mask.any(1),0]=True
        length=events.shape[1];position=torch.arange(length,device=events.device);causal=torch.triu(torch.ones(length,length,dtype=torch.bool,device=events.device),1)
        encoded=self.transformer(events+self.position(position)[None],mask=causal,src_key_padding_mask=~safe);last=mask.sum(1).sub(1).clamp_min(0)
        return self.norm(encoded[torch.arange(len(events),device=events.device),last])*mask.any(1,keepdim=True)

@dataclass
class InterestOutput:
    vector:torch.Tensor
    auxiliary:torch.Tensor
    selected:torch.Tensor|None=None

class MeanInterest(nn.Module):
    def forward(self,query,events,mask,**kwargs):return InterestOutput((events*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1),events.new_zeros(()))

class DINInterest(nn.Module):
    def __init__(self,dim,recent):super().__init__();self.recent=recent;self.esu=ExactSearchUnit(dim)
    def forward(self,query,events,mask,**kwargs):
        events,mask=events[:,-self.recent:],mask[:,-self.recent:];vector,_,_=self.esu(query,events,events,mask);return InterestOutput(vector,events.new_zeros(()))

class SIMInterest(nn.Module):
    """Hard category General Search Unit plus nonlinear Exact Search Unit."""
    def __init__(self,dim,topk):super().__init__();self.topk=topk;self.esu=ExactSearchUnit(dim)
    def forward(self,query,events,mask,history_categories,target_category,**kwargs):
        relevant=history_categories.eq(target_category[:,None])&mask;fallback=~relevant.any(1);relevant[fallback]=mask[fallback]
        recency=torch.arange(events.shape[1],device=events.device)[None].expand_as(history_categories);index=recency.masked_fill(~relevant,-1).topk(min(self.topk,events.shape[1]),1).indices
        selected=batched_gather(events,index);selected_mask=mask.gather(1,index);vector,_,_=self.esu(query,selected,selected,selected_mask)
        return InterestOutput(vector,events.new_zeros(()),index)

class TWINInterest(nn.Module):
    """Paper-style consistency-preserved multi-head target attention.

    Inherent video features form Q/K. User-item cross features are compressed
    into a scalar bias for every head. CP-GSU and ESU call the same relevance
    function and therefore share both architecture and parameter values.
    """
    def __init__(self,dim,topk,heads=4):
        super().__init__()
        if dim%heads:raise ValueError("TWIN dimension must be divisible by heads")
        self.topk,self.heads,self.head_dim=topk,heads,dim//heads
        self.query_projection=nn.Linear(dim,dim,bias=False)
        self.inherent_key_projection=nn.Linear(dim,dim,bias=False)
        self.cross_bias_projection=nn.Linear(dim,heads,bias=False)
        self.value_projection=nn.Linear(dim,dim,bias=False)
        self.output_projection=nn.Linear(dim,dim,bias=False)

    def relevance_scores(self,query,inherent_events,cross_events,mask):
        """The identical relevance metric used by CP-GSU and ESU."""
        batch,length,_=inherent_events.shape
        q=self.query_projection(query).reshape(batch,self.heads,self.head_dim)
        k=self.inherent_key_projection(inherent_events).reshape(batch,length,self.heads,self.head_dim)
        inherent=(k*q[:,None]).sum(-1)/(self.head_dim**.5)
        cross=self.cross_bias_projection(cross_events)
        return (inherent+cross).masked_fill(~mask.unsqueeze(-1),-1e9)

    def search(self,head_scores,mask):
        """CP-GSU cuts the lifelong sequence with the shared head scores."""
        relevance=head_scores.mean(-1).masked_fill(~mask,-1e9)
        return relevance.topk(min(self.topk,relevance.shape[1]),1).indices

    def exact(self,head_scores,values,mask,index):
        """ESU reuses CP-GSU scores and performs multi-head value pooling."""
        batch,length,_=values.shape
        projected=self.value_projection(values).reshape(batch,length,self.heads,self.head_dim)
        selected_values=projected.gather(1,index[:,:,None,None].expand(-1,-1,self.heads,self.head_dim))
        selected_scores=head_scores.gather(1,index[:,:,None].expand(-1,-1,self.heads))
        selected_mask=mask.gather(1,index)
        selected_scores=selected_scores.masked_fill(~selected_mask.unsqueeze(-1),-1e9)
        weight=torch.softmax(selected_scores,dim=1)*selected_mask.unsqueeze(-1)
        weight=weight/weight.sum(1,keepdim=True).clamp_min(1e-12)
        heads=(selected_values*weight.unsqueeze(-1)).sum(1).reshape(batch,-1)
        return self.output_projection(heads),weight

    def forward(self,query,events,mask,inherent_events,cross_events,**kwargs):
        head_scores=self.relevance_scores(query,inherent_events,cross_events,mask)
        index=self.search(head_scores,mask)
        vector,weight=self.exact(head_scores,events,mask,index)
        entropy=-(weight.clamp_min(1e-8).log()*weight).sum(1).mean()
        return InterestOutput(vector,-1e-4*entropy,index)

class DAREInterest(nn.Module):
    """Score in an attention space while aggregating in a representation space."""
    def __init__(self,dim,topk):super().__init__();self.topk=topk;self.scorer=LocalActivationUnit(dim)
    def forward(self,query,events,mask,match_query,match_events,**kwargs):
        all_score=self.scorer(match_query,match_events).masked_fill(~mask,-1e9);score,index=all_score.topk(min(self.topk,events.shape[1]),1);values=batched_gather(events,index);selected_mask=mask.gather(1,index);weight=masked_softmax(score,selected_mask)
        selected_match=batched_gather(match_events,index);alignment=1-F.cosine_similarity(values.detach(),selected_match,dim=-1);auxiliary=1e-3*(alignment*selected_mask).sum()/selected_mask.sum().clamp_min(1)
        return InterestOutput((values*weight.unsqueeze(-1)).sum(1),auxiliary,index)

class CTRTower(nn.Module):
    def __init__(self,dim):super().__init__();self.net=nn.Sequential(nn.Linear(6*dim,4*dim),nn.BatchNorm1d(4*dim),nn.PReLU(),nn.Dropout(.1),nn.Linear(4*dim,2*dim),nn.PReLU(),nn.Linear(2*dim,1))
    def forward(self,interest,target,user,context):return self.net(torch.cat((interest,target,interest-target,interest*target,user,context),-1)).squeeze(-1)

class LongSeqModel(nn.Module):
    def __init__(self,kind,num_items,num_categories,dim=32,topk=20,recent_length=20,time_buckets=32,history_length=200,twin_heads=4,feature_sizes=None,user_feature_cardinalities=(),context_cardinalities=(24,7,15,2)):
        super().__init__();self.kind=kind;self.recent_length=recent_length;feature_sizes=feature_sizes or {};self.encoder=EventEncoder(num_items,num_categories,dim,time_buckets,**feature_sizes);self.user_encoder=FeatureGroupEncoder(user_feature_cardinalities,dim);self.context_encoder=FeatureGroupEncoder(context_cardinalities,dim);self.cross=nn.ModuleDict({"same_author":nn.Embedding(2,dim),"same_music":nn.Embedding(2,dim),"tag_overlap":nn.Embedding(4,dim),"duration_difference":nn.Embedding(32,dim)});self.auxiliary_loss=None
        if kind=="mean":self.interest=MeanInterest()
        elif kind=="din":self.interest=DINInterest(dim,recent_length)
        elif kind=="sasrec":self.interest=SASRecEncoder(dim,history_length,heads=2 if dim%2==0 else 1)
        elif kind=="sim":self.interest=SIMInterest(dim,topk)
        elif kind=="twin":self.interest=TWINInterest(dim,topk,twin_heads)
        elif kind in ("dare","dare_shared","dare_no_long","dare_no_time"):self.interest=DAREInterest(dim,topk)
        else:raise ValueError(kind)
        self.tower=CTRTower(dim);self.match_encoder=None
        # Extra DARE parameters are constructed last so shared parameters have
        # identical initialization in matched ablations.
        if kind in ("dare","dare_no_long","dare_no_time"):
            self.match_encoder=EventEncoder(num_items,num_categories,dim,time_buckets,**feature_sizes);self.match_encoder.load_state_dict(self.encoder.state_dict())
    @property
    def item_repr(self):return self.encoder.item
    @property
    def item_match(self):return self.match_encoder.item if self.match_encoder is not None else self.encoder.item
    def forward(self,target_item,target_category,history_item,history_category,time_gap,target_author=None,target_video_type=None,target_upload_type=None,target_duration=None,target_music_type=None,target_tags=None,history_author=None,history_video_type=None,history_upload_type=None,history_duration=None,history_music_type=None,history_tags=None,history_behavior=None,history_play_ratio=None,user_features=None,context=None):
        mask=history_item.ne(0);use_time=self.kind!="dare_no_time"
        target_kwargs=dict(author=target_author,video_type=target_video_type,upload_type=target_upload_type,duration=target_duration,music_type=target_music_type,tags=target_tags,use_feedback=False)
        history_kwargs=dict(author=history_author,video_type=history_video_type,upload_type=history_upload_type,duration=history_duration,music_type=history_music_type,tags=history_tags,behavior=history_behavior,play_ratio=history_play_ratio)
        target=self.encoder(target_item,target_category,**target_kwargs);events=self.encoder(history_item,history_category,time_gap,use_time,**history_kwargs)
        if self.kind=="sasrec":output=InterestOutput(self.interest(events,mask),events.new_zeros(()))
        else:
            kwargs=dict(history_categories=history_category,target_category=target_category)
            if self.kind=="twin":
                inherent=self.encoder(history_item,history_category,author=history_author,video_type=history_video_type,upload_type=history_upload_type,duration=history_duration,music_type=history_music_type,tags=history_tags,use_feedback=False)
                cross=self.encoder.time(time_gap) if use_time else torch.zeros_like(events)
                if target_author is not None and history_author is not None:cross=cross+self.cross["same_author"](history_author.eq(target_author[:,None]).long())
                if target_music_type is not None and history_music_type is not None:cross=cross+self.cross["same_music"](history_music_type.eq(target_music_type[:,None]).long())
                if target_tags is not None and history_tags is not None:
                    overlap=(history_tags.unsqueeze(-1).eq(target_tags[:,None,None,:]) & history_tags.unsqueeze(-1).ne(0)).any(-1).sum(-1).clamp_max(3)
                    cross=cross+self.cross["tag_overlap"](overlap)
                if target_duration is not None and history_duration is not None:cross=cross+self.cross["duration_difference"]((history_duration-target_duration[:,None]).abs().clamp_max(31))
                kwargs.update(inherent_events=inherent,cross_events=cross)
            if self.kind=="dare_shared":kwargs.update(match_query=target,match_events=events)
            elif self.match_encoder is not None:
                match_events=self.match_encoder(history_item,history_category,time_gap,use_time,**history_kwargs)
                match_query=self.match_encoder(target_item,target_category,**target_kwargs)
                if self.kind=="dare_no_long":
                    events=events[:,-self.recent_length:];mask=mask[:,-self.recent_length:];match_events=match_events[:,-self.recent_length:]
                kwargs.update(match_query=match_query,match_events=match_events)
            output=self.interest(target,events,mask,**kwargs)
        self.auxiliary_loss=output.auxiliary;user=self.user_encoder(user_features,target);request=self.context_encoder(context,target);return self.tower(output.vector,target,user,request)
    def loss(self,logits,labels):return F.binary_cross_entropy_with_logits(logits,labels.float())+self.auxiliary_loss

def build_model(name,metadata,config):
    sizes={name:metadata.get(name,0) for name in ("num_authors","num_video_types","num_upload_types","num_durations","num_music_types","num_tags","num_behaviors","num_play_ratios")}
    return LongSeqModel(name,metadata["num_items"],metadata["num_categories"],config.get("dim",32),config.get("topk",20),config.get("recent_length",20),config.get("time_buckets",32),metadata["history_length"],config.get("twin_heads",4),sizes,metadata.get("user_feature_cardinalities",()),metadata.get("context_cardinalities",(24,7,15,2)))

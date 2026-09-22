"""Capacity-controlled long-history CTR models."""
import torch
from torch import nn

MODEL_NAMES = ("mean", "din", "sasrec", "sim", "twin", "dare", "dare_shared", "dare_no_long", "dare_no_time")


def masked_softmax(scores, mask):
    scores = scores.masked_fill(~mask, -1e9)
    weights = torch.softmax(scores, dim=1) * mask
    return weights / weights.sum(1, keepdim=True).clamp_min(1e-12)


class LongSeqModel(nn.Module):
    def __init__(self, kind, num_items, num_categories, dim=32, topk=20,
                 recent_length=20, time_buckets=32, history_length=200):
        super().__init__()
        if kind not in MODEL_NAMES:
            raise ValueError(f"unknown model: {kind}")
        self.kind, self.topk, self.recent_length = kind, topk, recent_length
        self.item_repr = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.category = nn.Embedding(num_categories + 1, dim, padding_idx=0)
        self.time = nn.Embedding(time_buckets + 1, dim, padding_idx=0)
        self.attention = nn.Sequential(nn.Linear(4*dim, 2*dim), nn.PReLU(), nn.Linear(2*dim, 1))
        self.head = nn.Sequential(nn.Linear(4*dim, 2*dim), nn.ReLU(), nn.Linear(2*dim, 1))
        if kind == "sasrec":
            self.position = nn.Embedding(history_length, dim)
            layer = nn.TransformerEncoderLayer(dim, 2 if dim % 2 == 0 else 1,
                4*dim, dropout=.1, activation="gelu", batch_first=True, norm_first=True)
            self.sequence = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.sequence_norm = nn.LayerNorm(dim)
        # Construct the extra DARE table last so all shared parameters have
        # identical initial values under a fixed seed in matched ablations.
        if kind == "dare":
            self.item_match = nn.Embedding(num_items + 1, dim, padding_idx=0)
            # Start from the same geometry as the representation table, then
            # let the CTR objective specialize the two roles independently.
            with torch.no_grad():
                self.item_match.weight.copy_(self.item_repr.weight)
        else:
            self.item_match = self.item_repr

    def _event_repr(self, items, categories, gaps, use_time=True):
        value = self.item_repr(items) + self.category(categories)
        if use_time:
            value = value + self.time(gaps)
        return value

    def _din(self, query, values, mask):
        expanded = query[:, None].expand_as(values)
        score = self.attention(torch.cat((expanded, values, expanded-values, expanded*values), -1)).squeeze(-1)
        return (values * masked_softmax(score, mask).unsqueeze(-1)).sum(1)

    def _learned_select(self, target_items, history_items, mask, k):
        query = self.item_match(target_items)
        keys = self.item_match(history_items)
        scores = (keys * query[:, None]).sum(-1) / query.shape[-1] ** .5
        scores = scores.masked_fill(~mask, -1e9)
        k = min(k, history_items.shape[1])
        selected_scores, index = scores.topk(k, dim=1)
        return index, selected_scores

    @staticmethod
    def _gather(values, index):
        return values.gather(1, index.unsqueeze(-1).expand(-1, -1, values.shape[-1]))

    def encode(self, target_items, target_categories, history_items,
               history_categories, time_gaps):
        mask = history_items.ne(0)
        use_time = self.kind != "dare_no_time"
        values = self._event_repr(history_items, history_categories, time_gaps, use_time)
        query = self.item_repr(target_items) + self.category(target_categories)
        if self.kind == "mean":
            return (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        if self.kind == "sasrec":
            length = history_items.shape[1]
            # Histories are left padded for retrieval models. Causal attention
            # needs valid tokens first, otherwise padded queries can see no key
            # and produce NaNs. Stable sorting preserves event order.
            order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
            values = values.gather(1, order.unsqueeze(-1).expand_as(values))
            mask = mask.gather(1, order)
            safe_mask = mask.clone();safe_mask[~mask.any(1),0] = True
            positions = torch.arange(length, device=history_items.device)
            causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=history_items.device), 1)
            encoded = self.sequence(values + self.position(positions)[None], mask=causal,
                                    src_key_padding_mask=~safe_mask)
            last = mask.sum(1).sub(1).clamp_min(0)
            result = self.sequence_norm(encoded[torch.arange(len(encoded), device=encoded.device), last])
            return result * mask.any(1,keepdim=True)
        if self.kind in ("din", "dare_no_long"):
            return self._din(query, values[:, -self.recent_length:], mask[:, -self.recent_length:])
        if self.kind == "sim":
            # SIM general search: cheap category equality before exact attention.
            related = history_categories.eq(target_categories[:, None]) & mask
            fallback = ~related.any(1)
            related[fallback] = mask[fallback]
            positions = torch.arange(history_items.shape[1], device=history_items.device)[None].expand_as(history_items)
            rank_score = positions.masked_fill(~related, -1)
            index = rank_score.topk(min(self.topk, history_items.shape[1]), 1).indices
        else:
            index, selected_scores = self._learned_select(target_items, history_items, mask, self.topk)
        selected_values = self._gather(values, index)
        selected_mask = mask.gather(1, index)
        if self.kind != "sim":
            # Keep the selected relevance scores in the differentiable path.
            # Top-k membership is discrete, but both TWIN's shared table and
            # DARE's independent matching table now learn from CTR loss.
            weights = masked_softmax(selected_scores, selected_mask)
            return (selected_values * weights.unsqueeze(-1)).sum(1)
        return self._din(query, selected_values, selected_mask)

    def forward(self, target_items, target_categories, history_items,
                history_categories, time_gaps):
        interest = self.encode(target_items, target_categories, history_items,
                               history_categories, time_gaps)
        target = self.item_repr(target_items) + self.category(target_categories)
        features = torch.cat((interest, target, interest-target, interest*target), -1)
        return self.head(features).squeeze(-1)


def build_model(name, metadata, model_config):
    return LongSeqModel(name, metadata["num_items"], metadata["num_categories"],
                        dim=model_config.get("dim", 32), topk=model_config.get("topk", 20),
                        recent_length=model_config.get("recent_length", 20),
                        time_buckets=model_config.get("time_buckets", 32),
                        history_length=metadata["history_length"])

# Architecture

LongSeqRec separates event representation, general search, exact aggregation,
and prediction. DIN uses a Dice-activated Local Activation Unit over recent
events. SASRec uses two causal Transformer layers. SIM performs cheap category
and recency search before a DIN Exact Search Unit. TWIN implements four-head
target attention with behavior feature splitting. Item, category, author,
video/upload type, duration, music type and multi-tag embeddings are inherent
features that form projected Q/K. Time gap, same-author, same-music-type and
tag-overlap signals form cross features compressed to one scalar bias per head.
CP-GSU ranks the full sequence
and ESU gathers the exact same per-head scores for Softmax value aggregation.
DARE owns separate
attention and representation Event Encoders, initialized equally and trained
with CTR plus a weak alignment loss.

User profile fields and request context (hour, weekday, tab and random-exposure
flag) enter the shared CTR tower. Historical interaction type and play ratio
enrich only strictly prior events, never the current target.

`dare_shared`, `dare_no_long`, and `dare_no_time` isolate representation
decoupling, long history, and time gaps. All models use the same final CTR tower.

## Full-data layout

`indexed_npz` stores every exposure once. Each row records how many positive
events preceded it, while per-user offsets point into one global array of
positive rows. A batch gathers only its last `history_length` prior positives.
Storage is O(events), instead of O(events × history length), and future events
cannot enter history.

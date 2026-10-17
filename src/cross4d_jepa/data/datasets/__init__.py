"""Per-dataset point-cloud-clip loaders (the STUDENT input).

Each provider exposes `records`, `load_full(rec) -> [T,N,3]` and `__getitem__`
returning {points, clip_id, label}; engine.linear_probe.build_labeled_splits wraps
them for pretraining and downstream evaluation.
"""

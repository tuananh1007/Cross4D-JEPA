"""DINOv3 ViT-L/16 per-frame image teacher (teacher-modality ablation).

A frozen DINOv3 ViT-L/16 (LVD-1689M) applied frame-wise. DINOv3 is distributed as the official Meta
torch.hub `.pth` (keys: patch_embed.proj, cls_token, storage_tokens, blocks.*,
norm), so we build the transformers `DINOv3ViTModel` for ViT-L/16 and remap the
checkpoint keys with the official conversion mapping
(transformers convert_dinov3_vit_to_hf.py), then load_state_dict.

DINOv3 prepends CLS + 4 register ("storage") tokens, so the patch grid starts at
token index 1 + num_register_tokens = 5 (NOT 1 like DINOv2). Embed dim 1024
(same as DINOv2-large), so the dense pipeline's random projection is unchanged.

Used once per pretraining run to build the targets (engine.pretrain_dense).
"""
import re

import torch

# Official Meta-original -> HF key remap (transformers convert_dinov3_vit_to_hf.py).
# Applied as ordered regex substitutions on each original key. The final top-level
# "norm.weight"/"norm.bias" need no remap (HF uses the same name).
_KEY_MAP = [
    (r"^cls_token$", "embeddings.cls_token"),
    (r"^mask_token$", "embeddings.mask_token"),
    (r"^storage_tokens$", "embeddings.register_tokens"),
    (r"^patch_embed\.proj", "embeddings.patch_embeddings"),
    (r"^blocks\.(\d+)\.attn\.proj", r"model.layer.\1.attention.o_proj"),
    (r"^blocks\.(\d+)\.attn\.", r"model.layer.\1.attention."),
    (r"^blocks\.(\d+)\.ls(\d+)\.gamma", r"model.layer.\1.layer_scale\2.lambda1"),
    (r"^blocks\.(\d+)\.mlp\.fc1", r"model.layer.\1.mlp.up_proj"),
    (r"^blocks\.(\d+)\.mlp\.fc2", r"model.layer.\1.mlp.down_proj"),
    (r"^blocks\.(\d+)\.norm", r"model.layer.\1.norm"),
]


def _convert_meta_to_hf(state: dict) -> dict:
    """Remap an original DINOv3 ViT-L/16 state_dict to transformers DINOv3ViTModel
    keys. Splits fused attn.qkv into q/k/v (k has NO bias); skips rope buffers."""
    out = {}
    for k, v in state.items():
        if "rope" in k or "periods" in k or "inv_freq" in k:
            continue                                      # recomputed buffer
        if "local_cls_norm" in k or "bias_mask" in k or k.startswith("projectors."):
            continue
        # split fused qkv -> q_proj / k_proj / v_proj (drop k bias)
        m = re.match(r"^blocks\.(\d+)\.attn\.qkv\.(weight|bias)$", k)
        if m:
            i, kind = m.group(1), m.group(2)
            q, kk, vv = torch.chunk(v, 3, dim=0)
            out[f"model.layer.{i}.attention.q_proj.{kind}"] = q
            out[f"model.layer.{i}.attention.v_proj.{kind}"] = vv
            if kind == "weight":                          # k_proj has no bias in HF config
                out[f"model.layer.{i}.attention.k_proj.weight"] = kk
            continue
        nk = k
        for pat, rep in _KEY_MAP:
            nk = re.sub(pat, rep, nk)
        out[nk] = v
    return out


def load_dinov3_vitl16(weights_path: str, device: str = "cuda"):
    """Build transformers DINOv3ViTModel (ViT-L/16, 4 registers) and load the
    official Meta .pth. Returns (model.eval() on device, embed_dim=1024)."""
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    cfg = DINOv3ViTConfig(
        patch_size=16, hidden_size=1024, intermediate_size=4096,
        num_hidden_layers=24, num_attention_heads=16,
        num_register_tokens=4, image_size=224, use_gated_mlp=False,
    )
    model = DINOv3ViTModel(cfg)
    raw = torch.load(weights_path, map_location="cpu")
    state = raw.get("model", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    hf = _convert_meta_to_hf(state)
    if "embeddings.mask_token" in hf and hf["embeddings.mask_token"].dim()==2:
        hf["embeddings.mask_token"] = hf["embeddings.mask_token"].reshape(1, 1, -1)
    # Whether DINOv3ViTModel nests its blocks under a `model.` submodule depends on the
    # transformers version (4.57 is flat; earlier builds nest). The converter above emits
    # the nested form, so pick whichever keying actually matches this instantiation --
    # by counting matches, since a few keys (rope buffers, embeddings) match either way.
    want = set(model.state_dict().keys())
    def _strip(d):
        return {(k[len("model."):] if k.startswith("model.") else k): v for k, v in d.items()}
    def _add(d):
        return {(k if k.startswith("model.") else f"model.{k}"): v for k, v in d.items()}
    cands = {"as-is": hf, "stripped": _strip(hf), "prefixed": _add(hf)}
    best = max(cands, key=lambda n: len(set(cands[n]) & want))
    hf = cands[best]
    missing, unexpected = model.load_state_dict(hf, strict=False)
    # Only rope inv_freq buffers may be "missing"; nothing should be unexpected.
    miss_real = [m for m in missing if "inv_freq" not in m and "rope" not in m]
    if miss_real or unexpected:
        raise RuntimeError(
            f"DINOv3 load mismatch: missing={miss_real[:8]} unexpected={unexpected[:8]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device), 1024


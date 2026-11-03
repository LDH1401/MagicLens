# pylint: disable=missing-function-docstring
"""Exports MagicLens' top-K candidates on FashionIQ for the VLM reranker.

The first stage is zero-shot MagicLens as in the baseline of fiq_experiments.py
(upstream preprocessing, query text "{c1} and {c2}"). For every val query it
keeps the top-K images with their MagicLens scores, the target's rank and the
dev/test split of fiq_experiments.py, and writes:
  <out_dir>/candidates.json  used by vlm_rerank.py (Colab) and vlm_rerank_eval.py
  <out_dir>/bundle.zip       what to upload to Google Drive: candidates.json,
                             vlm_rerank.py and the images the candidates use

Usage (from the magiclens/ folder, after fiq_experiments.py has cached the val
embeddings):
  python vlm_rerank_prepare.py --subtask dress
"""
from argparse import ArgumentParser
import json
import os
import zipfile

import fiq_experiments as fx
from inference import load_model
import numpy as np
from model import MagicLens

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--model_size", default="base", choices=["base", "large"])
  parser.add_argument("--model_path", default="./models/magic_lens_clip_base.pkl")
  parser.add_argument("--subtask", default="dress", choices=list(fx.CATEGORY))
  parser.add_argument("--data_dir", default="./data/fiq")
  parser.add_argument("--cache_dir", default="./cache")
  parser.add_argument("--out_dir", default="./vlm_rerank")
  parser.add_argument("--top_k", type=int, default=50, help="Candidates kept per query.")
  parser.add_argument("--seed", type=int, default=0, help="Seed of the dev/test split.")
  args = parser.parse_args()

  ids, queries = fx.load_fiq(args.data_dir, args.subtask)
  cache = os.path.join(args.cache_dir, f"fiq-{args.subtask}-{args.model_size}")
  meta_path = os.path.join(cache, "meta.json")
  meta = {"ids": ids, "queries": [[q["candidate"], q["target"]] for q in queries]}
  stale = not os.path.exists(meta_path)
  if not stale:
    with open(meta_path) as f:
      stale = json.load(f) != meta
  if stale:
    raise SystemExit("Val cache is missing or stale: run fiq_experiments.py first.")
  index_of = {iid: n for n, iid in enumerate(ids)}
  refs = np.array([index_of[q["candidate"]] for q in queries])
  targets = np.array([index_of[q["target"]] for q in queries])
  perm = np.random.default_rng(args.seed).permutation(len(queries))
  dev = set(perm[:len(perm) // 2].tolist())

  # Zero-shot MagicLens scores from the cached CLIP embeddings.
  img = np.load(os.path.join(cache, "img_original.npy"))
  txt = np.load(os.path.join(cache, "txt_and.npy"))
  empty = np.load(os.path.join(cache, "txt_empty.npy"))
  model, params = load_model(args.model_size, args.model_path)
  fuse = lambda i, t: np.asarray(model.apply(
      params, i, t, method=MagicLens.fuse)["multimodal_embed_norm"])
  scores = fuse(img[refs], txt) @ fuse(img, np.repeat(empty, len(ids), axis=0)).T
  base_ranks = fx.target_ranks(scores, targets)
  top = np.argsort(-scores, axis=1)[:, :args.top_k]
  splits = np.array(["dev" if i in dev else "test" for i in range(len(queries))])
  for s in ("dev", "test"):
    r = fx.recalls(base_ranks[splits == s])
    print(f"Baseline {s}: R@10 {r['R@10']:.1f}, R@50 {r['R@50']:.1f}, R_avg {r['R_avg']:.1f}; "
          f"target in top-{args.top_k}: {np.mean(base_ranks[splits == s] < args.top_k) * 100:.1f}%")

  cat = fx.CATEGORY[args.subtask]
  candidates = {
      "subtask": args.subtask, "category": cat, "model_size": args.model_size,
      "seed": args.seed, "top_k": args.top_k, "n_index": len(ids),
      "queries": [{
          "q": i, "split": str(splits[i]),
          "candidate": q["candidate"], "target": q["target"], "captions": q["captions"],
          "text": fx.TEMPLATES["and"].format(c1=q["captions"][0], c2=q["captions"][1], cat=cat),
          "base_rank": int(base_ranks[i]),
          "top": [ids[j] for j in top[i]],
          "ml_scores": [round(float(s), 6) for s in scores[i, top[i]]],
      } for i, q in enumerate(queries)]}

  os.makedirs(args.out_dir, exist_ok=True)
  cand_path = os.path.join(args.out_dir, "candidates.json")
  with open(cand_path, "w") as f:
    json.dump(candidates, f, ensure_ascii=False)
  needed = sorted({q["candidate"] for q in queries} | {ids[j] for j in top.ravel()})
  bundle = os.path.join(args.out_dir, "bundle.zip")
  with zipfile.ZipFile(bundle, "w") as z:
    z.write(cand_path, "candidates.json", compress_type=zipfile.ZIP_DEFLATED)
    z.write(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_rerank.py"),
            "vlm_rerank.py", compress_type=zipfile.ZIP_DEFLATED)
    for iid in needed:  # PNGs are already compressed
      z.write(os.path.join(args.data_dir, "images", iid + ".png"), f"images/{iid}.png")
  print(f"{len(queries)} queries x top-{args.top_k} = {len(queries) * args.top_k} pairs, "
        f"{len(needed)} images.")
  print(f"Wrote {cand_path} and {bundle} ({os.path.getsize(bundle) / 1e6:.0f} MB).")

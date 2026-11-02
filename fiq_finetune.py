# pylint: disable=missing-function-docstring
"""Fine-tunes MagicLens' fusion layers on the FashionIQ train split.

The CLIP backbone stays frozen, so the CLIP embeddings of the train images and
captions are computed once and cached. Only the multimodal encoder (4
self-attention layers), the attention pooler and the temperature are trained,
with the paper's contrastive loss: other targets in the batch and the query
images themselves (with empty text) are the negatives.

The learning rate and epoch are chosen on the dev half of the val queries, and
the result is reported on the same test half as fiq_experiments.py.

Usage (from the magiclens/ folder, after fiq_experiments.py has cached the val
embeddings):
  python fiq_finetune.py --subtask dress
"""
from argparse import ArgumentParser
import json
import os
import time

from clip.simple_tokenizer import default_bpe
import fiq_experiments as fx
from inference import load_model
import jax
import jax.numpy as jnp
import numpy as np
import optax
from model import MagicLens
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer

TRAINABLE = ("multimodal_encoder", "contrastive_multimodal_pooler")


def load_split(data_dir, subtask, split):
  ids, queries = fx.load_fiq(data_dir, subtask, split)
  index_of = {iid: n for n, iid in enumerate(ids)}
  refs = np.array([index_of[q["candidate"]] for q in queries])
  targets = np.array([index_of[q["target"]] for q in queries])
  return ids, queries, refs, targets


def negatives_mask(refs, targets):
  """[B, 2B] True where a negative is in fact the positive image.

  Columns are the batch's targets, then its query images. A column is masked
  when it shows the row's target image, except the row's own positive.
  """
  cols = np.concatenate([targets, refs])
  mask = cols[None, :] == targets[:, None]
  mask[np.arange(len(targets)), np.arange(len(targets))] = False
  return mask


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--model_size", default="base", choices=["base", "large"])
  parser.add_argument("--model_path", default="./models/magic_lens_clip_base.pkl")
  parser.add_argument("--subtask", default="dress", choices=list(fx.CATEGORY))
  parser.add_argument("--data_dir", default="./data/fiq")
  parser.add_argument("--cache_dir", default="./cache")
  parser.add_argument("--output", default="./output")
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--epochs", type=int, default=15)
  parser.add_argument("--lrs", type=float, nargs="+", default=[1e-5, 3e-5, 1e-4])
  parser.add_argument("--seed", type=int, default=0, help="Seed of the dev/test split.")
  args = parser.parse_args()

  cache = os.path.join(args.cache_dir, f"fiq-{args.subtask}-{args.model_size}")
  ids, queries, refs, targets = load_split(args.data_dir, args.subtask, "val")
  with open(os.path.join(cache, "meta.json")) as f:
    meta = json.load(f)
  if meta != {"ids": ids, "queries": [[q["candidate"], q["target"]] for q in queries]}:
    raise SystemExit("Val cache is missing or stale: run fiq_experiments.py first.")
  perm = np.random.default_rng(args.seed).permutation(len(queries))
  dev, test = np.sort(perm[:len(perm) // 2]), np.sort(perm[len(perm) // 2:])
  val_img = np.load(os.path.join(cache, "img_original.npy"))
  val_txt = np.load(os.path.join(cache, "txt_and.npy"))
  empty = np.load(os.path.join(cache, "txt_empty.npy"))

  tokenizer = clip_tokenizer.build_tokenizer(bpe_path=default_bpe())
  model, params = load_model(args.model_size, args.model_path)
  enc_img = jax.jit(lambda p, x: model.apply(p, x, method=MagicLens.encode_image))
  enc_txt = jax.jit(lambda p, x: model.apply(p, x, method=MagicLens.encode_text))

  # CLIP embeddings of the train split, cached like the val ones.
  tr_ids, tr_queries, tr_refs, tr_targets = load_split(args.data_dir, args.subtask, "train")
  tr_meta = {"ids": tr_ids, "queries": [[q["candidate"], q["target"]] for q in tr_queries]}
  tr_meta_path = os.path.join(cache, "train_meta.json")
  if os.path.exists(tr_meta_path):
    with open(tr_meta_path) as f:
      if json.load(f) != tr_meta:
        raise SystemExit(f"Train cache in {cache} is stale; delete train_*.npy and re-run.")
  else:
    with open(tr_meta_path, "w") as f:
      json.dump(tr_meta, f)

  def cached(name, compute):
    path = os.path.join(cache, name + ".npy")
    if not os.path.exists(path):
      np.save(path, compute())
    return np.load(path)

  tr_img = cached("train_img_original", lambda: fx.image_embeds(
      lambda x: enc_img(params, x), tr_ids, f"{args.data_dir}/images", "original", 32))
  # Both caption orders, used as augmentation.
  tr_txt = np.stack([cached(f"train_txt_{n}", lambda t=fx.TEMPLATES[n]: fx.text_embeds(
      lambda x: enc_txt(params, x), tokenizer,
      [t.format(c1=q["captions"][0], c2=q["captions"][1], cat="") for q in tr_queries],
      64)) for n in ("and", "swap")])  # [2, Q_train, D]

  init = {"model": {k: params["params"][k] for k in TRAINABLE},
          "log_scale": jnp.log(1 / 0.07)}  # temperature 0.07, as in the paper
  fuse = lambda p, img, txt: model.apply(
      {"params": p["model"]}, img, txt, method=MagicLens.fuse)["multimodal_embed_norm"]

  def loss_fn(p, ref_img, txt, tgt_img, mask):
    e = jnp.broadcast_to(empty, txt.shape)
    q = fuse(p, ref_img, txt)
    cands = jnp.concatenate([fuse(p, tgt_img, e), fuse(p, ref_img, e)])
    logits = q @ cands.T * jnp.exp(p["log_scale"])
    logits = jnp.where(mask, -1e9, logits)
    labels = jnp.arange(len(q))
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

  def ranks_of(p):
    index = fuse(p, val_img, np.repeat(empty, len(ids), axis=0))
    query = fuse(p, val_img[refs], val_txt)
    return fx.target_ranks(np.asarray(query @ index.T), targets)

  base_ranks = ranks_of(init)  # zero-shot MagicLens, same as the baseline
  runs, best = {}, None
  t_start = time.time()
  for lr in args.lrs:
    opt = optax.adamw(lr, weight_decay=1e-4)

    @jax.jit
    def step(p, s, ref_img, txt, tgt_img, mask, opt=opt):
      loss, g = jax.value_and_grad(loss_fn)(p, ref_img, txt, tgt_img, mask)
      u, s = opt.update(g, s, p)
      return optax.apply_updates(p, u), s, loss

    p, s = init, opt.init(init)
    rng = np.random.default_rng(0)
    curve = []
    for epoch in range(1, args.epochs + 1):
      order = rng.permutation(len(tr_queries))
      losses = []
      for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
        b = order[i:i + args.batch_size]
        txt = tr_txt[rng.integers(0, 2, len(b)), b]
        p, s, loss = step(p, s, tr_img[tr_refs[b]], txt, tr_img[tr_targets[b]],
                          negatives_mask(tr_refs[b], tr_targets[b]))
        losses.append(float(loss))
      r = ranks_of(p)
      dev_r = fx.recalls(r[dev])["R_avg"]
      curve.append({"epoch": epoch, "loss": float(np.mean(losses)), "dev_R_avg": dev_r})
      print(f"lr {lr:g} epoch {epoch:2d}: loss {np.mean(losses):.3f}, dev R_avg {dev_r:.1f}")
      if best is None or dev_r > best[0]:
        best = (dev_r, lr, epoch, r)
    runs[f"{lr:g}"] = curve
  train_time = time.time() - t_start

  _, best_lr, best_epoch, ft_ranks = best
  ev = lambda r: {"dev": fx.recalls(r[dev]), "test": fx.recalls(r[test])}
  hits = lambda r: {"R@10": r < 10, "R_avg": ((r < 10).astype(float) + (r < 50)) / 2}
  boot = {c: fx.paired_bootstrap(hits(base_ranks[test])[c], hits(ft_ranks[test])[c])
          for c in ("R@10", "R_avg")}

  md = [f"# Tinh chỉnh khối hợp nhất MagicLens-{args.model_size} trên FashionIQ {args.subtask} train",
        "",
        f"Train: {len(tr_queries)} truy vấn, {len(tr_ids)} ảnh (backbone CLIP đóng băng; học "
        f"{sum(x.size for x in jax.tree_util.tree_leaves(init['model'])) / 1e6:.1f}M tham số). "
        f"Đánh giá: dev {len(dev)} / test {len(test)} truy vấn của tập val, như fiq_experiments.py. "
        f"Chọn tốc độ học và epoch theo R_avg của dev.",
        "", fx.md_table([("MagicLens zero-shot (baseline)", ev(base_ranks)),
                         (f"Tinh chỉnh (lr={best_lr:g}, epoch {best_epoch})", ev(ft_ranks))]),
        "",
        f"Chênh lệch trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%): "
        f"R@10 {boot['R@10'][0]:+.1f} [{boot['R@10'][1]:+.1f}, {boot['R@10'][2]:+.1f}], "
        f"R_avg {boot['R_avg'][0]:+.1f} [{boot['R_avg'][1]:+.1f}, {boot['R_avg'][2]:+.1f}] điểm.",
        "", "## Đường học (dev R_avg theo epoch)", "",
        "| Epoch | " + " | ".join(f"lr={lr}" for lr in runs) + " |",
        "|---|" + "---|" * len(runs)]
  for e in range(args.epochs):
    md.append(f"| {e + 1} | " + " | ".join(f"{runs[lr][e]['dev_R_avg']:.1f}" for lr in runs) + " |")
  md += ["", f"Thời gian huấn luyện (CPU, mọi lần thử, gồm đánh giá mỗi epoch): {train_time / 60:.1f} phút."]
  md = "\n".join(md) + "\n"
  os.makedirs(args.output, exist_ok=True)
  stem = os.path.join(args.output, f"fiq-{args.subtask}-{args.model_size}_finetune")
  with open(stem + ".md", "w") as f:
    f.write(md)
  with open(stem + ".json", "w") as f:
    json.dump({"baseline": ev(base_ranks), "finetuned": ev(ft_ranks),
               "choice": {"lr": best_lr, "epoch": best_epoch}, "curves": runs,
               "bootstrap_test_vs_baseline": boot, "train_time_s": train_time,
               "n_train_queries": len(tr_queries), "n_train_images": len(tr_ids)},
              f, ensure_ascii=False, indent=1)
  print(md)
  print("Results written to", stem + ".md")

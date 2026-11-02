# pylint: disable=missing-function-docstring
"""Training-free improvements of MagicLens on FashionIQ, with an ablation study.

  1. Preprocessing that keeps the aspect ratio (data_utils.process_img modes).
  2. Query ensemble: average the query embeddings of several phrasings.
  3. Rerank: re-score the top-K candidates by fusing the MagicLens score with
     CLIP text->image and image->image scores from MagicLens' own backbone.

CLIP embeddings, the expensive part, are cached under --cache_dir, so re-runs
only redo the cheap fusion and scoring. Every choice (preprocessing mode,
phrasing, rerank weights) is made on a fixed dev half of the queries and
reported on the held-out test half.

Usage (from the magiclens/ folder):
  python fiq_experiments.py --subtask dress --modes original
  python fiq_experiments.py --subtask dress
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

from clip.simple_tokenizer import default_bpe
from data_utils import process_img
from inference import load_model
import jax
import numpy as np
from model import MagicLens
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer
from tqdm import tqdm

MODES = ("original", "crop", "pad")
# "and" is the upstream query text (data_utils.build_fiq_dataset).
TEMPLATES = {
    "and": "{c1} and {c2}",
    "swap": "{c2} and {c1}",
    "that": "a {cat} that {c1} and {c2}",
    "same": "same {cat} but {c1} and {c2}",
}
CATEGORY = {"dress": "dress", "shirt": "shirt", "toptee": "top"}
KS = (1, 5, 10, 50)
COLS = [f"R@{k}" for k in KS] + ["R_avg"]
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 2)  # rerank grid for each weight
# MagicLens-{size} (CLIP) R@10, R@50 on the full FashionIQ val set (Table 12).
PAPER = {
    ("base", "dress"): (21.5, 41.3), ("base", "shirt"): (27.3, 48.8),
    ("base", "toptee"): (30.2, 52.3), ("large", "dress"): (25.5, 46.1),
    ("large", "shirt"): (32.7, 53.8), ("large", "toptee"): (34.0, 57.7),
}


def l2_normalize(x):
  return x / np.linalg.norm(x, axis=-1, keepdims=True)


def load_fiq(data_dir, subtask, split="val"):
  """Index ids and queries, skipping missing images like build_fiq_dataset."""
  have = lambda i: os.path.exists(f"{data_dir}/images/{i}.png")
  with open(f"{data_dir}/image_splits/split.{subtask}.{split}.json") as f:
    ids = [i for i in json.load(f) if have(i)]
  with open(f"{data_dir}/captions/cap.{subtask}.{split}.json") as f:
    queries = [q for q in json.load(f)
               if have(q["candidate"]) and have(q["target"])]
  return ids, queries


def run_padded(fn, x, batch_size):
  """Runs a jitted fn on a batch padded to batch_size (one compiled shape)."""
  n = len(x)
  if n < batch_size:
    x = np.concatenate([x, np.zeros((batch_size - n,) + x.shape[1:], x.dtype)])
  return np.asarray(fn(x))[:n]


def image_embeds(enc, ids, image_dir, mode, batch_size):
  load = lambda i: process_img(os.path.join(image_dir, i + ".png"), 224, mode)
  out = []
  with ThreadPoolExecutor() as ex:
    for i in tqdm(range(0, len(ids), batch_size), desc=f"Image embeds ({mode})"):
      x = np.concatenate(list(ex.map(load, ids[i:i + batch_size])))
      out.append(run_padded(enc, x, batch_size))
  return np.concatenate(out)


def text_embeds(enc, tokenizer, texts, batch_size):
  ids = np.concatenate([np.array(tokenizer(t)) for t in texts])
  return np.concatenate([
      run_padded(enc, ids[i:i + batch_size], batch_size)
      for i in tqdm(range(0, len(ids), batch_size), desc="Text embeds")])


def target_ranks(scores, targets):
  """0-based rank of each query's target, from scores [Q, N]."""
  t = scores[np.arange(len(targets)), targets]
  return (scores > t[:, None]).sum(1)


def recalls(ranks):
  r = {f"R@{k}": float(np.mean(ranks < k) * 100) for k in KS}
  r["R_avg"] = (r["R@10"] + r["R@50"]) / 2  # usual FashionIQ summary
  return r


class Reranker:
  """Re-scores each query's top-k by z(ML) + sum_i w_i * z(feature_i).

  z() standardizes a score over the query's k candidates, so the weights
  compare scores of different scales. Targets outside the top-k keep their
  first-stage rank.
  """

  def __init__(self, scores, targets, feats, k):
    q = np.arange(len(scores))[:, None]
    self.top = np.argsort(-scores, axis=1)[:, :k]  # [Q, k], first-stage order
    hit = self.top == targets[:, None]
    self.pos = np.where(hit.any(1), hit.argmax(1), -1)  # target's slot
    self.base = target_ranks(scores, targets)
    z = lambda s: (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6)
    self.ml = z(scores[q, self.top])
    self.feats = [z(f[q, self.top]) for f in feats]

  def scores(self, weights):
    return self.ml + sum(w * f for w, f in zip(weights, self.feats))

  def order(self, weights):
    """Reranked top-k image indices [Q, k]."""
    return np.take_along_axis(self.top, np.argsort(-self.scores(weights), axis=1), 1)

  def ranks(self, weights):
    s = self.scores(weights)
    inside = self.pos >= 0
    t = s[inside, self.pos[inside]]
    r = self.base.copy()
    r[inside] = (s[inside] > t[:, None]).sum(1)
    return r


def paired_bootstrap(hits_a, hits_b, n=2000, seed=0):
  """Mean and 95% CI (in points) of hits_b - hits_a over queries."""
  d = (hits_b.astype(float) - hits_a.astype(float)) * 100
  idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
  m = d[idx].mean(1)
  return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def md_table(rows):
  """rows: [(name, {"dev": recalls, "test": recalls})] -> markdown table."""
  lines = ["| Cấu hình | dev R_avg | " + " | ".join(f"test {c}" for c in COLS) + " |",
           "|---|---|" + "---|" * len(COLS)]
  for name, r in rows:
    lines.append(f"| {name} | {r['dev']['R_avg']:.1f} | "
                 + " | ".join(f"{r['test'][c]:.1f}" for c in COLS) + " |")
  return "\n".join(lines)


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--model_size", default="base", choices=["base", "large"])
  parser.add_argument("--model_path", default="./models/magic_lens_clip_base.pkl")
  parser.add_argument("--subtask", default="dress", choices=list(CATEGORY))
  parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES,
                      help="Preprocessing modes to compare (original = baseline).")
  parser.add_argument("--data_dir", default="./data/fiq")
  parser.add_argument("--cache_dir", default="./cache")
  parser.add_argument("--output", default="./output")
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--rerank_k", type=int, default=100)
  parser.add_argument("--seed", type=int, default=0, help="Seed of the dev/test split.")
  args = parser.parse_args()
  if "original" not in args.modes:
    parser.error("--modes must include 'original' (the baseline).")
  modes = [m for m in MODES if m in args.modes]

  ids, queries = load_fiq(args.data_dir, args.subtask)
  index_of = {iid: n for n, iid in enumerate(ids)}
  refs = np.array([index_of[q["candidate"]] for q in queries])
  targets = np.array([index_of[q["target"]] for q in queries])
  perm = np.random.default_rng(args.seed).permutation(len(queries))
  dev, test = np.sort(perm[:len(perm) // 2]), np.sort(perm[len(perm) // 2:])

  cache = os.path.join(args.cache_dir, f"fiq-{args.subtask}-{args.model_size}")
  os.makedirs(cache, exist_ok=True)
  meta = {"ids": ids, "queries": [[q["candidate"], q["target"]] for q in queries]}
  meta_path = os.path.join(cache, "meta.json")
  if os.path.exists(meta_path):
    with open(meta_path) as f:
      if json.load(f) != meta:
        raise SystemExit(f"{cache} was built for other images/queries; delete it and re-run.")
  else:
    with open(meta_path, "w") as f:
      json.dump(meta, f)
  timing_path = os.path.join(cache, "timing.json")
  timing = {}
  if os.path.exists(timing_path):
    with open(timing_path) as f:
      timing = json.load(f)

  tokenizer = clip_tokenizer.build_tokenizer(bpe_path=default_bpe())
  model, params = load_model(args.model_size, args.model_path)
  enc_img = jax.jit(lambda p, x: model.apply(p, x, method=MagicLens.encode_image))
  enc_txt = jax.jit(lambda p, x: model.apply(p, x, method=MagicLens.encode_text))

  def cached(name, compute, n_items):
    """Loads cache/<name>.npy, computing it (and its time per item) if missing."""
    path = os.path.join(cache, name + ".npy")
    if not os.path.exists(path):
      t0 = time.time()
      np.save(path, compute())
      timing[name] = (time.time() - t0) / n_items
      with open(timing_path, "w") as f:
        json.dump(timing, f, indent=1)
    return np.load(path)

  # CLIP embeddings: images once per preprocessing mode, texts once per phrasing.
  img = {m: cached(f"img_{m}", lambda m=m: image_embeds(
      lambda x: enc_img(params, x), ids, f"{args.data_dir}/images", m,
      args.batch_size), len(ids)) for m in modes}
  cat = CATEGORY[args.subtask]
  txt = {n: cached(f"txt_{n}", lambda t=t: text_embeds(
      lambda x: enc_txt(params, x), tokenizer,
      [t.format(c1=q["captions"][0], c2=q["captions"][1], cat=cat) for q in queries],
      64), len(queries)) for n, t in TEMPLATES.items()}
  empty = cached("txt_empty", lambda: text_embeds(
      lambda x: enc_txt(params, x), tokenizer, [""], 64), 1)

  # Fusion with the multimodal encoder is cheap, so it is re-run every time.
  fuse = lambda i, t: np.asarray(model.apply(
      params, i, t, method=MagicLens.fuse)["multimodal_embed_norm"])
  index_embeds = {m: fuse(img[m], np.repeat(empty, len(ids), axis=0)) for m in modes}
  query_embeds = {}

  def ml_scores(mode, names):
    """MagicLens scores [Q, N], query embeds averaged over the phrasings."""
    for n in names:
      if (mode, n) not in query_embeds:
        query_embeds[(mode, n)] = fuse(img[mode][refs], txt[n])
    query = l2_normalize(sum(query_embeds[(mode, n)] for n in names))
    return query @ index_embeds[mode].T

  def evaluate(ranks):
    return {"dev": recalls(ranks[dev]), "test": recalls(ranks[test]), "all": recalls(ranks)}

  dev_score = lambda ranks: recalls(ranks[dev])["R_avg"]

  # Step 1: preprocessing mode.
  step1 = {m: target_ranks(ml_scores(m, ["and"]), targets) for m in modes}
  base_ranks = step1["original"]
  best_mode = max(modes, key=lambda m: dev_score(step1[m]))

  # Step 2: phrasing, single or ensembled.
  options = {n: [n] for n in TEMPLATES}
  options["ensemble"] = list(TEMPLATES)
  step2 = {o: target_ranks(ml_scores(best_mode, names), targets)
           for o, names in options.items()}
  best_opt = max(options, key=lambda o: dev_score(step2[o]))

  # Step 3: rerank the top-k with text->image and image->image CLIP scores.
  first_stage = step2[best_opt]
  scores = ml_scores(best_mode, options[best_opt])
  clip_img = l2_normalize(img[best_mode])
  clip_txt = l2_normalize(sum(l2_normalize(txt[n]) for n in options[best_opt]))
  t0 = time.time()  # upper bound: also scores all N images, not just the top-k
  rr = Reranker(scores, targets, [clip_txt @ clip_img.T, clip_img[refs] @ clip_img.T],
                args.rerank_k)
  rr.ranks((1.0, 1.0))
  rerank_time = (time.time() - t0) / len(queries)
  grid = {(lam, mu): rr.ranks((lam, mu)) for lam in WEIGHTS for mu in WEIGHTS}
  lam_only = max(WEIGHTS, key=lambda w: dev_score(grid[(w, 0.0)]))
  mu_only = max(WEIGHTS, key=lambda w: dev_score(grid[(0.0, w)]))
  best_w = max(grid, key=lambda w: dev_score(grid[w]))
  step3 = {f"Chỉ văn bản → ảnh (λ={lam_only}, μ=0)": grid[(lam_only, 0.0)],
           f"Chỉ ảnh → ảnh (λ=0, μ={mu_only})": grid[(0.0, mu_only)],
           f"Cả hai (λ={best_w[0]}, μ={best_w[1]})": grid[best_w]}
  final_ranks = grid[best_w]
  base_top10 = np.argsort(-ml_scores("original", ["and"]), axis=1)[:, :10]
  final_top10 = rr.order(best_w)[:, :10]

  # Report.
  summary = [("Baseline (original, \"and\")", base_ranks),
             (f"+ Tiền xử lý: {best_mode}", step1[best_mode]),
             (f"+ Truy vấn: {best_opt}", step2[best_opt]),
             (f"+ Rerank top-{args.rerank_k} (λ={best_w[0]}, μ={best_w[1]})", final_ranks)]
  hits = lambda r: {"R@10": r < 10, "R_avg": ((r < 10).astype(float) + (r < 50)) / 2}
  boot = {c: paired_bootstrap(hits(base_ranks[test])[c], hits(final_ranks[test])[c])
          for c in ("R@10", "R_avg")}
  base_all = recalls(base_ranks)
  paper = PAPER.get((args.model_size, args.subtask))
  t_img = timing.get("img_original")
  t_txt = timing.get("txt_and")

  md = [f"# MagicLens-{args.model_size} trên FashionIQ {args.subtask}: cải tiến không cần huấn luyện",
        "",
        f"Kho ảnh: {len(ids)} ảnh. Truy vấn: {len(queries)} (dev {len(dev)} / test {len(test)}, "
        f"chia ngẫu nhiên với seed {args.seed}). Mọi lựa chọn dựa trên R_avg = (R@10 + R@50)/2 "
        "của dev; các cột test là kết quả báo cáo.",
        "", "## Tóm tắt (cộng dồn từng cải tiến)", "",
        md_table([(n, evaluate(r)) for n, r in summary]), "",
        f"Chênh lệch so với baseline trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%): "
        f"R@10 {boot['R@10'][0]:+.1f} [{boot['R@10'][1]:+.1f}, {boot['R@10'][2]:+.1f}], "
        f"R_avg {boot['R_avg'][0]:+.1f} [{boot['R_avg'][1]:+.1f}, {boot['R_avg'][2]:+.1f}] điểm.",
        "", "## Baseline trên toàn bộ truy vấn", "",
        f"R@10 {base_all['R@10']:.1f}, R@50 {base_all['R@50']:.1f} trên {len(queries)} truy vấn."
        + (f" Paper: R@10 {paper[0]}, R@50 {paper[1]} trên tập đầy đủ (không thiếu ảnh)."
           if paper else ""),
        "", "## 1. Tiền xử lý ảnh (truy vấn \"and\")", "",
        md_table([(m, evaluate(step1[m])) for m in modes]),
        "", f"## 2. Cách viết truy vấn (tiền xử lý {best_mode})", "",
        md_table([(o + (" (" + ", ".join(options[o]) + ")" if o == "ensemble" else
                        f": \"{TEMPLATES[o]}\""), evaluate(step2[o])) for o in options]),
        "", f"## 3. Rerank top-{args.rerank_k}", "",
        md_table([(f"Không rerank ({best_mode}, {best_opt})", evaluate(first_stage))]
                 + [(n, evaluate(r)) for n, r in step3.items()]),
        "",
        f"Trần của rerank: {np.mean(first_stage[test] < args.rerank_k) * 100:.1f}% truy vấn test có ảnh "
        f"đích trong top-{args.rerank_k} của bước 1.",
        "", "## Thời gian (CPU)", ""]
  if t_img and t_txt:
    md.append(f"Mã hóa ảnh {t_img:.2f} s/ảnh, mã hóa văn bản {t_txt:.3f} s/câu, "
              f"rerank ≤ {rerank_time * 1000:.2f} ms/truy vấn.")
  md = "\n".join(md) + "\n"

  os.makedirs(args.output, exist_ok=True)
  stem = os.path.join(args.output, f"fiq-{args.subtask}-{args.model_size}_experiments")
  with open(stem + ".md", "w") as f:
    f.write(md)
  with open(stem + ".json", "w") as f:
    json.dump({"summary": {n: evaluate(r) for n, r in summary},
               "preprocessing": {m: evaluate(step1[m]) for m in modes},
               "phrasing": {o: evaluate(step2[o]) for o in options},
               "rerank": {n: evaluate(r) for n, r in step3.items()},
               "bootstrap_test_vs_baseline": boot,
               "choices": {"mode": best_mode, "phrasing": best_opt,
                           "lambda": float(best_w[0]), "mu": float(best_w[1])},
               "timing_s": dict(timing, rerank_per_query=rerank_time),
               # Top-10 lists of the test queries, for qualitative figures.
               "test_examples": [
                   {"candidate": queries[i]["candidate"], "target": queries[i]["target"],
                    "captions": queries[i]["captions"],
                    "baseline_rank": int(base_ranks[i]), "final_rank": int(final_ranks[i]),
                    "baseline_top10": [ids[j] for j in base_top10[i]],
                    "final_top10": [ids[j] for j in final_top10[i]]} for i in test]},
              f, ensure_ascii=False, indent=1)
  print(md)
  print("Results written to", stem + ".md")

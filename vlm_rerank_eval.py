# pylint: disable=missing-function-docstring
"""Evaluates the VLM rerank of MagicLens' top-K on FashionIQ.

Re-scores each query's top-k (k <= K scored by vlm_rerank.py) by
z(MagicLens) + w * z(VLM), where z() standardizes over the query's k
candidates as in fiq_experiments.Reranker, or by the VLM alone. k and w are
chosen on the dev half and reported on the test half, with a paired bootstrap
against the zero-shot baseline. Targets outside the top-k keep their rank.
Also reported: the VLM alone, the gain once each query's own image is filtered
out of both rankings, how well each score separates the target from the other
candidates, and R@10 over the whole (k, w) grid.

Usage (from the magiclens/ folder, after copying the Colab output to
vlm_rerank/):
  python vlm_rerank_eval.py --scores vlm_rerank/vlm_scores.jsonl
"""
from argparse import ArgumentParser
import json
import os

import fiq_experiments as fx
import numpy as np

# Rerank depths tried, capped by the scored top-K (which is tried too). Not 10:
# reordering inside the top-10 cannot change R@10 or R@50, so the dev R_avg
# could not judge it.
KS = (20, 30, 50)
WEIGHTS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0)
VLM_ONLY = None  # weight value meaning "rank by the VLM score alone"


def z(s):
  return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6)


def fused(ml, vlm, k, w):
  """New scores of each query's top-k."""
  if w is VLM_ONLY:
    return z(vlm[:, :k]) + 1e-3 * z(ml[:, :k])  # MagicLens only breaks ties
  return z(ml[:, :k]) + w * z(vlm[:, :k])


def rerank_ranks(ml, vlm, pos, base, k, w):
  """0-based target ranks after re-scoring the top-k of each query."""
  s = fused(ml, vlm, k, w)
  inside = (pos >= 0) & (pos < k)
  r = base.copy()
  t = s[inside, pos[inside]]
  r[inside] = (s[inside] > t[:, None]).sum(1)
  return r


def drop_query_image(r, s, pos, ref):
  """Ranks r once each query's own image (slot ref of s, or -1) is filtered out.

  s holds the scores that ordered the first s.shape[1] slots. Ranks below that
  depth are exact, which covers R@10 for any depth >= 20.
  """
  k = s.shape[1]
  i = np.flatnonzero((pos >= 0) & (pos < k) & (ref >= 0) & (ref < k))
  r = r.copy()
  r[i] -= s[i, ref[i]] > s[i, pos[i]]
  return r


def auc(s, pos, sel):
  """Mean over the sel queries of the target-vs-others AUC in s (ties count half).

  Only queries whose target is among the columns of s count; None if none does.
  """
  sel = sel & (pos >= 0) & (pos < s.shape[1])
  if not sel.any():
    return None
  t = s[sel, pos[sel]][:, None]
  wins = (s[sel] < t).sum(1) + 0.5 * ((s[sel] == t).sum(1) - 1)
  return float(np.mean(wins / (s.shape[1] - 1)))


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--candidates", default="./vlm_rerank/candidates.json")
  parser.add_argument("--scores", default="./vlm_rerank/vlm_scores.jsonl")
  parser.add_argument("--output", default="./output")
  parser.add_argument("--allow_partial", action="store_true",
                      help="Evaluate only the queries scored so far (results are then "
                           "not comparable with the other tables).")
  args = parser.parse_args()

  with open(args.candidates) as f:
    data = json.load(f)
  if not os.path.exists(args.scores + ".meta.json"):
    raise SystemExit(f"Missing {args.scores}.meta.json: download it from Drive together with "
                     f"{os.path.basename(args.scores)}.")
  with open(args.scores + ".meta.json") as f:
    meta = json.load(f)
  expected = {c: data[c] for c in ("subtask", "model_size", "seed", "top_k", "n_index")}
  expected["n_queries"] = len(data["queries"])
  if meta["candidates"] != expected:
    raise SystemExit(f"{args.scores} was scored on another candidates.json.")
  rec = {}
  with open(args.scores) as f:
    lines = [l for l in f if l.strip()]
  for n, line in enumerate(lines):
    try:
      r = json.loads(line)
      rec[r["q"]] = r
    except (json.JSONDecodeError, KeyError):
      if n < len(lines) - 1:
        raise SystemExit(f"Line {n + 1} of {args.scores} is corrupt.")
      print("Ignoring a truncated last line (the Colab run was cut while writing it).")
  queries = [q for q in data["queries"] if q["q"] in rec]
  if len(queries) < len(data["queries"]) and not args.allow_partial:
    raise SystemExit(f"Only {len(queries)}/{len(data['queries'])} queries are scored; finish "
                     "the Colab run (it resumes) or pass --allow_partial.")
  K = meta["top_k"]
  if K < min(KS):
    raise SystemExit(f"top_k {K} < {min(KS)}: reordering such a short list cannot change "
                     "R@10 or R@50. Score at least the top-20 on Colab.")
  if any(len(rec[q["q"]]["scores"]) != K for q in queries):
    raise SystemExit("Some queries do not have top_k scores.")

  ml = np.array([q["ml_scores"][:K] for q in queries])
  vlm = np.array([rec[q["q"]]["scores"] for q in queries])
  base = np.array([q["base_rank"] for q in queries])
  pos = np.array([q["top"].index(q["target"]) if q["target"] in q["top"][:K] else -1
                  for q in queries])
  # The index also holds each query's own image, as in the original evaluation.
  ref = np.array([q["top"].index(q["candidate"]) if q["candidate"] in q["top"][:K] else -1
                  for q in queries])
  split = np.array([q["split"] for q in queries])
  dev, test = split == "dev", split == "test"
  if not dev.any() or not test.any():
    raise SystemExit(f"Need scored queries in both halves (dev {dev.sum()}, test {test.sum()}).")
  ks = sorted({k for k in KS if k <= K} | {K})  # the scored depth itself is also tried
  dev_score = lambda r: fx.recalls(r[dev])["R_avg"]
  ev = lambda r: {"dev": fx.recalls(r[dev]), "test": fx.recalls(r[test])}

  # Choices on dev: the best weight for each depth, then the best depth.
  grid = {(k, w): rerank_ranks(ml, vlm, pos, base, k, w)
          for k in ks for w in WEIGHTS + (VLM_ONLY,)}
  best_w = {k: max(WEIGHTS, key=lambda w, k=k: dev_score(grid[(k, w)])) for k in ks}
  best_k = max(ks, key=lambda k: dev_score(grid[(k, best_w[k])]))
  only_k = max(ks, key=lambda k: dev_score(grid[(k, VLM_ONLY)]))
  final = grid[(best_k, best_w[best_k])]
  hits = lambda r: {"R@10": r < 10, "R_avg": ((r < 10).astype(float) + (r < 50)) / 2}
  boot = {c: fx.paired_bootstrap(hits(base[test])[c], hits(final[test])[c])
          for c in ("R@10", "R_avg")}
  vlm_s = np.array([rec[q["q"]]["sec"] for q in queries]).sum()
  name_w = lambda w: "chỉ VLM" if w is VLM_ONLY else f"w={w:g}"

  # Further test comparisons on R@10: a rerank within the top-50 cannot move
  # R@50, so R_avg adds nothing beyond R@10 there.
  r10 = lambda a, b: fx.paired_bootstrap(a[test] < 10, b[test] < 10)
  only = grid[(only_k, VLM_ONLY)]
  s_final = fused(ml, vlm, best_k, best_w[best_k])
  base_f, final_f = drop_query_image(base, ml, pos, ref), drop_query_image(final, s_final, pos, ref)
  extra = {"vlm_only_vs_baseline": r10(base, only), "fused_vs_vlm_only": r10(only, final),
           "fused_vs_baseline_without_query_image": r10(base_f, final_f)}
  moved = {"into_top10": int(((base >= 10) & (final < 10))[test].sum()),
           "out_of_top10": int(((base < 10) & (final >= 10))[test].sum())}
  n_auc = int((test & (pos >= 0) & (pos < best_k)).sum())
  aucs = {"MagicLens": auc(ml[:, :best_k], pos, test), "VLM": auc(vlm[:, :best_k], pos, test),
          "MagicLens + VLM": auc(s_final, pos, test)}
  if n_auc:
    auc_line = (f"- Khả năng tách ảnh đích khỏi {best_k - 1} ứng viên còn lại, trên {n_auc} truy vấn "
                f"có ảnh đích trong top-{best_k} (AUC trung bình theo truy vấn; 0.5 là ngẫu nhiên): "
                + ", ".join(f"{n} {v:.3f}" for n, v in aucs.items()) + ".")
  else:
    auc_line = f"- Chưa có truy vấn test nào có ảnh đích trong top-{best_k}, nên chưa tính được AUC."
  corr = float(np.corrcoef(z(ml[test, :best_k]).ravel(), z(vlm[test, :best_k]).ravel())[0, 1])
  ref_new = rerank_ranks(ml, vlm, ref, np.where(ref >= 0, ref, K), best_k, best_w[best_k])
  ref_top10 = {"before": int(((ref >= 0) & (ref < 10))[test].sum()),
               "after": int(((ref >= 0) & (ref_new < 10))[test].sum())}
  ci = lambda b: f"{b[0]:+.1f} [{b[1]:+.1f}, {b[2]:+.1f}]"
  pct10 = lambda r: float(np.mean(r[test] < 10) * 100)
  r10_dt = lambda r: f"{np.mean(r[dev] < 10) * 100:.1f} / {pct10(r):.1f}"
  key_w = lambda w: "vlm_only" if w is VLM_ONLY else f"{w:g}"

  md = [f"# Rerank top-K của MagicLens-{data['model_size']} bằng VLM trên FashionIQ "
        f"{data['subtask']}",
        "",
        f"VLM: `{meta['model']}` ({meta['dtype']}), chấm {K} ứng viên đầu của mỗi truy vấn bằng "
        "log-odds logit(\"Yes\") − logit(\"No\"). Điểm mới = z(MagicLens) + w · z(VLM) trên k ứng "
        f"viên đầu. Truy vấn: dev {dev.sum()} / test {test.sum()} (cùng cách chia với "
        "fiq_experiments.py). Chọn k và w theo R_avg của dev. Không mô hình nào được huấn luyện "
        "trên FashionIQ; nhãn của nửa dev chỉ dùng để chọn k và w.",
        "", fx.md_table([("MagicLens zero-shot (baseline)", ev(base)),
                         (f"Chỉ VLM, rerank top-{only_k}", ev(grid[(only_k, VLM_ONLY)])),
                         (f"MagicLens + VLM, top-{best_k} ({name_w(best_w[best_k])})", ev(final))]),
        "",
        f"Chênh lệch trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%): "
        f"R@10 {boot['R@10'][0]:+.1f} [{boot['R@10'][1]:+.1f}, {boot['R@10'][2]:+.1f}], "
        f"R_avg {boot['R_avg'][0]:+.1f} [{boot['R_avg'][1]:+.1f}, {boot['R_avg'][2]:+.1f}] điểm. "
        + ("Rerank trong top-k (k ≤ 50) không đổi tập top-50, nên R@50 giữ nguyên, R_avg chỉ "
           "đổi theo R@10 và khoảng tin cậy của R_avg đúng bằng một nửa của R@10. "
           if best_k <= 50 else f"Rerank top-{best_k} có thể đổi cả R@50. ")
        + f"Trên test, ảnh đích được đưa vào top-10 ở {moved['into_top10']} truy vấn và bị đẩy "
        f"ra ở {moved['out_of_top10']} truy vấn.",
        "", "## Theo độ sâu rerank k (w tốt nhất trên dev cho từng k)", "",
        fx.md_table([(f"top-{k} ({name_w(best_w[k])}); trần R@10 test "
                      f"{np.mean(base[test] < k) * 100:.1f}", ev(grid[(k, best_w[k])]))
                     for k in ks]
                    + [(f"top-{k}, chỉ VLM", ev(grid[(k, VLM_ONLY)])) for k in ks]),
        "", "## So sánh thêm trên test (R@10, bootstrap như trên)", "",
        f"- Chỉ VLM (top-{only_k}) so với baseline: {ci(extra['vlm_only_vs_baseline'])}.",
        f"- MagicLens + VLM so với chỉ VLM: {ci(extra['fused_vs_vlm_only'])}.",
        "- Kho ảnh chứa cả ảnh truy vấn (giống cách đánh giá gốc). Ảnh này nằm trong top-10 của "
        f"{ref_top10['before']} truy vấn test trước rerank, còn {ref_top10['after']} sau rerank. "
        f"Nếu lọc ảnh truy vấn khỏi danh sách kết quả của cả hai hệ: R@10 {pct10(base_f):.2f} → "
        f"{pct10(final_f):.2f}, chênh lệch {ci(extra['fused_vs_baseline_without_query_image'])}.",
        "", "## Phân tích trên test", "",
        auc_line,
        f"- Tương quan Pearson giữa z(MagicLens) và z(VLM) trên mọi cặp (truy vấn, ứng viên) trong "
        f"top-{best_k} của {test.sum()} truy vấn test: {corr:.2f}.",
        "", "## Toàn bộ lưới: R@10 dev / test (chỉ để tham khảo; k và w chỉ được chọn trên dev)",
        "", "| w | " + " | ".join(f"top-{k}" for k in ks) + " |", "|---" * (len(ks) + 1) + "|",
        "| baseline | " + " | ".join(r10_dt(base) for _ in ks) + " |"]
  md += [f"| {name_w(w)} | " + " | ".join(r10_dt(grid[(k, w)]) for k in ks) + " |"
         for w in WEIGHTS + (VLM_ONLY,)]
  md += ["", "## Chi phí", "",
         f"Thời gian chấm VLM: {vlm_s / 3600:.1f} giờ GPU cho {len(queries) * K} cặp "
         f"({vlm_s / (len(queries) * K) * 1000:.0f} ms/cặp, gồm cả đọc ảnh, không gồm tải và nạp "
         "mô hình)."]
  if len(queries) < len(data["queries"]):
    md.insert(2, f"**Chưa đủ: chỉ {len(queries)}/{len(data['queries'])} truy vấn đã được chấm.**\n")
  md = "\n".join(md) + "\n"

  os.makedirs(args.output, exist_ok=True)
  stem = os.path.join(args.output, f"fiq-{data['subtask']}-{data['model_size']}_vlm_rerank")
  with open(stem + ".md", "w") as f:
    f.write(md)
  with open(stem + ".json", "w") as f:
    json.dump({"vlm": meta, "baseline": ev(base),
               "vlm_only": {"k": only_k, **ev(grid[(only_k, VLM_ONLY)])},
               "final": {"k": best_k, "w": best_w[best_k], **ev(final)},
               "per_k": {str(k): {"w": best_w[k], "fused": ev(grid[(k, best_w[k])]),
                                  "vlm_only": ev(grid[(k, VLM_ONLY)])} for k in ks},
               "bootstrap_test_vs_baseline": boot, "moved_test": moved,
               "bootstrap_test_r10": extra,
               "without_query_image_test_r10": {"baseline": pct10(base_f), "final": pct10(final_f)},
               "query_image_in_top10_test": ref_top10, "auc_test": aucs, "corr_ml_vlm_test": corr,
               "grid_r10": {str(k): {key_w(w): {"dev": float(np.mean(grid[(k, w)][dev] < 10) * 100),
                                                "test": pct10(grid[(k, w)])}
                                     for w in WEIGHTS + (VLM_ONLY,)} for k in ks},
               "vlm_gpu_s": float(vlm_s),
               "n_queries": len(queries), "complete": len(queries) == len(data["queries"])},
              f, ensure_ascii=False, indent=1)
  print(md)
  print("Results written to", stem + ".md")

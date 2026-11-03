# pylint: disable=missing-function-docstring
"""Scores MagicLens' top-K FashionIQ candidates with a vision-language model.

For each query and each of its top-K candidates (candidates.json from
vlm_rerank_prepare.py), the VLM sees the reference image, the candidate image
and the modification text, and is asked whether the candidate is the reference
after the change. The score is the log-odds logit("Yes") - logit("No") of the
first answer token, computed in float32 from the last hidden state;
vlm_rerank_eval.py fuses it with the MagicLens score.

Each image's vision features are computed once and reused: the ~3.6K images
appear in ~93K pairs, and the vision tower is most of the cost of a pair.
--no_vision_cache falls back to the plain (slower) forward pass. The CPU
preprocessing of the next batch runs while the GPU scores the current one.

Scores are appended to a JSONL file, one line per query, so an interrupted run
(e.g. a Colab disconnect) resumes where it stopped. A <out>.meta.json file
records the settings; a run with other settings refuses to append to it.

Uses PyTorch + transformers only (not the JAX code of this repo), and is meant
for a GPU runtime such as Google Colab (see vlm_rerank_colab.ipynb).

Usage:
  python vlm_rerank.py --bundle_dir /content/bundle \
    --out /content/drive/MyDrive/magiclens_vlm/vlm_scores.jsonl
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import math
import os
import time

from PIL import Image
import torch
from tqdm import tqdm
import transformers
from transformers import AutoProcessor
try:
  from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:  # transformers < 4.46
  from transformers import AutoModelForVision2Seq as AutoVLM

PROMPT = (
    "The first image shows a reference {cat}. The second image shows a candidate {cat}.\n"
    "Requested change to the reference {cat}: \"{text}\".\n"
    "Does the candidate {cat} look like the reference {cat} after the requested change? "
    "Answer Yes or No.")


def pick_dtype(name):
  if name != "auto":
    return getattr(torch, name)
  if not torch.cuda.is_available():
    return torch.float32
  # Turing GPUs such as the Colab T4 (compute capability 7.5) lack fast bfloat16.
  return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


class Scorer:
  """Yes/No log-odds of a Qwen2-VL-style model for (reference, candidate, prompt).

  Images are given by key; load(key) returns the PIL image. prepare() is the
  CPU part and can run in another thread; score() is the GPU part.
  """

  def __init__(self, model_id, dtype, max_pixels, device, load, vision_cache=True):
    self.device, self.load = device, load
    self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
    self.processor.tokenizer.padding_side = "left"  # last position = answer token
    self.model = AutoVLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    tok = self.processor.tokenizer
    yes, no = (tok.encode(w, add_special_tokens=False) for w in ("Yes", "No"))
    if len(yes) != 1 or len(no) != 1:
      raise SystemExit(f"'Yes'/'No' are not single tokens for {model_id}.")
    # Only the Yes - No logit at the last position is needed. The head is fed
    # just that position, and the difference is taken in float32 (a bfloat16
    # head would round the log-odds to steps of ~0.1).
    head = self.model.get_output_embeddings()
    self.w = (head.weight[yes[0]] - head.weight[no[0]]).float()
    self.b = 0.0 if head.bias is None else float(head.bias[yes[0]] - head.bias[no[0]])
    self.last = None

    def keep_last(_, args):
      self.last = args[0][:, -1].float()
      return (args[0][:, -1:],)

    head.register_forward_pre_hook(keep_last)
    # transformers >= 4.52 nests the vision tower and get_rope_index in .model.
    self.inner = self.model.model if hasattr(self.model.model, "visual") else self.model
    self.cache = {} if vision_cache else None  # key -> vision features on CPU

  def prepare(self, triples):
    """Chat template, images and processor, on CPU."""
    texts, keys = [], []
    for ref, cand, prompt in triples:
      messages = [{"role": "user", "content": [
          {"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]}]
      texts.append(self.processor.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=True))
      keys += [ref, cand]  # in the order of the image placeholders
    inputs = self.processor(text=texts, images=[self.load(k) for k in keys], padding=True,
                            return_tensors="pt")
    return inputs, keys

  def _inputs_embeds(self, inputs, keys):
    """Token embeddings with each image's cached vision features scattered in.

    Only the pixels of images not cached yet are moved to the GPU.
    """
    grid = inputs["image_grid_thw"]
    patches = grid.prod(-1).tolist()  # rows of pixel_values per image
    new = {}
    for i, key in enumerate(keys):
      if key not in self.cache and key not in new:
        new[key] = i
    if new:
      rows = torch.split(inputs["pixel_values"], patches)
      visual = self.inner.visual
      pixels = torch.cat([rows[i] for i in new.values()]).to(
          self.device, next(visual.parameters()).dtype)
      feats = visual(pixels, grid_thw=grid[list(new.values())].to(self.device))
      merge = self.model.config.vision_config.spatial_merge_size ** 2
      for key, f in zip(new, torch.split(feats, [patches[i] // merge for i in new.values()])):
        self.cache[key] = f.cpu()
    ids = inputs["input_ids"].to(self.device)
    mask = inputs["attention_mask"].to(self.device)
    embeds = self.model.get_input_embeddings()(ids)
    # Placeholders are filled in batch-row order, as the processor laid them out.
    embeds[ids == self.model.config.image_token_id] = torch.cat(
        [self.cache[k] for k in keys]).to(embeds.device, embeds.dtype)
    position_ids, _ = self.inner.get_rope_index(
        input_ids=ids, image_grid_thw=grid.to(self.device), attention_mask=mask)
    return {"inputs_embeds": embeds, "attention_mask": mask, "position_ids": position_ids}

  @torch.inference_mode()
  def score(self, prepared):
    inputs, keys = prepared
    if self.cache is not None:
      inputs = self._inputs_embeds(inputs, keys)
    else:
      inputs = inputs.to(self.device)
    self.model(**inputs)
    return (self.last @ self.w + self.b).tolist()


def load_done(path):
  """Query ids already in the JSONL output; drops a line cut by a crash."""
  done, lines = set(), []
  if not os.path.exists(path):
    return done
  with open(path) as f:
    for line in f:
      try:
        done.add(json.loads(line)["q"])
        lines.append(line)
      except (json.JSONDecodeError, KeyError):
        print("Dropping a truncated line at the end of", path)
        with open(path, "w") as g:
          g.writelines(lines)
        break
  return done


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--bundle_dir", default="./bundle",
                      help="Unzipped bundle.zip: candidates.json and images/.")
  parser.add_argument("--out", required=True, help="Output JSONL (put it on Google Drive).")
  parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
  parser.add_argument("--top_k", type=int, default=0,
                      help="Candidates scored per query (0 = all in candidates.json).")
  parser.add_argument("--batch_size", type=int, default=10, help="Pairs per forward pass.")
  parser.add_argument("--max_pixels", type=int, default=448 * 448,
                      help="Image size cap of the Qwen2-VL processor.")
  parser.add_argument("--dtype", default="auto",
                      choices=["auto", "float16", "bfloat16", "float32"])
  parser.add_argument("--limit", type=int, default=0,
                      help="Only the first N queries (for a quick test).")
  parser.add_argument("--allow_cpu", action="store_true",
                      help="Run without a GPU (very slow; for testing only).")
  parser.add_argument("--no_vision_cache", action="store_true",
                      help="Re-encode every image in every pair (slower fallback).")
  args = parser.parse_args()

  if not torch.cuda.is_available() and not args.allow_cpu:
    raise SystemExit("No GPU found. On Colab: Runtime > Change runtime type > T4 GPU.")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  dtype = pick_dtype(args.dtype)
  with open(os.path.join(args.bundle_dir, "candidates.json")) as f:
    data = json.load(f)
  k = args.top_k or data["top_k"]
  if k > data["top_k"]:
    raise SystemExit(f"--top_k {k} > {data['top_k']} candidates in candidates.json.")
  queries = data["queries"][:args.limit or None]

  meta = {"model": args.model, "prompt": PROMPT, "top_k": k, "max_pixels": args.max_pixels,
          "dtype": str(dtype).replace("torch.", ""), "vision_cache": not args.no_vision_cache,
          "transformers": transformers.__version__,
          "candidates": {c: data[c] for c in ("subtask", "model_size", "seed", "top_k", "n_index")}
                        | {"n_queries": len(data["queries"])}}
  meta_path = args.out + ".meta.json"
  if os.path.exists(meta_path):
    with open(meta_path) as f:
      old = json.load(f)
    if old != meta:
      changed = [c for c in meta if old.get(c) != meta[c]]
      raise SystemExit(f"{args.out} was made with other settings ({', '.join(changed)}); "
                       "use the same settings to resume, or another --out.")
  else:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(meta_path, "w") as f:
      json.dump(meta, f, indent=1)

  done = load_done(args.out)
  todo = [q for q in queries if q["q"] not in done]
  print(f"{len(done)} queries already scored, {len(todo)} to go "
        f"({len(todo) * k} pairs, top-{k}); model {args.model}, {meta['dtype']} on {device}"
        + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else "")
        + ("" if meta["vision_cache"] else ", no vision cache"))
  if not todo:
    raise SystemExit("Nothing to do.")
  image = lru_cache(maxsize=256)(lambda iid: Image.open(
      os.path.join(args.bundle_dir, "images", iid + ".png")).convert("RGB"))
  scorer = Scorer(args.model, dtype, args.max_pixels, device, image,
                  vision_cache=meta["vision_cache"])

  # One job per batch of a query's candidates; batches never span two queries.
  jobs = []
  for q in todo:
    prompt = PROMPT.format(cat=data["category"], text=q["text"])
    cands = q["top"][:k]
    jobs += [(q, [(q["candidate"], c, prompt) for c in cands[i:i + args.batch_size]])
             for i in range(0, k, args.batch_size)]
  with open(args.out, "a") as out, tqdm(total=len(todo) * k, unit="pair") as bar, \
       ThreadPoolExecutor(max_workers=1) as pool:
    t0, scores = time.time(), []
    pending = pool.submit(scorer.prepare, jobs[0][1])
    for j, (q, triples) in enumerate(jobs):
      prepared = pending.result()
      if j + 1 < len(jobs):  # preprocess the next batch while the GPU works
        pending = pool.submit(scorer.prepare, jobs[j + 1][1])
      scores += scorer.score(prepared)
      bar.update(len(triples))
      if len(scores) < k:
        continue
      if not all(math.isfinite(s) for s in scores):
        raise SystemExit(f"Non-finite scores for query {q['q']}: {meta['dtype']} overflows "
                         "for this model; re-run with --dtype float32 and a new --out.")
      out.write(json.dumps({"q": q["q"], "scores": [round(s, 4) for s in scores],
                            "sec": round(time.time() - t0, 2)}) + "\n")
      out.flush()
      t0, scores = time.time(), []
  print(f"Done: {len(load_done(args.out))}/{len(queries)} queries scored in {args.out}")

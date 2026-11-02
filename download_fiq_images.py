# pylint: disable=missing-module-docstring,missing-function-docstring
"""Downloads FashionIQ images from their original Amazon URLs.

The official FashionIQ repo points to hongwang600/fashion-iq-metadata for
image URLs. Images are saved as real PNG files to <data_dir>/images/<asin>.png,
which is the layout expected by data_utils.build_fiq_dataset.

Usage (from the magiclens/ folder):
  python download_fiq_images.py --subtasks dress
  python download_fiq_images.py --subtasks dress shirt toptee

Re-running is safe: images already on disk are skipped. ASINs whose links are
dead are written to <data_dir>/missing.<subtask>.<split>.txt.
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
import time
import urllib.error
import urllib.request

from PIL import Image
from tqdm import tqdm

URL_LIST = ("https://raw.githubusercontent.com/hongwang600/"
            "fashion-iq-metadata/master/image_url/asin2url.{}.txt")
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch(url: str, timeout: int = 20) -> bytes:
  req = urllib.request.Request(url, headers=HEADERS)
  with urllib.request.urlopen(req, timeout=timeout) as r:
    return r.read()


def load_url_map(subtask: str, data_dir: str) -> dict:
  cache = os.path.join(data_dir, "image_url", f"asin2url.{subtask}.txt")
  if not os.path.exists(cache):
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
      f.write(fetch(URL_LIST.format(subtask), timeout=60))
  url_map = {}
  with open(cache) as f:
    for line in f:
      parts = line.split()
      if len(parts) >= 2:
        url_map[parts[0]] = parts[1]
  return url_map


def download_one(asin: str, url: str, out_path: str, retries: int) -> bool:
  for attempt in range(retries):
    for candidate in (url, url.replace("http://", "https://", 1)):
      try:
        img = Image.open(io.BytesIO(fetch(candidate))).convert("RGB")
        tmp = out_path + ".part"
        img.save(tmp, format="PNG")
        os.replace(tmp, out_path)  # atomic: no half-written .png on Ctrl+C
        return True
      except urllib.error.HTTPError as e:
        if e.code == 404:
          return False  # dead link, retrying will not help
      except Exception:  # pylint: disable=broad-except
        pass
    time.sleep(1 + attempt)
  return False


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--subtasks", nargs="+", default=["dress"],
                      choices=["dress", "shirt", "toptee"])
  parser.add_argument("--split", default="val",
                      choices=["train", "val", "test"])
  parser.add_argument("--data_dir", default="./data/fiq")
  parser.add_argument("--workers", type=int, default=16)
  parser.add_argument("--retries", type=int, default=3)
  parser.add_argument("--limit", type=int, default=0,
                      help="Only download the first N images (for testing).")
  args = parser.parse_args()

  image_dir = os.path.join(args.data_dir, "images")
  os.makedirs(image_dir, exist_ok=True)

  for subtask in args.subtasks:
    split_file = os.path.join(args.data_dir, "image_splits",
                              f"split.{subtask}.{args.split}.json")
    with open(split_file) as f:
      asins = json.load(f)
    if args.limit:
      asins = asins[:args.limit]

    url_map = load_url_map(subtask, args.data_dir)
    todo = [a for a in asins
            if not os.path.exists(os.path.join(image_dir, a + ".png"))]
    no_url = [a for a in todo if a not in url_map]
    todo = [a for a in todo if a in url_map]
    print(f"[{subtask}] need {len(asins)} | already on disk "
          f"{len(asins) - len(todo) - len(no_url)} | to download {len(todo)}"
          f" | no URL {len(no_url)}")

    def job(asin):
      path = os.path.join(image_dir, asin + ".png")
      return asin, download_one(asin, url_map[asin], path, args.retries)

    failed = list(no_url)
    with ThreadPoolExecutor(args.workers) as ex:
      for asin, ok in tqdm(ex.map(job, todo), total=len(todo), desc=subtask):
        if not ok:
          failed.append(asin)

    missing_file = os.path.join(args.data_dir,
                                f"missing.{subtask}.{args.split}.txt")
    with open(missing_file, "w") as f:
      f.write("\n".join(sorted(failed)) + ("\n" if failed else ""))
    have = len(asins) - len(failed)
    print(f"[{subtask}] done: {have}/{len(asins)} images "
          f"({have / len(asins) * 100:.1f}%), {len(failed)} missing "
          f"-> {missing_file}")

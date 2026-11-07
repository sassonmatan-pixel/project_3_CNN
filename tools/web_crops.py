"""Download chosen Commons photos at higher res, detect fruits with Roboflow, crop single fruits.
Output goes to scratchpad/web_out/<split>/<Fruit>/*.jpg  (merged into Fruit_v3 in a separate, explicit step)."""
import csv, json, os, random, sys, time, urllib.request
from PIL import Image

SCR = sys.argv[1]
KEY = os.environ["ROBOFLOW_API_KEY"]
UA = "FruitVisionStudentProject/1.0 (student machine-learning project)"
CHOSEN = {
    "Apple": [1, 2, 3, 4, 5, 6, 10, 12, 14, 15, 16, 18, 20, 21, 23, 24, 25, 47, 64],
    "Avocado": [0, 1, 4, 6, 9, 10, 12, 14, 15, 16, 17, 20, 21, 22, 24, 25, 26, 27, 55, 56, 57, 58, 61, 62, 65, 66],
    "Banana": [0, 1, 4, 11, 12, 17, 18, 22, 23, 24, 29, 32, 35, 43, 47, 49, 59, 60, 61, 62, 64, 67],
    "Eggplant": [0, 1, 2, 3, 10, 12, 13, 17, 18, 22, 39, 40, 41, 43],
    "Tomato": [1, 2, 4, 5, 9, 13, 19, 20, 21, 22, 31, 32, 33, 35, 45, 47, 48, 49, 50, 51, 52, 53, 55, 57, 58, 59, 61, 62, 63, 65, 66, 68],
}
MIN_CONF, MIN_SIDE, MAX_CROPS, OUT_SIZE, PAD = 0.50, 60, 10, 224, 0.12
meta = json.load(open(os.path.join(SCR, "commons", "meta.json"), encoding="utf-8"))
hi = os.path.join(SCR, "hires"); os.makedirs(hi, exist_ok=True)
cache = os.path.join(SCR, "rf_cache"); os.makedirs(cache, exist_ok=True)

from inference_sdk import InferenceHTTPClient, InferenceConfiguration
client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=KEY).configure(
    InferenceConfiguration(api_key_transport="header"))


def fetch_hi(item, fruit):
    path = os.path.join(hi, item["file"])
    if os.path.exists(path):
        return path
    url = item["thumb"].replace("/500px-", "/960px-")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        open(path, "wb").write(urllib.request.urlopen(req, timeout=40).read())
        time.sleep(0.5)
    except Exception:
        return os.path.join(SCR, "commons", fruit, item["file"])  # fall back to the 500px version
    return path


def detect(path, fruit):
    cp = os.path.join(cache, os.path.basename(path) + ".json")
    if os.path.exists(cp):
        return json.load(open(cp))
    res = client.run_workflow(workspace_name="matan-sasson", workflow_id="general-segmentation-api",
                              images={"image": path}, parameters={"classes": "apple, banana, tomato, eggplant, avocado"},
                              use_cache=True)
    preds = [{k: p[k] for k in ("x", "y", "width", "height", "confidence", "class")}
             for p in res[0]["predictions"]["predictions"]]
    json.dump(preds, open(cp, "w"))
    return preds


def iou(a, b):
    ax0, ay0, ax1, ay1 = a["x"] - a["width"] / 2, a["y"] - a["height"] / 2, a["x"] + a["width"] / 2, a["y"] + a["height"] / 2
    bx0, by0, bx1, by1 = b["x"] - b["width"] / 2, b["y"] - b["height"] / 2, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / (a["width"] * a["height"] + b["width"] * b["height"] - inter + 1e-9)


rng = random.Random(7)
stats, manifest, sources = {}, [], []
for fruit, idxs in CHOSEN.items():
    items = {int(m["file"][-7:-4]): m for m in meta[fruit]}
    picked = [items[i] for i in idxs]
    rng.shuffle(picked)
    n_test = max(2, round(len(picked) * 0.25))
    for k, it in enumerate(picked):
        split = "test" if k < n_test else "train"
        path = fetch_hi(it, fruit)
        try:
            preds = detect(path, fruit)
        except Exception as e:
            print("  detect failed", it["file"], str(e)[:80]); continue
        img = Image.open(path).convert("RGB")
        W, H = img.size
        # Roboflow answers in the coordinates of the image we sent -> same as `img`
        good = [p for p in preds if p["class"].strip().lower() == fruit.lower() and p["confidence"] >= MIN_CONF
                and min(p["width"], p["height"]) >= MIN_SIDE * (W / 960 if W < 960 else 1)]
        good.sort(key=lambda p: -p["confidence"])
        kept = []
        for p in good:
            if all(iou(p, q) < 0.3 for q in kept):
                kept.append(p)
            if len(kept) >= MAX_CROPS:
                break
        n = 0
        for j, p in enumerate(kept):
            s = max(p["width"], p["height"]) * (1 + PAD * 2)
            x0, y0 = int(p["x"] - s / 2), int(p["y"] - s / 2)
            box = (max(0, x0), max(0, y0), min(W, int(x0 + s)), min(H, int(y0 + s)))
            c = img.crop(box)
            if min(c.size) < 40:
                continue
            side = max(c.size)
            sq = Image.new("RGB", (side, side), (128, 128, 128))  # pad to square (keeps the fruit un-stretched)
            sq.paste(c, ((side - c.size[0]) // 2, (side - c.size[1]) // 2))
            d = os.path.join(SCR, "web_out", split, fruit); os.makedirs(d, exist_ok=True)
            name = f"web_{it['file'][:-4]}_{j}.jpg"
            sq.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS).save(os.path.join(d, name), quality=92)
            manifest.append([split, fruit, name, "web_commons", it["file"]]); n += 1
        sources.append([it["file"], fruit, split, n, it["title"], it["page"], it["license"], it["author"]])
        s = stats.setdefault((fruit, split), [0, 0]); s[0] += 1; s[1] += n
        print(f"{fruit:8s} {it['file']} {split:5s} detections(all/{fruit})={len(preds)}/{len(good)} -> {n} crops", flush=True)

json.dump(manifest, open(os.path.join(SCR, "web_manifest.json"), "w"))
with open(os.path.join(SCR, "web_sources.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["file", "class", "split", "crops", "title", "page", "license", "author"]); w.writerows(sources)
print("\nSUMMARY (photos used, crops)")
for (fruit, split), (a, b) in sorted(stats.items()):
    print(f"  {fruit:9s} {split:5s} {a:3d} photos -> {b:4d} crops")

"""Build Fruit_v3: a diverse train/test dataset for Apple, Avocado, Banana, Eggplant, Tomato.

Sources
  A) Fruits-360 100x100  (lab photos, white background, many varieties)
  B) Fruits-360 "multi"  (real phone photos, natural backgrounds, many fruits per photo)

Anti-leakage rules
  * Fruits-360: when a fruit has >=3 varieties, whole VARIETIES are held out for the test set.
    Otherwise (Eggplant: 2 varieties) a contiguous block of frames is held out, with a gap.
  * Phone photos: split by SOURCE PHOTO first, then crop, so crops of one photo never
    land in both train and test.
  * Files are sampled evenly across sorted frame names (different angles), never consecutive frames.
"""
import csv, os, random, re, shutil, sys, zipfile
import numpy as np
from PIL import Image, ImageOps

SRC = r"C:\Users\Matan\Downloads\חדש"
OUT = r"C:\Users\Matan\Downloads\Fruit_v3"
FRUITS = ["Apple", "Avocado", "Banana", "Eggplant", "Tomato"]
TRAIN_PER_CLASS = 1500   # target of Fruits-360 images per class
TEST_PER_CLASS = 300
CROP_SIZE = 224
CROPS_PER_PHOTO = 4
TEST_PHOTO_FRACTION = 0.25
rng = random.Random(42)

if os.path.exists(OUT):
    sys.exit(f"{OUT} already exists - refusing to overwrite")

rows = []  # manifest


def cls_of(name):
    for f in FRUITS:
        if name == f or name.startswith(f + " "):
            return f


def frame_key(fn):
    m = re.match(r"^(r\d*_)?(\d+)_100\.jpg$", fn, re.I)
    return ((m.group(1) or ""), int(m.group(2))) if m else (fn, 0)


def evenly(items, k):
    if k >= len(items):
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def allocate(avail, total):
    """Split a quota evenly across varieties, giving leftovers to the ones that have more."""
    quota = {v: 0 for v in avail}
    left = total
    open_v = [v for v in avail if avail[v] > 0]
    while left > 0 and open_v:
        share = max(1, left // len(open_v))
        nxt = []
        for v in open_v:
            take = min(share, avail[v] - quota[v], left)
            quota[v] += take
            left -= take
            if quota[v] < avail[v] and left > 0:
                nxt.append(v)
        open_v = nxt
    return quota


def put(src_path, split, fruit, tag, origin, source_id):
    d = os.path.join(OUT, split, fruit)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, tag)
    shutil.copyfile(src_path, dst)
    rows.append([split, fruit, tag, origin, source_id])


# ---------------- A) Fruits-360 100x100 ----------------
base = os.path.join(SRC, "fruits-360_100x100", "fruits-360")
report = []
for fruit in FRUITS:
    # pool = Training + Test folders of the original dataset, per variety
    varieties = {}
    for sub in ("Training", "Test"):
        root = os.path.join(base, sub)
        for d in sorted(os.listdir(root)):
            if cls_of(d) == fruit:
                files = [os.path.join(root, d, f) for f in os.listdir(os.path.join(root, d)) if f.lower().endswith(".jpg")]
                varieties.setdefault(d, []).extend(files)
    names = sorted(varieties)
    for v in names:
        varieties[v].sort(key=lambda p: frame_key(os.path.basename(p)))

    if len(names) >= 3:
        n_hold = max(1, round(len(names) * 0.2))
        step = len(names) / n_hold
        held = {names[int(step * i + step / 2)] for i in range(n_hold)}
        train_v = {v: varieties[v] for v in names if v not in held}
        test_v = {v: varieties[v] for v in names if v in held}
        mode = "held-out varieties: " + ", ".join(sorted(held))
    else:  # too few varieties: hold out a contiguous block of frames, leaving a gap
        train_v, test_v = {}, {}
        for v in names:
            fs = varieties[v]
            n = len(fs)
            test_v[v] = fs[int(n * 0.80):]
            train_v[v] = fs[:int(n * 0.75)]
        mode = "block split (only %d varieties, test is optimistic)" % len(names)

    tq = allocate({v: len(f) for v, f in train_v.items()}, TRAIN_PER_CLASS)
    sq = allocate({v: len(f) for v, f in test_v.items()}, TEST_PER_CLASS)
    n_tr = n_te = 0
    for v, k in tq.items():
        for p in evenly(train_v[v], k):
            put(p, "train", fruit, f"f360_{v.replace(' ', '_')}_{os.path.basename(p)}", "fruits360", v)
            n_tr += 1
    for v, k in sq.items():
        for p in evenly(test_v[v], k):
            put(p, "test", fruit, f"f360_{v.replace(' ', '_')}_{os.path.basename(p)}", "fruits360", v)
            n_te += 1
    report.append(f"{fruit:9s} Fruits-360: train {n_tr:4d} from {len(train_v)} varieties | test {n_te:4d} from {len(test_v)} varieties | {mode}")

# ---------------- B) real phone photos ----------------
multi = os.path.join(SRC, "fruits-360_multi", "test-multiple_fruits")
pat = re.compile(r"^(apple|tomato|eggplant|banana|Banana\(lady_finger\))_\d+\.jpe?g$", re.I)
single = {f: [] for f in FRUITS}
for fn in sorted(os.listdir(multi)):
    m = pat.match(fn)
    if m:
        single[m.group(1).split("(")[0].capitalize()].append(fn)


def crops(img):
    w, h = img.size
    s = min(w, h)
    out = [img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))]  # whole short side
    for _ in range(CROPS_PER_PHOTO - 1):
        c = int(s * rng.uniform(0.55, 0.85))
        x, y = rng.randint(0, w - c), rng.randint(0, h - c)
        out.append(img.crop((x, y, x + c, y + c)))
    return [o.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS) for o in out]


os.makedirs(os.path.join(OUT, "_tmp"), exist_ok=True)
for fruit in FRUITS:
    files = single[fruit]
    if not files:
        report.append(f"{fruit:9s} phone photos: none available in the dataset")
        continue
    files = files[:]
    rng.shuffle(files)
    n_test = max(1, round(len(files) * TEST_PHOTO_FRACTION)) if len(files) >= 4 else 0
    test_src, train_src = files[:n_test], files[n_test:]
    cnt = {"train": 0, "test": 0}
    for split, group in (("train", train_src), ("test", test_src)):
        for fn in group:
            img = ImageOps.exif_transpose(Image.open(os.path.join(multi, fn))).convert("RGB")
            img.thumbnail((1200, 1200))
            for i, c in enumerate(crops(img)):
                tmp = os.path.join(OUT, "_tmp", "c.jpg")
                c.save(tmp, quality=92)
                put(tmp, split, fruit, f"real_{os.path.splitext(fn)[0]}_c{i}.jpg", "phone_photo", fn)
                cnt[split] += 1
    report.append(f"{fruit:9s} phone photos: {len(train_src)} photos -> {cnt['train']} train crops | {len(test_src)} photos -> {cnt['test']} test crops")
shutil.rmtree(os.path.join(OUT, "_tmp"))

with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["split", "class", "file", "origin", "source"])
    w.writerows(rows)

print("\n".join(report))
print("\nFINAL COUNTS")
for split in ("train", "test"):
    print(split, {f: len(os.listdir(os.path.join(OUT, split, f))) for f in FRUITS})

# zip for easy upload to Google Drive / Colab
zpath = OUT + ".zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
    for r, _, fs in os.walk(OUT):
        for fn in fs:
            full = os.path.join(r, fn)
            z.write(full, os.path.join("Fruit_v3", os.path.relpath(full, OUT)))
print("zip:", zpath, round(os.path.getsize(zpath) / 1e6, 1), "MB")

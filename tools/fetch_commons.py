"""Collect candidate photos from Wikimedia Commons (open licenses only) into the scratchpad."""
import json, os, re, sys, time, urllib.parse, urllib.request

OUT = sys.argv[1]
UA = "FruitVisionStudentProject/1.0 (student machine-learning project)"
API = "https://commons.wikimedia.org/w/api.php"
QUERIES = {
    "Apple": ["apple fruit", "apples on table", "green apple", "red apple", "apple kitchen"],
    "Avocado": ["avocado fruit", "avocado whole", "avocados on table", "hass avocado", "avocado market"],
    "Banana": ["banana fruit", "bananas on table", "banana bunch", "banana peel table", "banana market"],
    "Eggplant": ["eggplant", "aubergine vegetable", "eggplants market", "eggplant kitchen", "eggplant on table"],
    "Tomato": ["tomato", "tomatoes on table", "tomatoes market", "red tomato", "tomato kitchen"],
}
BAD = re.compile(r"tree|flower|blossom|leaf|leaves|juice|pie|salad|sauce|guacamole|paint|draw|illustr|logo|map|diagram|"
                 r"seed|toast|soup|curry|dish|cake|bread|plant|seedling|disease|pest|insect|cartoon|icon|stamp|"
                 r"cross.?section|sliced|halved|peeled|jam|ketchup|pizza|burger|sandwich|smoothie|cocktail|"
                 r"cultivar|flag|coat|museum|still.?life|poster|label|packag|crate", re.I)
OK_LICENSE = re.compile(r"^(cc0|cc[- ]by|cc[- ]by[- ]sa|pd|public domain)", re.I)


def get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def candidates(fruit):
    seen = {}
    for q in QUERIES[fruit]:
        data = get({"action": "query", "format": "json", "generator": "search", "gsrnamespace": 6, "gsrlimit": 40,
                    "gsrsearch": q + " filetype:bitmap", "prop": "imageinfo", "iiprop": "url|extmetadata|size|mime",
                    "iiurlwidth": 500})
        for p in (data.get("query", {}).get("pages", {})).values():
            ii = (p.get("imageinfo") or [None])[0]
            if not ii or ii.get("mime") != "image/jpeg" or min(ii["width"], ii["height"]) < 400:
                continue
            title = p["title"]
            meta = ii.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            if BAD.search(title) or not OK_LICENSE.match(lic):
                continue
            seen[p["pageid"]] = {"title": title, "thumb": ii["thumburl"], "page": ii["descriptionurl"], "license": lic,
                                 "author": re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", ""))[:80]}
        time.sleep(1.0)
    return list(seen.values())


meta_all = {}
for fruit in QUERIES:
    d = os.path.join(OUT, fruit)
    os.makedirs(d, exist_ok=True)
    cands = candidates(fruit)
    print(fruit, "candidates after filters:", len(cands), flush=True)
    meta_all[fruit] = []
    for i, c in enumerate(cands[:70]):
        fn = f"{fruit}_{i:03d}.jpg"
        try:
            req = urllib.request.Request(c["thumb"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                open(os.path.join(d, fn), "wb").write(r.read())
            meta_all[fruit].append({"file": fn, **c})
        except Exception as e:
            print("  skip", c["title"], e)
        time.sleep(0.6)
    print(fruit, "downloaded:", len(meta_all[fruit]), flush=True)
json.dump(meta_all, open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

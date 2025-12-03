
# Save this as scraper_no_chunking.py

import argparse
import hashlib
import json
import time
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
try:
    from readability import Document
except Exception:
    Document = None

try:
    import trafilatura
except Exception:
    trafilatura = None

def extract_text(html, url=None):
    if Document is not None:
        doc = Document(html)
        summary = doc.summary()
        soup = BeautifulSoup(summary, "html.parser")
        text = soup.get_text("\n", strip=True)
        title = doc.title() or ""
        return title, text
    if trafilatura is not None:
        text = trafilatura.extract(html, url=url) or ""
        title = ""  # keep it simple
        return title, text
    # fallback naive
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string if soup.title else ""
    for s in soup(["script", "style", "nav", "footer", "header", "aside"]):
        s.extract()
    text = soup.get_text("\n", strip=True)
    return title, text

def make_id(url):
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return h

def main(args):
    urls = []
    with open(args.urls) as f:
        urls = [line.strip() for line in f if line.strip()]
    with open(args.out, "w", encoding="utf-8") as out_f:
        for url in urls:
            try:
                r = requests.get(url, timeout=5, headers={"User-Agent":"sysagent-scraper/1.0"})
                r.raise_for_status()
                html = r.text
                title, text = extract_text(html, url=url)
                if not text or len(text.split()) < 5:
                    print(f"[WARN] extracted too small text for {url}")
                item = {
                    "id": make_id(url),
                    "source_url": url,
                    "title": title,
                    "text": text,
                    "meta": {
                        "site": urlparse(url).netloc,
                        "crawl_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "original_len_words": len(text.split())
                    }
                }
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"[OK] {url} -> {len(text.split())} words")
            except Exception as e:
                print(f"[ERR] {url} -> {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", default="data/urls.txt", help="file with newline-separated URLs")
    parser.add_argument("--out", default="data/parsed.jsonl", help="output jsonl staging file")
    args = parser.parse_args()
    main(args)
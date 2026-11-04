"""Download UCI Online Retail II into data/raw/. Run: python data/download.py"""
import pathlib, urllib.request, zipfile

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW = pathlib.Path(__file__).parent / "raw"
XLSX = RAW / "online_retail_II.xlsx"

def main():
    RAW.mkdir(exist_ok=True)
    if XLSX.exists():
        print(f"Already present: {XLSX}"); return
    zpath = RAW / "online_retail_ii.zip"
    print("Downloading...")
    try:
        urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(RAW)
        zpath.unlink()
        print("Done:", [p.name for p in RAW.iterdir()])
    except Exception as e:
        print(f"Download failed ({e}).\nManual fallback: download 'online_retail_II.xlsx' from\n"
              "https://archive.ics.uci.edu/dataset/502/online+retail+ii (or the Kaggle mirror)\n"
              f"and place it in {RAW}")

if __name__ == "__main__":
    main()

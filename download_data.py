"""Download and extract the official MovieLens Small dataset from GroupLens."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

ROOT = Path(__file__).resolve().parent
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DEST = ROOT / "data"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / "ml-latest-small"
    if (target / "movies.csv").exists() and (target / "ratings.csv").exists():
        print(f"Dataset already exists: {target}")
        return
    print(f"Downloading official MovieLens Small dataset from {URL}")
    with urlopen(URL, timeout=45) as response:
        archive = BytesIO(response.read())
    with ZipFile(archive) as zipped:
        zipped.extractall(DEST)
    print(f"Extracted MovieLens files to {target}")
    print("Review data/ml-latest-small/README.html for GroupLens terms and attribution.")


if __name__ == "__main__":
    main()

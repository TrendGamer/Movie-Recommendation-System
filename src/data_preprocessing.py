"""Load and prepare the MovieLens Small CSV files."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-latest-small"


def load_movies(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load movies and attach observed rating statistics when ratings.csv exists."""
    data_dir = Path(data_dir)
    movies_path = data_dir / "movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError(
            f"{movies_path} is missing. Run `python download_data.py` to fetch MovieLens Small."
        )
    movies = pd.read_csv(movies_path)
    movies["title"] = movies["title"].fillna("").astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str).replace("(no genres listed)", "")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce").astype("Int64")
    ratings_path = data_dir / "ratings.csv"
    if ratings_path.exists():
        ratings = pd.read_csv(ratings_path, usecols=["movieId", "rating"])
        stats = ratings.groupby("movieId")["rating"].agg(average_rating="mean", rating_count="count")
        movies = movies.merge(stats, how="left", left_on="movieId", right_index=True)
    else:
        movies["average_rating"] = float("nan")
        movies["rating_count"] = 0
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)
    return movies


def load_ratings(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "ratings.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run `python download_data.py` first.")
    return pd.read_csv(path)


def load_tags(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / "tags.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["movieId", "tag"])


def movie_features(movies: pd.DataFrame, tags: pd.DataFrame | None = None) -> pd.Series:
    """Create a text document per movie from title, genres, and observed user tags."""
    features = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True)
    features = features + " " + movies["genres"].str.replace("|", " ", regex=False)
    if tags is not None and not tags.empty:
        tag_text = tags.dropna(subset=["tag"]).groupby("movieId")["tag"].apply(lambda values: " ".join(values.astype(str)))
        features = features + " " + movies["movieId"].map(tag_text).fillna("")
    return features.str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

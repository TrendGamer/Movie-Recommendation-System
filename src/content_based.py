"""TF-IDF and cosine similarity based movie recommendations."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .data_preprocessing import movie_features


def build_similarity_matrix(movies: pd.DataFrame, tags: pd.DataFrame | None = None):
    """Return the vectorizer and a dense movie-to-movie cosine similarity matrix."""
    documents = movie_features(movies, tags)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(documents)
    # Keep similarities sparse: a dense 9,742 x 9,742 float64 matrix uses about 720 MiB.
    return vectorizer, cosine_similarity(tfidf, dense_output=False)


def recommend_similar_movies(movie_title: str, movies: pd.DataFrame, similarity, n: int = 5) -> pd.DataFrame:
    """Find the best title match, then return distinct similar movies (excluding itself)."""
    if n < 1:
        return movies.iloc[0:0].copy()
    title = str(movie_title).strip().casefold()
    exact = movies.index[movies["title"].str.casefold() == title].tolist()
    if exact:
        index = exact[0]
    else:
        matches = movies.index[movies["title"].str.casefold().str.contains(title, regex=False)].tolist()
        if not matches:
            raise ValueError(f"No movie title contains {movie_title!r}.")
        index = matches[0]
    scores = similarity[index].toarray().ravel()
    ordered = sorted(range(len(movies)), key=lambda i: (-float(scores[i]), str(movies.iloc[i]["title"])))
    seen, rows = set(), []
    selected_id = movies.iloc[index]["movieId"]
    for i in ordered:
        row = movies.iloc[i]
        if row["movieId"] == selected_id or row["movieId"] in seen:
            continue
        seen.add(row["movieId"])
        rows.append(i)
        if len(rows) >= n:
            break
    result = movies.iloc[rows].copy()
    result["similarity"] = [float(scores[i]) for i in rows]
    return result.reset_index(drop=True)

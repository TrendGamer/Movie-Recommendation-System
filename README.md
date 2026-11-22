# 🎬 CineMatch — Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

CineMatch is a beginner-friendly movie discovery app built with Python, Pandas, scikit-learn, and Streamlit. It recommends similar movies from movie titles, genres, and community tags; ranks popular films using real user ratings; and includes an optional user-based collaborative filtering demo.

> Ratings, titles, genres, and tags come from MovieLens. The dataset does not contain plot summaries or posters, so the app does not make them up. The data files are downloaded locally and excluded from Git.

## Project objectives

- Practice loading and exploring a real recommendation dataset.
- Build TF-IDF and cosine similarity content recommendations.
- Rank movies with a rating-count-aware weighted score.
- Explore simple user-based collaborative filtering and top-K evaluation.
- Share a usable, explainable Streamlit demo.

## Features

- Search the catalog and get 5–10 similar movies.
- Browse popular movies, with genre and minimum-vote filters.
- See MovieLens average ratings, rating counts, genre, and release year parsed from the title.
- Try recommendations for an existing MovieLens user ID.
- Explore an EDA notebook with readable charts and notes.
- Run without any poster API key or third-party metadata service.

## Algorithms in plain language

### Content-based filtering

The app combines title words, genres, and user-applied tags into one text document per movie. TF-IDF represents each word numerically: words that are informative for a movie get more weight than words that appear in nearly every movie. Cosine similarity compares the direction of two movie vectors. A score closer to 1 means their text features overlap more. The selected movie is excluded from results. Similarity is about metadata overlap, not a claim that people rated the films alike.

### Popularity-based ranking

The app uses a Bayesian weighted rating:

`WR = (v / (v + m)) × R + (m / (v + m)) × C`

`R` is the movie's average rating, `v` its rating count, `C` the average score in the displayed catalog, and `m` the minimum-vote prior. A high average supported by very few votes is pulled toward `C`; a well-reviewed, widely rated movie stays closer to its own average.

### Optional user-based collaborative filtering

This baseline compares user rating histories with Pearson correlation. For unrated films it aggregates ratings from similar users, weighted by similarity. Sparse overlap can leave a user with no recommendations; this is an educational baseline, not a production recommender.

## Dataset and attribution

Download **MovieLens Small** from [GroupLens MovieLens datasets](https://grouplens.org/datasets/movielens/) (direct download: [ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)). GroupLens describes the Small set as approximately 100,000 ratings, 3,600 tags, 9,000 movies and 600 users; the included dataset README is the authoritative source for the exact downloaded release and its terms. The CSV columns are:

| File | Columns | Meaning |
|---|---|---|
| `movies.csv` | `movieId`, `title`, `genres` | Movie identifier, title (often includes year), pipe-separated genre labels |
| `ratings.csv` | `userId`, `movieId`, `rating`, `timestamp` | User's observed 0.5–5 star rating and its Unix timestamp |
| `tags.csv` | `userId`, `movieId`, `tag`, `timestamp` | Free-text tag contributed by a user |
| `links.csv` | `movieId`, `imdbId`, `tmdbId` | External identifiers; not descriptions or ratings |

MovieLens includes anonymous user IDs without demographic details. Please read `data/ml-latest-small/README.html` and comply with the dataset's terms. Attribute GroupLens Research, University of Minnesota when sharing results. Data is excluded from this repository because GroupLens does not generally permit public redistribution; the download script gets it directly from the source.

## Architecture

```text
app.py                          Streamlit navigation, cards, and user interactions
download_data.py                Fetches and extracts official MovieLens Small files
src/data_preprocessing.py       CSV loading, missing-value handling, feature text
src/content_based.py            TF-IDF and sparse cosine similarity
src/popularity_based.py         Bayesian weighted-rating ranking
src/collaborative.py            Optional user-based Pearson baseline
src/evaluation.py               Leave-one-out split, Precision@K, Recall@K, RMSE
notebooks/exploratory_analysis.ipynb   EDA and explanatory charts
data/ml-latest-small/           Local dataset (downloaded; gitignored)
```

The content similarity matrix is sparse so it avoids allocating a large dense movie-by-movie matrix. It is created when the app starts and cached for that session.

## Setup and run

Requires Python 3.10 or newer.

```bash
cd "project 2"
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_data.py
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). To explore the notebook, run `jupyter notebook` and open `notebooks/exploratory_analysis.ipynb`. The model is regenerated from data at app launch; no model binary needs to be committed.

## Learning roadmap

1. Read the dataset README and inspect the four CSV files.
2. Run the EDA notebook. Compare rating counts, averages, genres, and rating distribution.
3. Read `src/data_preprocessing.py`, then `src/content_based.py` to see how text becomes vectors and similarities.
4. Explore popularity ranking and change its minimum-vote threshold.
5. Try collaborative filtering for several user IDs and observe where sparse histories matter.
6. Use an honest time-aware holdout before tuning: `leave_one_out_split` holds out each user's latest interaction. Define relevant test items (for example rating ≥ 4), generate recommendations from train history only, and average Precision@K and Recall@K across users. RMSE is suitable only for rating predictions evaluated against held-out observed ratings. This repository supplies metric helpers, not an evaluated performance claim.

## Example recommendations

Results are computed from the downloaded MovieLens version at runtime and can change with dataset contents. Choose a catalog title in **Find similar movies** to see current results; do not treat an example list copied into the README as a fixed benchmark.

## Screenshots

Add your own screenshots after running the app, for example:

- `docs/home.png` — landing page and popular picks
- `docs/similar-movies.png` — selected title and recommendations
- `docs/popular-movies.png` — genre-filtered rankings
- `docs/eda.png` — notebook chart

## Limitations

- Similarity depends on title, genre, and user tags; there are no plot summaries in this dataset.
- Content filtering has a cold-start problem for movies with little or no metadata, can over-specialize around one selected film, and has limited diversity.
- Popularity may favor mainstream titles.
- Collaborative filtering needs overlapping user histories and cannot help a new user with no ratings.
- Ratings reflect this dataset's users and collection period, not current public opinion.
- This is a learning project, not a guarantee of personal taste or a claim of measured accuracy.

## Future improvements

- Add evaluation scripts that compare methods over multiple per-user temporal splits.
- Add diversity controls and hybrid content/collaborative ranking.
- Optionally enrich movie info with a legitimate metadata API, displaying it only when configured and retaining the no-key fallback.
- Add user-entered preferences and save ratings locally with clear consent.
- Capture app and notebook screenshots for the portfolio.

## License

The code is released under the MIT License. The MovieLens dataset is separately governed by GroupLens terms; see its included README. Software license does not grant redistribution rights to MovieLens data.
"# Movie-Recommendation-System" 

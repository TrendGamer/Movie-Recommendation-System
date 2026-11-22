"""CineMatch: discover movies using real MovieLens metadata and ratings."""
from html import escape
import streamlit as st
from src.data_preprocessing import load_movies, load_ratings, load_tags
from src.content_based import build_similarity_matrix, recommend_similar_movies
from src.popularity_based import get_popular_movies
from src.collaborative import recommend_for_user

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp { background: radial-gradient(ellipse at 15% 0%, #18233a 0%, #0b0e14 48%); color: #f4f5f8; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif; }
.hero { padding: 2rem 2rem 1.6rem; border: 1px solid #293248; border-radius: 22px; background: linear-gradient(120deg,#171e2b,#131620 65%,#1e1727); margin-bottom: 1.4rem; }
.eyebrow { color:#ae8bff; text-transform:uppercase; letter-spacing:.16em; font-size:.76rem; font-weight:700; }
.muted { color:#a8afbd; }
.movie-card { border:1px solid #2b3140; border-radius:16px; padding:1.1rem; background:linear-gradient(145deg,#171b25,#11141b); min-height:168px; margin-bottom:.65rem; }
.movie-card h3 { margin:.3rem 0 .45rem; font-size:1.12rem; }
.tag { color:#c7b3ff; font-size:.84rem; }
div[data-testid="stMetric"] { background:#151923; border:1px solid #293040; padding:1rem; border-radius:14px; }
button[kind="primary"] { background:#9b75ff; border-color:#9b75ff; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading MovieLens Small…")
def data():
    return load_movies(), load_ratings(), load_tags()


@st.cache_resource(show_spinner="Preparing movie text similarity…")
def model(movie_frame, tag_frame):
    return build_similarity_matrix(movie_frame, tag_frame)[1]


def render_movies(frame, show_popularity=False):
    if frame.empty:
        st.info("No movies match these filters yet.")
        return
    for start in range(0, len(frame), 3):
        columns = st.columns(3)
        for column, (_, row) in zip(columns, frame.iloc[start:start + 3].iterrows()):
            year = int(row["year"]) if "year" in row and row["year"] == row["year"] else "Year unavailable"
            rating = f"⭐ {row['average_rating']:.1f}" if "average_rating" in row and row["average_rating"] == row["average_rating"] else "Rating unavailable"
            count = int(row.get("rating_count", 0))
            score = f" · {row['similarity']:.2f} similarity" if "similarity" in row else ""
            if show_popularity and "weighted_rating" in row:
                score = f" · {row['weighted_rating']:.2f} weighted score"
            title = escape(str(row["title"]))
            genres = escape(str(row.get("genres", "")).replace("|", " · ")) or "Genre unavailable"
            with column:
                st.markdown(f"""<div class="movie-card"><div class="eyebrow">{year}</div>
                <h3>{title}</h3><div class="tag">{genres}</div>
                <p class="muted">{rating} · {count:,} ratings{score}</p></div>""", unsafe_allow_html=True)


try:
    movies, ratings, tags = data()
    similarity = model(movies, tags)
except FileNotFoundError as error:
    st.title("🎬 CineMatch")
    st.error(str(error))
    st.code("python download_data.py\npip install -r requirements.txt\nstreamlit run app.py", language="bash")
    st.stop()

with st.sidebar:
    st.markdown("# 🎬 CineMatch")
    st.caption("Your personal movie discovery assistant")
    page = st.radio("Explore", ["Home", "Find similar movies", "Popular movies", "For a MovieLens user", "About & learning"])
    st.markdown("---")
    st.caption("Recommendations use MovieLens titles, genres, tags, and observed user ratings.")

if page == "Home":
    st.markdown("<div class='hero'><div class='eyebrow'>Movie discovery, made personal</div><h1>Find your next<br>all-time favorite.</h1><p class='muted'>Pick a film you love and CineMatch finds movies with similar genres, titles, and community tags. Or browse well-rated favorites backed by real MovieLens ratings.</p></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Movies", f"{len(movies):,}")
    c2.metric("User ratings", f"{len(ratings):,}")
    c3.metric("Community tags", f"{len(tags):,}")
    st.subheader("Popular picks")
    render_movies(get_popular_movies(movies, n=6, min_votes=100), show_popularity=True)
    st.markdown("<p class='muted'>Content recommendations use TF-IDF to represent words by how informative they are, then cosine similarity to compare movie descriptions. Popularity uses a weighted average so a handful of votes cannot dominate the chart.</p>", unsafe_allow_html=True)

elif page == "Find similar movies":
    st.markdown("<div class='hero'><div class='eyebrow'>Content-based recommendations</div><h1>What have you enjoyed?</h1><p class='muted'>Search or choose a movie. We compare its title, genres, and user tags with the rest of the catalog.</p></div>", unsafe_allow_html=True)
    query = st.text_input("Search titles", placeholder="Try The Matrix")
    choices = movies["title"].tolist()
    if query.strip():
        filtered = [title for title in choices if query.casefold() in title.casefold()]
        if filtered:
            choices = filtered
        else:
            st.warning("No titles matched that search. Try another spelling.")
    selected = st.selectbox("Choose a movie", choices)
    n = st.slider("Number of recommendations", 5, 10, 5)
    if st.button("Recommend movies", type="primary"):
        try:
            result = recommend_similar_movies(selected, movies, similarity, n=n)
            st.subheader(f"Because you selected {selected}")
            render_movies(result)
        except ValueError as error:
            st.warning(str(error))

elif page == "Popular movies":
    st.markdown("<div class='hero'><div class='eyebrow'>Loved by MovieLens users</div><h1>Popular movies</h1><p class='muted'>Ranked by a Bayesian weighted rating that balances average score with the number of ratings.</p></div>", unsafe_allow_html=True)
    genres = sorted({g for value in movies["genres"].dropna() for g in value.split("|") if g and g != "(no genres listed)"})
    genre = st.selectbox("Filter by genre", ["All genres"] + genres)
    minimum = st.slider("Minimum number of ratings", 10, 300, 50, step=10)
    count = st.slider("Movies to show", 5, 30, 15)
    render_movies(get_popular_movies(movies, n=count, min_votes=minimum, genre=genre), show_popularity=True)

elif page == "For a MovieLens user":
    st.markdown("<div class='hero'><div class='eyebrow'>Optional collaborative filtering</div><h1>Find picks for a user</h1><p class='muted'>User-based filtering looks for people with overlapping rating histories and predicts movies you have not rated.</p></div>", unsafe_allow_html=True)
    user_id = st.number_input("MovieLens user ID", min_value=1, max_value=int(ratings.userId.max()), value=1, step=1)
    if st.button("Find user recommendations", type="primary"):
        try:
            result = recommend_for_user(int(user_id), ratings, movies, n=10)
            if result.empty:
                st.info("There was not enough overlapping rating history to make recommendations for this user.")
            else:
                st.caption("Predicted rating is an experimental user-similarity baseline, not an observed rating.")
                render_movies(result)
        except ValueError as error:
            st.warning(str(error))

else:
    st.markdown("<div class='hero'><div class='eyebrow'>Learn the building blocks</div><h1>How CineMatch works</h1></div>", unsafe_allow_html=True)
    st.subheader("TF-IDF + cosine similarity")
    st.write("TF-IDF turns each movie's title, genre names, and community tags into weighted word values. A rare useful word gets more weight than a word found everywhere. Cosine similarity compares the angle between two movie vectors: values closer to 1 indicate more similar text profiles.")
    st.subheader("Weighted popularity")
    st.latex(r"WR = \frac{v}{v+m}R + \frac{m}{v+m}C")
    st.write("R is a movie's average rating, v is its number of ratings, C is the catalog average, and m is the minimum-count prior. A movie with few ratings is pulled toward the overall average.")
    st.subheader("Dataset & limitations")
    st.write("MovieLens Small contains movies.csv (movie ID, title, genres), ratings.csv (user ID, movie ID, 0.5–5 rating, timestamp), tags.csv (user-applied tags), and links.csv (external identifiers). It has no plot descriptions or poster images, so the app avoids inventing them. Content-based methods can over-specialize and cannot personalize a new user's taste without history; popularity favors broadly liked movies. The optional collaborative page needs enough overlapping ratings.")
    st.write("For honest offline evaluation, hold out later interactions per user, generate recommendations from the remaining history, then measure precision@K and recall@K against held-out relevant ratings. RMSE applies only when you produce rating predictions. Evaluation helpers are in `src/evaluation.py`; no accuracy claim is made by this demo.")
    st.caption("MovieLens data is provided by GroupLens Research, University of Minnesota. Review the included dataset README for terms and attribution.")

import os
import zipfile
import requests
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# 1. Download MovieLens Dataset
# ============================================================

def download_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = "ml-latest-small.zip"

    if not os.path.exists("ml-latest-small"):
        print("Downloading MovieLens small dataset...")
        r = requests.get(url)
        open(zip_path, "wb").write(r.content)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall()

        print("Extracted to ml-latest-small")
    else:
        print("Dataset already exists.")

    movies = pd.read_csv("ml-latest-small/movies.csv")
    ratings = pd.read_csv("ml-latest-small/ratings.csv")
    return movies, ratings


# ============================================================
# 2. Build User–Item Matrix
# ============================================================

def build_user_item_matrix(ratings):
    pivot = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    return pivot.fillna(0)


# ============================================================
# 3. Item-Based Collaborative Filtering (Cosine)
# ============================================================

def build_item_item_cf(pivot):
    print("\nBuilding item-item collaborative filter (cosine)...")
    item_sim = cosine_similarity(pivot.T)
    item_sim_df = pd.DataFrame(item_sim, index=pivot.columns, columns=pivot.columns)
    return item_sim_df


def recommend_cf(user_id, pivot, item_sim_df, movies, top_n=10):
    user_ratings = pivot.loc[user_id]
    liked = user_ratings[user_ratings > 4].index.tolist()

    scores = {}
    for movie in liked:
        similar_movies = item_sim_df[movie].sort_values(ascending=False)
        for m, score in similar_movies.items():
            if m not in liked:
                scores[m] = scores.get(m, 0) + score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    movie_ids = [m for m, _ in ranked]
    return movies[movies["movieId"].isin(movie_ids)]


# ============================================================
# 4. Content-Based (TF-IDF on Genres)
# ============================================================

def build_tfidf_genres(movies):
    print("Building content TF-IDF (genres)...")
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies["genres"])
    return tfidf, tfidf_matrix


def recommend_content(movie_id, movies, tfidf_matrix, tfidf, top_n=10):
    idx = movies.index[movies["movieId"] == movie_id][0]
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    top = np.argsort(sim)[-top_n-1:-1][::-1]
    return movies.iloc[top]


# ============================================================
# 5. SVD Approximation for Matrix Factorization
# ============================================================

def svd_reconstruct_and_eval(pivot, ratings, n_components=20, test_fraction=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    train_r, test_r = train_test_split(ratings, test_size=test_fraction, random_state=random_state)

    pivot_train = train_r.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)

    pivot_train = pivot_train.reindex(index=pivot.index, columns=pivot.columns, fill_value=0)

    k = min(n_components, min(pivot_train.shape) - 1)
    svd = TruncatedSVD(n_components=max(2, k), random_state=random_state)

    U = svd.fit_transform(pivot_train)
    VT = svd.components_
    recon = np.dot(U, VT)

    preds, truths = [], []
    for _, row in test_r.iterrows():
        u = row["userId"]
        m = row["movieId"]
        r = row["rating"]

        if u in pivot.index and m in pivot.columns:
            ui = pivot.index.get_loc(u)
            mi = pivot.columns.get_loc(m)
            preds.append(recon[ui, mi])
            truths.append(r)

    rmse = (mean_squared_error(truths, preds) ** 0.5) if preds else None
    return svd, rmse


# ============================================================
# 6. Main Driver
# ============================================================

def main():
    movies, ratings = download_movielens()
    print(f"Movies: {movies.shape} Ratings: {ratings.shape}")

    pivot = build_user_item_matrix(ratings)
    print(f"User-item matrix: {pivot.shape}")

    item_sim_df = build_item_item_cf(pivot)
    tfidf, tfidf_matrix = build_tfidf_genres(movies)

    # --------------------------
    # Collaborative Filtering
    # --------------------------
    user_id = 1
    print(f"\nSample user id: {user_id}")

    cf_recs = recommend_cf(user_id, pivot, item_sim_df, movies)
    print("\nItem-based CF recommendations:")
    for t in cf_recs["title"].tolist():
        print(" -", t)

    # --------------------------
    # Content-Based
    # --------------------------
    seed_movie = 11
    print(f"\nSeed movie: {seed_movie} — {movies[movies['movieId']==seed_movie]['title'].values[0]}")

    content_recs = recommend_content(seed_movie, movies, tfidf_matrix, tfidf)
    print("Content-based recommendations:")
    for t in content_recs["title"].tolist():
        print(" -", t)

    # --------------------------
    # SVD
    # --------------------------
    print("\nRunning SVD approximation & evaluation...")
    svd, rmse = svd_reconstruct_and_eval(pivot, ratings, n_components=20)

    print(f"SVD RMSE: {rmse:.4f}")

    print("\n🎯 Done!")


if __name__ == "__main__":
    main()
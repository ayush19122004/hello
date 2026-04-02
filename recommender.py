from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import pandas as pd
import random
from datetime import datetime
import json
import os

from config import SECRET_KEY, SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from models import db, Signup, SearchHistory, Wishlist
from sentiment import predict_sentiment
from rating_model import predict_rating
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS
db.init_app(app)

# -------------------------
# Jinja filters
# -------------------------
@app.template_filter("truncate")
def truncate_filter(s, length=40):
    if not s:
        return ""
    s = str(s)
    return s if len(s) <= length else s[:length] + "..."

def comma_separate_filter(value):
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return value

app.jinja_env.filters['comma_separate'] = comma_separate_filter

# -------------------------
# Price points helper
# -------------------------
PRICE_POINTS = [
    199, 299, 499, 999, 1499, 1999, 2999, 3999,
    5999, 7999, 9999, 12999, 19999, 29999, 49999, 69999, 99999
]

def add_random_price(df):
    """Add random_price column if absent."""
    if df is None or df.empty:
        return df
    if 'random_price' not in df.columns:
        df = df.copy()
        df['random_price'] = [random.choice(PRICE_POINTS) for _ in range(len(df))]
    return df

# -------------------------
# Load dataset
# -------------------------
DATA_CSV = os.path.join("data", "amazon_combined_20k.csv")
if os.path.exists(DATA_CSV):
    df = pd.read_csv(DATA_CSV, index_col='asin', dtype=object)
    df.fillna("", inplace=True)
    # Normalize numeric columns safely
    df['overall'] = pd.to_numeric(df.get('overall', 0), errors='coerce').fillna(0.0).astype(float)
    # reviewCount fallback
    if 'reviewCount' in df.columns:
        df['reviewCount'] = pd.to_numeric(df['reviewCount'], errors='coerce').fillna(0).astype(int)
    else:
        df['reviewCount'] = df.groupby(df.index)["reviewText"].transform("count") if 'reviewText' in df.columns else 0
    # combined fields
    text_cols = ["title", "description", "brand", "category"]
    for c in text_cols:
        if c not in df.columns:
            df[c] = ""
    df["combined_meta"] = (df["title"].astype(str) + " " + df["description"].astype(str) + " " +
                           df["brand"].astype(str) + " " + df["category"].astype(str)).str.strip()
    df = add_random_price(df)
else:
    # empty skeleton
    cols = ['title', 'description', 'brand', 'category', 'overall', 'reviewCount', 'random_price', 'combined_meta']
    df = pd.DataFrame(columns=cols)
    df.index.name = 'asin'
    df = add_random_price(df)

# -------------------------
# Trending storage (persist to JSON)
# -------------------------
TREND_FILE = "trending.json"

def load_trending():
    if os.path.exists(TREND_FILE):
        try:
            with open(TREND_FILE, "r") as f:
                data = json.load(f)
            # keys are asin -> int score
            return {str(k): int(v) for k,v in data.items()}
        except Exception:
            return {}
    return {}

def save_trending(trend_map):
    try:
        with open(TREND_FILE, "w") as f:
            json.dump(trend_map, f)
    except Exception:
        pass

trending_scores = load_trending()

def update_trending(asin, weight=1, allow_low=False):
    """
    Increment trending score for asin.
    Only increments if product exists in df and is high-rated (overall >= 4.0),
    unless allow_low=True (used for newly created high-rated items where overall might be set programmatically).
    """
    try:
        asin = str(asin)
        if asin not in df.index:
            return
        overall = df.at[asin, 'overall'] if 'overall' in df.columns else 0.0
        try:
            overall_val = float(overall)
        except Exception:
            overall_val = 0.0
        if overall_val < 4.0 and not allow_low:
            return
        trending_scores[asin] = trending_scores.get(asin, 0) + int(weight)
        save_trending(trending_scores)
    except Exception:
        pass

def get_trending_products(limit=8):
    # Step 1: Filter only products that still exist and are high-rated
    valid_scores = {
        k: v for k, v in trending_scores.items()
        if k in df.index
    }

    # Step 2: Sort by score (highest trending first)
    sorted_asins = sorted(valid_scores.keys(), key=lambda k: valid_scores[k], reverse=True)

    trending_list = []

    # Step 3: Add trending products first
    for asin in sorted_asins:
        if asin in df.index and len(trending_list) < limit:
            trending_list.append(asin)

    # Step 4: Fill remaining slots using top overall rating
    if len(trending_list) < limit:
        top_rated = df.sort_values("overall", ascending=False).index.tolist()
        for asin in top_rated:
            if asin not in trending_list and len(trending_list) < limit:
                trending_list.append(asin)

    # Step 5: Final fallback → add any products to reach 8
    if len(trending_list) < limit:
        remaining = df.index.tolist()
        for asin in remaining:
            if asin not in trending_list and len(trending_list) < limit:
                trending_list.append(asin)

    # Step 6: Build final DataFrame
    result = df.loc[trending_list].copy()
    result = add_random_price(result)
    return result

    

# -------------------------
# Recommender (your content_based_recommendations)
# -------------------------
def content_based_recommendations(df_local, item_name, top_n=10, min_sim=0.2):
    """
    Returns a DataFrame of recommendations. The returned df preserves index as asin (if df_local has asin index).
    """
    if df_local is None or df_local.empty:
        return pd.DataFrame(columns=['title','brand','category','overall','reviewCount'])

    # STEP 1: Build global TF-IDF → used for matching input text to best product
    try:
        tfidf_full = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix_full = tfidf_full.fit_transform(df_local['combined_meta'].astype(str).tolist())
    except Exception:
        # fallback: can't build global tfidf
        return pd.DataFrame(columns=['title','brand','category','overall','reviewCount'])

    # STEP 2: Identify Target Item
    if item_name in df_local['title'].values:
        target_item = df_local[df_local['title'] == item_name].iloc[0]
    else:
        input_vector = tfidf_full.transform([item_name])
        similarity_to_input = cosine_similarity(input_vector, tfidf_matrix_full).ravel()
        best_match_idx = int(np.argmax(similarity_to_input))

        if similarity_to_input[best_match_idx] < 0.05:
            # No meaningful match
            return pd.DataFrame(columns=['title','brand','category','overall','reviewCount'])

        target_item = df_local.iloc[best_match_idx]

    # STEP 3: Filter only category-matching items
    item_cat = target_item.get('category', "")
    candidates = df_local[df_local['category'] == item_cat].copy()
    if candidates.empty:
        # fallback to whole df
        candidates = df_local.copy()

    # STEP 4: Category-specific TF-IDF
    try:
        tfidf_cat = TfidfVectorizer(stop_words='english', max_features=2000)
        tfidf_matrix_cat = tfidf_cat.fit_transform(candidates['combined_meta'].astype(str).tolist())
    except Exception:
        return pd.DataFrame(columns=['title','brand','category','overall','reviewCount'])

    # Re-vectorize target item in same vector space (if target not in candidates, transform its combined_meta)
    try:
        target_vector = tfidf_cat.transform([str(target_item['combined_meta'])])
    except Exception:
        # If error, try to use the closest candidate as target
        target_vector = tfidf_cat.transform([candidates.iloc[0]['combined_meta']])

    # Similarity scores
    sims = cosine_similarity(target_vector, tfidf_matrix_cat).ravel()

    rec_df = candidates.copy()
    rec_df['sim_score'] = sims.tolist()

    # Remove the selected product itself (using index)
    rec_df = rec_df[rec_df.index != target_item.name]

    # Filter low-similarity items
    rec_df_filtered = rec_df[rec_df['sim_score'] >= min_sim].copy()

    # STEP 6: Fallback if empty
    if rec_df_filtered.empty:
        fallback = candidates[candidates.index != target_item.name] \
            .sort_values(by="overall", ascending=False) \
            .head(top_n)
        return fallback[['title','brand','category','overall','reviewCount']]

    # STEP 7: Predict High/Low rating using your model
    # predict_rating should return "High"/"Low" or 1/0
    rec_df_filtered['predicted_overall'] = rec_df_filtered['combined_meta'].apply(lambda x: predict_rating(str(x)))

    # Convert High/Low/1/0 → numeric (1 = high, 0 = low)
    rec_df_filtered['predicted_overall'] = rec_df_filtered['predicted_overall'].map({
        "High": 1,
        "Low": 0,
        1: 1,
        0: 0
    }).fillna(0)

    # STEP 8: Weighted scoring for ranking
    rec_df_filtered['combined_score'] = (
        0.7 * rec_df_filtered['sim_score'] +
        0.3 * (rec_df_filtered['predicted_overall'] / 5.0)
    )

    # Top results
    final_recs = rec_df_filtered.sort_values(by="combined_score", ascending=False).head(top_n)

    return final_recs[['title','brand','category','overall','reviewCount']]

# -------------------------
# Helper: boost similar items into trending
# -------------------------
def boost_similar_items(asin, top_k=3, min_sim=0.12, weight=1):
    """
    Find items similar to the given asin (using recommender) and bump their trending score.
    """
    try:
        asin = str(asin)
        if asin not in df.index:
            return
        # use the product title as input to the recommender
        title = df.at[asin, 'title'] if 'title' in df.columns else ""
        if not title:
            return
        sims = content_based_recommendations(df, title, top_n=top_k, min_sim=min_sim)
        if isinstance(sims, pd.DataFrame) and not sims.empty:
            # sims should preserve index as asin
            for sim_asin in sims.index.tolist():
                if sim_asin and sim_asin in df.index and sim_asin != asin:
                    update_trending(sim_asin, weight=weight)
    except Exception:
        pass

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    # Show top 8 trending items on index
    trending_products = get_trending_products(limit=8)

    # wishlist recs
    wishlist_recs = pd.DataFrame()
    if "username" in session:
        username = session["username"]
        try:
            user_wishlist = Wishlist.query.filter_by(username=username).all()
            if user_wishlist and not df.empty:
                wishlist_ids = [w.product_id for w in user_wishlist]
                wishlist_recs = df.loc[df.index.isin(wishlist_ids)]
                wishlist_recs = add_random_price(wishlist_recs)
        except Exception:
            wishlist_recs = pd.DataFrame()

    # recent searches
    recent_searches = []
    if "username" in session:
        try:
            recent_searches = SearchHistory.query.filter_by(username=session["username"]).order_by(
                SearchHistory.timestamp.desc()
            ).limit(10).all()
        except Exception:
            recent_searches = []

    # If a new product was just added (session), show banner
    new_product = session.pop('new_product', None)
    new_product_recs_data = session.pop('new_product_recs_data', [])
    new_product_recs = pd.DataFrame()
    if new_product_recs_data:
        try:
            new_product_recs = pd.DataFrame(new_product_recs_data).set_index('asin')
        except Exception:
            new_product_recs = pd.DataFrame()

    return render_template("index.html",
                           trending_products=trending_products,
                           wishlist_recs=wishlist_recs,
                           recent_searches=recent_searches,
                           new_product=new_product,
                           new_product_recs=new_product_recs,
                           signup_message=session.pop("signup_message", None)
                           )

@app.route("/main")
def main():
    # show blank recommendation UI
    return render_template("main.html",
                           content_based_rec=pd.DataFrame(columns=df.columns),
                           message=None,
                           sentiment=None,
                           truncate=lambda t, l=60: t[:l]+"..." if isinstance(t,str) and len(t)>l else t
                           )

@app.route("/recommendations", methods=["POST"])
def recommendations():
    prod_input = request.form.get("prod", "").strip()
    review_text = request.form.get("review", "").strip()
    nbr_raw = request.form.get("nbr", "10")
    try:
        nbr = max(1, min(20, int(nbr_raw)))
    except ValueError:
        nbr = 10

    sentiment = predict_sentiment(review_text) if review_text else None

    # Save search history
    if prod_input and session.get("username"):
        try:
            history = SearchHistory(
                username=session["username"],
                product_searched=prod_input,
                timestamp=datetime.now()
            )
            db.session.add(history)
            db.session.commit()
        except Exception:
            db.session.rollback()

    # Get recommendations using the recommender (operate on df)
    try:
        recs = content_based_recommendations(df, prod_input, top_n=nbr, min_sim=0.2)
    except Exception:
        recs = pd.DataFrame()

    recs = add_random_price(recs)

    # IMPORTANT: bump trending for top returned items from this search (search-driven trending)
    if isinstance(recs, pd.DataFrame) and not recs.empty:
        # bump top 3 items returned (or fewer)
        top_to_bump = recs.head(min(3, len(recs))).index.tolist()
        for asin in top_to_bump:
            update_trending(asin, weight=1)
        message = f"Recommendations for '{prod_input}':"
        return render_template("main.html",
                               content_based_rec=recs.head(nbr),
                               message=message,
                               sentiment=sentiment,
                               truncate=lambda t, l=60: t[:l]+"..." if isinstance(t,str) and len(t)>l else t
                               )
    else:
        fallback = df.sort_values(by="overall", ascending=False).head(5)[['title', 'brand', 'category', 'overall']] if not df.empty else pd.DataFrame(columns=df.columns)
        fallback = add_random_price(fallback)
        message = f"No exact match or suitable recommendations for '{prod_input}'. Showing top trending items instead."
        return render_template("main.html",
                               content_based_rec=fallback,
                               message=message,
                               sentiment=sentiment,
                               truncate=lambda t, l=60: t[:l]+"..." if isinstance(t,str) and len(t)>l else t
                               )

@app.route("/predict_rating", methods=["POST"])
def predict_rating_api():
    data = request.get_json() or {}
    # Accept both 'text' or combination fields
    text = data.get("text") or " ".join([data.get(k,"") for k in ["title","brand","category","description"] if data.get(k)])
    label = predict_rating(text)  # "High" or "Low"
    return jsonify({"rating": label})

@app.route("/add", methods=["GET", "POST"])
def add():
    if request.method == "POST":
        title = request.form.get("title", "Untitled Product")
        desc = request.form.get("description", "")
        brand = request.form.get("brand", "Unspecified")
        category = request.form.get("category", "Unspecified")

        combined = f"{title} {brand} {category} {desc}"
        label = predict_rating(combined)  # returns "High" or "Low"
        is_high = (label == "High" or label == 1 or label == "1")

        # Build new product dict and add to df only if High
        if is_high:
            new_asin = "new_prod_" + str(int(datetime.now().timestamp()))
            new_product = {
                "title": title,
                "description": desc,
                "brand": brand,
                "category": category,
                "overall": 5.0,  # mark excellent numeric for display
                "reviewCount": 0,
                "random_price": random.choice(PRICE_POINTS),
                "combined_meta": combined
            }
            # insert new row into df (index = asin)
            try:
                # set item using .loc to preserve dtype/columns
                df.loc[new_asin] = new_product
            except Exception:
                try:
                    row = pd.DataFrame(new_product, index=[new_asin])
                    df = pd.concat([df, row])
                except Exception:
                    pass

            # compute recommendations for banner (use high-similarity on existing df)
            try:
                recs = content_based_recommendations(df, combined, top_n=4, min_sim=0.1)
            except Exception:
                recs = pd.DataFrame()
            recs = add_random_price(recs)
            recs_for_session = []
            if isinstance(recs, pd.DataFrame) and not recs.empty:
                # If recommender preserved index as asin, include that; else try to include title
                try:
                    recs_for_session = recs.reset_index()[["asin","title","brand","category","overall","random_price"]].to_dict("records")
                except Exception:
                    recs_for_session = recs.reset_index().rename(columns={recs.index.name or 0: "asin"}).to_dict("records")
            else:
                recs_for_session = []

            # store new product in session to display banner on index
            session['new_product'] = {"asin": new_asin, "title": title, "brand": brand, "overall": 5.0, "random_price": new_product["random_price"]}
            session['new_product_recs_data'] = recs_for_session

            # Allow the new product to be trended immediately (allow_low True to bypass overall check)
            try:
                update_trending(new_asin, weight=5, allow_low=True)
            except Exception:
                pass

            # Also boost similar items
            try:
                boost_similar_items(new_asin, top_k=4, min_sim=0.12, weight=1)
            except Exception:
                pass

            # Optionally flash success and redirect to index
            flash("Product added and promoted to trending (if high-rated).", "success")
            return redirect(url_for("index"))
        else:
            flash("Product predicted Low — not added to catalog.", "warning")
            return redirect(url_for("main"))

    # GET request
    return render_template("add.html")

# -------------------------
# Wishlist endpoints
# -------------------------
@app.route("/add_wishlist", methods=["POST"])
def add_wishlist():
    if 'username' not in session:
        flash("Please sign in to add to wishlist.", "warning")
        return redirect(url_for("main"))

    asin = request.form.get("asin")
    username = session.get("username")
    if not asin:
        flash("No product specified.", "danger")
        return redirect(request.referrer or url_for("index"))

    try:
        # avoid duplicates
        existing = Wishlist.query.filter_by(username=username, product_id=asin).first()
        if existing:
            flash("Already in wishlist.", "info")
            return redirect(request.referrer or url_for("index"))

        entry = Wishlist(username=username, product_id=asin)
        db.session.add(entry)
        db.session.commit()

        # Boost trending for wishlist add (stronger signal)
        try:
            update_trending(asin, weight=3)
        except Exception:
            pass

        # Boost similar items too (small weight)
        try:
            boost_similar_items(asin, top_k=3, min_sim=0.12, weight=1)
        except Exception:
            pass

        flash("Added to wishlist and promoted to trending.", "success")
    except Exception as e:
        db.session.rollback()
        flash("Unable to add to wishlist.", "danger")

    return redirect(request.referrer or url_for("index"))

@app.route("/remove_wishlist", methods=["POST"])
def remove_wishlist():
    if 'username' not in session:
        return redirect(url_for("main"))
    asin = request.form.get("asin")
    username = session.get("username")
    try:
        Wishlist.query.filter_by(username=username, product_id=asin).delete()
        db.session.commit()
        flash("Removed from wishlist.", "success")
    except Exception:
        db.session.rollback()
        flash("Could not remove from wishlist.", "danger")
    return redirect(request.referrer or url_for("index"))

# -------------------------
# App runner
# -------------------------
if __name__ == "__main__":
    # flask run
    app.run(debug=True, host="0.0.0.0", port=5000)

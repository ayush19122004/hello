from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import pandas as pd
import random
from datetime import datetime
import json
import os

from config import SECRET_KEY, SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from models import db, Signup, SearchHistory, Wishlist
from sentiment import predict_sentiment
from recommender import content_based_recommendations
from rating_model import predict_rating

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
    df['overall'] = pd.to_numeric(df.get('overall', 0), errors='coerce').fillna(0.0).astype(float)
    if 'reviewCount' in df.columns:
        df['reviewCount'] = pd.to_numeric(df['reviewCount'], errors='coerce').fillna(0).astype(int)
    else:
        df['reviewCount'] = df.groupby(df.index)["reviewText"].transform("count") if 'reviewText' in df.columns else 0
    text_cols = ["title", "description", "brand", "category"]
    for c in text_cols:
        if c not in df.columns:
            df[c] = ""
    df["combined_meta"] = (df["title"].astype(str) + " " + df["description"].astype(str) + " " +
                           df["brand"].astype(str) + " " + df["category"].astype(str)).str.strip()
    df = add_random_price(df)
else:
    cols = ['title', 'description', 'brand', 'category', 'overall', 'reviewCount', 'random_price', 'combined_meta']
    df = pd.DataFrame(columns=cols)
    df.index.name = 'asin'
    df = add_random_price(df)

# -------------------------
# Trending storage
# -------------------------
TREND_FILE = "trending.json"

def load_trending():
    if os.path.exists(TREND_FILE):
        try:
            with open(TREND_FILE, "r") as f:
                data = json.load(f)
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
    try:
        asin = str(asin)
        if asin not in df.index:
            return
        overall = float(df.at[asin, 'overall'] if 'overall' in df.columns else 0.0)
        if overall < 4.0 and not allow_low:
            return
        trending_scores[asin] = trending_scores.get(asin, 0) + int(weight)
        save_trending(trending_scores)
    except Exception:
        pass

def get_trending_products(limit=8):
    valid_scores = {k:v for k,v in trending_scores.items() if k in df.index and float(df.at[k,'overall']) >= 4.0}
    if valid_scores:
        sorted_asins = sorted(valid_scores.keys(), key=lambda k: valid_scores[k], reverse=True)
        asins_to_show = sorted_asins[:limit]
        res = df.loc[df.index.isin(asins_to_show)].copy()
        res['__trend_score'] = res.index.map(lambda x: valid_scores.get(x, 0))
        res = res.sort_values('__trend_score', ascending=False).drop(columns='__trend_score')
        return add_random_price(res)
    else:
        fallback = df[df['overall'] >= 4.0].sort_values('overall', ascending=False).head(limit)
        return add_random_price(fallback)

def extract_value(data, key, asin):
    value = data.get(key, {})
    if isinstance(value, dict) and asin in value:
        return value[asin]
    if isinstance(value, dict) and len(value) == 1:
        return next(iter(value.values()))
    return value if isinstance(value, (str, int, float)) else ""

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    trending_products = get_trending_products(limit=8)
    wishlist_recs = pd.DataFrame()
    if "username" in session:
        try:
            user_wishlist = Wishlist.query.filter_by(username=session["username"]).all()
            if user_wishlist and not df.empty:
                wishlist_ids = [w.product_id for w in user_wishlist]
                wishlist_recs = df.loc[df.index.isin(wishlist_ids)]
                wishlist_recs = add_random_price(wishlist_recs)
        except Exception:
            wishlist_recs = pd.DataFrame()

    recent_searches = []
    if "username" in session:
        try:
            recent_searches = SearchHistory.query.filter_by(username=session["username"]).order_by(
                SearchHistory.timestamp.desc()
            ).limit(10).all()
        except Exception:
            recent_searches = []

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
    try:
        nbr = max(1, min(20, int(request.form.get("nbr", "10"))))
    except ValueError:
        nbr = 10

    sentiment = predict_sentiment(review_text) if review_text else None

    if prod_input and session.get("username"):
        try:
            history = SearchHistory(username=session["username"], product_searched=prod_input, timestamp=datetime.now())
            db.session.add(history)
            db.session.commit()
        except Exception:
            db.session.rollback()

    try:
        recs = content_based_recommendations(df, prod_input, top_n=nbr, min_sim=0.2)
    except Exception:
        recs = pd.DataFrame()

    recs = add_random_price(recs)

    if isinstance(recs, pd.DataFrame) and not recs.empty:
        top_to_bump = recs.head(min(3, len(recs))).index.tolist()
        for asin in top_to_bump:
            update_trending(asin, weight=1)
        message = f"Recommendations for '{prod_input}':"
        return render_template("main.html", content_based_rec=recs.head(nbr), message=message, sentiment=sentiment,
                               truncate=lambda t, l=60: t[:l]+"..." if isinstance(t,str) and len(t)>l else t)
    else:
        fallback = df.sort_values(by="overall", ascending=False).head(5)[['title', 'brand', 'category', 'overall']] if not df.empty else pd.DataFrame(columns=df.columns)
        fallback = add_random_price(fallback)
        message = f"No exact match or suitable recommendations for '{prod_input}'. Showing top trending items instead."
        return render_template("main.html", content_based_rec=fallback, message=message, sentiment=sentiment,
                               truncate=lambda t, l=60: t[:l]+"..." if isinstance(t,str) and len(t)>l else t)

@app.route("/predict_rating", methods=["POST"])
def predict_rating_api():
    data = request.get_json() or {}
    text = data.get("text") or " ".join([data.get(k,"") for k in ["title","brand","category","description"] if data.get(k)])
    label = predict_rating(text)
    return jsonify({"rating": label})

@app.route("/add", methods=["GET", "POST"])
def add():
    if request.method == "POST":
        # Step 1: Collect user input
        title = request.form.get("title", "Untitled Product")
        desc = request.form.get("description", "")
        brand = request.form.get("brand", "Unspecified")
        category = request.form.get("category", "Unspecified")

        # Step 2: Predict rating
        combined = f"{title} {brand} {category} {desc}"
        label = predict_rating(combined)
        is_high = (label == "High")

        # Step 3: Show prediction first, then add only if High
        if is_high:
            new_asin = "new_prod_" + str(int(datetime.now().timestamp()))
            new_product = {
                "title": title,
                "description": desc,
                "brand": brand,
                "category": category,
                "overall": random.choice([4.0, 5.0]),
                "reviewCount": 0,
                "random_price": random.choice(PRICE_POINTS)
            }
            try:
                df.loc[new_asin] = new_product
            except Exception:
                row = pd.DataFrame(new_product, index=[new_asin])
                df.append(row)

            # Add to trending immediately for High-rated
            update_trending(new_asin, weight=2, allow_low=False)

            # Get similar products recommendations
            try:
                recs = content_based_recommendations(df, combined, top_n=4, min_sim=0.1)
            except Exception:
                recs = pd.DataFrame()
            recs = add_random_price(recs)
            recs_for_session = []
            if isinstance(recs, pd.DataFrame) and not recs.empty:
                recs_for_session = recs.reset_index()[["asin","title","brand","category","overall","random_price"]].to_dict("records")
            session['new_product'] = {"asin": new_asin, "title": title, "brand": brand,
                                      "overall": new_product["overall"], "random_price": new_product["random_price"]}
            session['new_product_recs_data'] = recs_for_session

            flash(f"Prediction Success! Your product '{title}' was predicted High-rated and added.", "success")
        else:
            flash(f"Prediction Result: Your product '{title}' was predicted Low-rated and not added.", "warning")
            session.pop('new_product', None)
            session.pop('new_product_recs_data', None)

        # Step 4: Redirect to index to show trending/new product
        return redirect(url_for("index"))

    # GET request → show form
    return render_template("add.html", prediction=None)


@app.route("/product/<string:product_id>")
def product_detail(product_id):
    if product_id not in df.index:
        return "Product not found", 404

    raw_product = df.loc[product_id].to_dict()
    product = {
        'asin': product_id,
        'title': extract_value(raw_product, 'title', product_id),
        'brand': extract_value(raw_product, 'brand', product_id),
        'category': extract_value(raw_product, 'category', product_id),
        'description': extract_value(raw_product, 'description', product_id),
        'overall': float(extract_value(raw_product, 'overall', product_id) or 0.0),
        'reviewCount': int(float(extract_value(raw_product, 'reviewCount', product_id) or 0)),
        'random_price': int(extract_value(raw_product, 'random_price', product_id) or random.choice(PRICE_POINTS))
    }

    update_trending(product_id, weight=1)  # only High-rated actually trended inside function

    category = product.get("category", "")
    if not df.empty and category:
        final_similar = df[
            (df["category"].astype(str).str.lower() == category.lower()) &
            (df.index != product_id)
        ].sort_values(by="overall", ascending=False).head(5)
    else:
        final_similar = pd.DataFrame(columns=df.columns)

    final_similar = add_random_price(final_similar)
    similar_products_list = []
    for asin, row in final_similar.iterrows():
        item_dict = row.to_dict()
        cleaned_item = {
            'asin': asin,
            'title': extract_value(item_dict, 'title', asin),
            'brand': extract_value(item_dict, 'brand', asin),
            'category': extract_value(item_dict, 'category', asin),
            'overall': float(extract_value(item_dict, 'overall', asin) or 0.0),
            'reviewCount': int(float(extract_value(item_dict, 'reviewCount', asin) or 0)),
            'random_price': int(extract_value(item_dict, 'random_price', asin) or random.choice(PRICE_POINTS))
        }
        similar_products_list.append(cleaned_item)

    return render_template("product_detail.html",
                           product=product,
                           similar_products=similar_products_list,
                           truncate=lambda t, l=60: t[:l]+"..." if isinstance(t,str) and len(t)>l else t
                           )

@app.route("/wishlist/add/<string:product_id>", methods=["POST"])
def add_to_wishlist(product_id):
    if session.get("username") and product_id in df.index:
        existing = Wishlist.query.filter_by(username=session["username"], product_id=product_id).first()
        if not existing:
            try:
                entry = Wishlist(username=session["username"], product_id=product_id, timestamp=datetime.now())
                db.session.add(entry)
                db.session.commit()
            except Exception:
                db.session.rollback()
        # Wishlist always boosts trending
        update_trending(product_id, weight=2)
    return redirect(url_for("product_detail", product_id=product_id))

@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("username")
    password = request.form.get("password")
    email = request.form.get("email")

    if Signup.query.filter_by(username=username).first():
        flash("Username already exists", "danger")
        return redirect(url_for("main"))

    new_user = Signup(username=username, password=password, email=email)
    db.session.add(new_user)
    db.session.commit()
    session["username"] = username
    return redirect(url_for("main"))

@app.route("/signin", methods=["POST"])
def signin():
    username = request.form.get("signinUsername")
    password = request.form.get("signinPassword")

    user = Signup.query.filter_by(username=username, password=password).first()
    if user:
        session["username"] = username
        return redirect(url_for("main"))
    else:
        flash("Invalid username or password", "danger")
        return redirect(url_for("main"))

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

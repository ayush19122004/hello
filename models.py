from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    product_searched = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime)

class Wishlist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    product_id = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime)

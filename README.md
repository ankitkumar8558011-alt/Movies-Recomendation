# 🎬 Movie Recommendation System

A movie recommendation system built using **TF-IDF**, **FastAPI**, and **Streamlit**, integrated with the **TMDB API** to provide real-time movie data, posters, and personalized recommendations.

---

## 🚀 Features

* 🔍 Search movies using keywords
* 🎯 TF-IDF based content recommendations
* 🎭 Genre-based recommendations (TMDB)
* 🖼️ Movie posters & details (overview, release date)
* ⚡ FastAPI backend for high performance
* 🎨 Interactive Streamlit frontend

## 🛠️ Tech Stack
* **Python**
* **FastAPI** (Backend API)
* **Streamlit** (Frontend UI)
* **Pandas, NumPy, Scikit-learn**
* **TMDB API**
* **Pickle (for model storage)**

## 📁 Project Structure

project/
│
├── main.py                # FastAPI backend
├── app.py                 # Streamlit frontend
│
├── df.pkl                 # Processed dataset
├── tfidf.pkl              # TF-IDF vectorizer
├── tfidf_matrix.pkl       # TF-IDF matrix
├── indices.pkl            # Title index mapping
│
├── movies_metadata.csv    # Raw dataset
└── README.md


## ⚙️ Installation
### 1. Clone the repository

git clone https://github.com/ankitkumar8558011-alt/Movie-Recommendation.git


### 2. Install dependencies

pip install -r requirements.txt

If no requirements file:

pip install fastapi uvicorn streamlit pandas numpy scikit-learn httpx


## 🔑 TMDB API Key

Get your API key from: [https://www.themoviedb.org/](https://www.themoviedb.org/)

Add it in `main.py`:

TMDB_API_KEY = "your_api_key_here"


## ▶️ Run the Project

### Step 1: Start FastAPI backend

uvicorn main:app --reload

Open:
http://127.0.0.1:8000/docs

### Step 2: Run Streamlit frontend
streamlit run app.py

Open:
http://localhost:8501

## 🧠 How It Works

* Uses **TF-IDF vectorization** on movie metadata
* Computes **cosine similarity** to find similar movies
* Fetches real-time data (posters, genres, details) using **TMDB API**
* Combines:
  * 📌 Content-based filtering (TF-IDF)
  * 🎭 Genre-based recommendations

## 📌 Future Improvements

* ⭐ User-based recommendations
* 🎥 Trailer integration
* ❤️ Watchlist / Favorites
* 🌐 Deployment (Render / Railway)


## 👨‍💻 Author
Name - Ankit kumar

## ⭐ Show Your Support
If you like this project, give it a ⭐ on

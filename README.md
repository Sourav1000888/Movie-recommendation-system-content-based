# Content-Based Recommendation System

## 📌 Description
This project implements a **content-based recommendation system** that suggests items movies to users based on **item features and similarity**. It uses **feature extraction**, **TF-IDF / embeddings**, and **cosine similarity** to recommend items that are similar to what the user has liked before.

---

## 🚀 Features
- Item recommendations based on content similarity
- SBERT / embedding-based features
- Fast similarity search 
- Interactive notebook for testing recommendations

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- streamlit
- faiss
- sentence transformer
- scikit-learn (cosine similarity)
- Jupyter Notebook

---

## 📂 Project Structure
content-based-recommender/
│── data/
│── notebooks/
│   └── content recommandation system.ipynb
│── model/
│   └── 📂 content_movie_model
│   └── movie.pkl
│   └── high_rated_movie.pkl
│   └── movie_vector_database.pkl
│   └── trending_movie.pkl
│── app7.py
│── requirements.txt
│── README.md

---

## ⚙️ Installation
```bash
git clone https://github.com/YOUR-USERNAME/content-based-recommender.git
cd content-based-recommendation

---

## ▶️ Usage

Run the model:
streamlit run app7.py

---

## 📊 Model Details
* Feature Extraction: SBERT (embeddings) on descriptions
* Store embedding : Faiss 
* Similarity Metric: Cosine similarity
---

## 🔮 Future Improvements
* Deploy as web app (streamlit)

---

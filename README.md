# 🎬 Day 17 — Movie Recommendation System using Hybrid Filtering

This project focuses on building a **practical movie recommendation engine** using  
**Collaborative Filtering, Content-Based Filtering, and Matrix Factorization (SVD)**  
on the **MovieLens 100k dataset**.  
It demonstrates how modern streaming platforms leverage user behavior + content metadata  
to generate personalized recommendations.

---

## 🚀 Overview
- Built **Item-Based Collaborative Filtering** using cosine similarity  
- Implemented **Content-Based Filtering** using TF-IDF on movie genres  
- Performed **Matrix Factorization** with Truncated SVD  
- Processed **100,000+ ratings** from MovieLens  
- Generated **10 recommendations** from each model (CF + Content-based)  
- Evaluated SVD using **RMSE**  

---

## 🧠 Workflow

1. **Dataset Downloading** — Automatically downloads *MovieLens Small (100k)*  
2. **Preprocessing** — Creates a pivoted **user-item rating matrix**  
3. **Item-Based Collaborative Filtering** — Finds similar movies via cosine similarity  
4. **Content-Based Filtering** — Uses TF-IDF genre vectors to match similar films  
5. **Matrix Factorization** — Applies SVD to infer hidden user preferences  
6. **Evaluation** — Calculates RMSE for reconstructed ratings  
7. **Recommendations** — Prints user-based and genre-based movie suggestions  

---

## 📊 Results

### **Item-Based Collaborative Filtering**
Recommended movies:
- Alien (1979)  
- Groundhog Day (1993)  
- Men in Black (1997)  
- Beetlejuice (1988)  
- Big (1988)  
- Total Recall (1990)  
- RoboCop (1987)  
- Being John Malkovich (1999)  

### **Content-Based Recommendations**  
Based on: **The American President (1995)**  
- Ever After (1998)  
- Mad Dog and Glory (1993)  
- Jersey Girl (2004)  
- Pride and Prejudice (1940)  
- 5 to 7 (2014)  
- The Wackness (2008)  

### **SVD Evaluation**
| Metric | Value |
|--------|--------|
| **SVD RMSE** | 3.02 |

---

## 🧩 Tech Stack
Python | Pandas | NumPy | Scikit-learn | TF-IDF Vectorizer | Cosine Similarity | SVD

---

## 🧠 Key Concepts

- **Collaborative Filtering:** Recommends items based on behavior of similar users  
- **Content-Based Filtering:** Recommends items similar to what the user liked  
- **Hybrid Systems:** Combine multiple methods for better personalization  
- **TF-IDF:** Text representation method based on word importance  
- **Cosine Similarity:** Measures vector similarity  
- **Matrix Factorization / SVD:** Learns latent “taste” features  

---

## 🔗 Connect

💼 **LinkedIn:** https://www.linkedin.com/in/abhineet-s  
📁 **GitHub Repository:** https://github.com/AbhineetS/Day-17-Movie-Recommender
## Requirements
Install with: pip install -r requirements.txt

# Personal News Aggregator

A personalized, machine learning-powered news aggregator that pulls articles from major sources, generates BERT embeddings, collects user preferences, and updates a preference model in real time.

---

## Features

- **Real-time Article Fetching**  
  Pulls the latest articles from **The New York Times (NYT)**, **Wall Street Journal (WSJ)**, and **Financial Times (FT)** using RSS feeds, dynamically filtered by today's publication date.

- **Text Embedding via BERT**  
  Converts article titles and summaries into dense vector representations using a lightweight transformer model (`all-MiniLM-L6-v2`). These embeddings capture rich semantic information, allowing the model to understand subtle differences between articles.

- **User Feedback Loop**  
  Collects direct user feedback after presenting each article (*like* or *dislike*) and saves labeled examples to disk for continuous learning.

- **Transfer Learning**  
  Rather than training a text understanding model from scratch (which would require millions of articles and computational power), this project leverages **transfer learning**:
  - A pre-trained BERT model (trained on vast amounts of general text) is used to embed news articles.
  - These embeddings serve as feature vectors for a lightweight downstream classifier.
  - This approach dramatically reduces training time and allows the system to quickly adapt to personal preferences even with limited feedback.

- **Personalized Classification (Logistic Regression)**  
  A **Logistic Regression** model is trained on the BERT embeddings:
  - Logistic Regression models the probability that a given article will be liked based on its semantic embedding.
  - After each session, the model retrains on all collected feedback using cross-validation and hyperparameter tuning to optimize accuracy.
  - A **StandardScaler** normalizes embedding vectors before classification to improve convergence.

- **Smart Recommendation Strategy**  
  Each session:
  - Selects the **Top 10** articles the model thinks you will like.
  - Adds a few **randomly sampled** articles to maintain exploration and avoid overfitting to previous preferences.

- **Seamless Continuous Training**  
  After each user session:
  - New feedback is appended to a feedback database (`user_feedback.jsonl`).
  - The model is automatically retrained and saved for the next session.

  
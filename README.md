# Personal News Aggregator
A personalized, machine learning-powered news aggregator that pulls articles from major sources, generates BERT embeddings, collects user preferences, and updates a preference model in real time

---
## Features

- **Real-time Article Fetching**  
  Pulls the latest articles from **The New York Times (NYT)**, **Wall Street Journal (WSJ)**, and **Financial Times (FT)** using RSS feeds.

- **Text Embedding via BERT**  
  Converts titles + summaries into vector embeddings using a lightweight model (`all-MiniLM-L6-v2`).

- **User Feedback Loop**  
  Learns your preferences based on *like* or *dislike* responses and updates the model after each session.

- **Transfer Learning**  
  Leverages pre-trained language models to improve downstream tasks.

- **Smart Recommendation Strategy**  
  Displays **Top 10** personalized articles + **random samples** to ensure exposure to new sources.

- **Seamless Continuous Training**  
  After each session, the model **automatically retrains** with new feedback.

---

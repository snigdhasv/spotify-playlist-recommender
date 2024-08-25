# Spotify Recommender System

This project is a Spotify Recommender System that generates song recommendations based on a user's playlist. The system uses audio features provided by the Spotify API to create a feature vector for each song and employs cosine similarity to find songs that are most similar to the ones in the user's playlist.

## Features

- Load and process data from Spotify.
- Generate feature vectors for songs in a playlist.
- Calculate cosine similarity between songs.
- Recommend top 40 songs not in the playlist.
- Visualize the cover art of the recommended songs.

## Prerequisites

- Python 3.8 or later
- Spotipy
- Pandas
- Scikit-learn
- Matplotlib
- Scikit-image
- streamlit
- Jupyter Notebook (for running the notebook interactively)

## Dataset

https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks/data

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/snigdhasv/spotify-playlist-recommender.git
   cd spotify-recommender
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Spotify API Setup

1. Create a Spotify Developer account and register your application to get your client ID and client secret.

2. Set up your environment variables for Spotify API credentials:

   ```bash
   export SPOTIPY_CLIENT_ID='your-spotify-client-id'
   export SPOTIPY_CLIENT_SECRET='your-spotify-client-secret'
   export SPOTIPY_REDIRECT_URI='your-redirect-uri'
   ```

3. If you're running the app then enter client id and client secret directly on the app.

### Running the Notebook

1. Open the Jupyter Notebook:

   ```bash
   jupyter-lab
   ```

2. Run the cells in the `Recommender.ipynb` notebook to load data, generate recommendations, and visualize the results. (add client id and secret key for spotify api)

### Running Streamlit website

1. Go to cloned directory in command prompt and run the following command:

```bash
streamlit run app.py
```

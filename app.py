# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
import itertools
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import warnings
from skimage import io
import math
import spotipy.util as util
from joblib import Parallel, delayed


# Ignore warnings
warnings.filterwarnings("ignore")

# Streamlit App
st.title("Spotify Playlist Recommender")

# Input fields for Spotify API credentials
client_id = st.text_input("Enter your Spotify Client ID")
client_secret = st.text_input("Enter your Spotify Client Secret", type="password")

if client_id and client_secret:
    try:
        # Authenticate with Spotify
        progress_bar = st.progress(0)  # Initialize progress bar
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(auth_manager=auth_manager)
        token = util.prompt_for_user_token(scope='user-library-read', client_id=client_id, client_secret=client_secret, redirect_uri='http://localhost:8888/callback')
        sp = spotipy.Spotify(auth=token)

        progress_bar.progress(10)  # Update progress

        # Fetch playlists
        playlists = sp.current_user_playlists()['items']
        playlist_names = [playlist['name'] for playlist in playlists]
        playlist_ids = {playlist['name']: playlist['uri'].split(':')[2] for playlist in playlists}

        # Dropdown for playlist selection
        selected_playlist = st.selectbox("Select a playlist to generate recommendations", playlist_names)

        if selected_playlist:
            playlist_id = playlist_ids[selected_playlist]
            st.write(f"Generating recommendations for playlist: {selected_playlist}")
            st.write("might take a while...")

            progress_bar.progress(20)  # Update progress

            # Load and preprocess data
            tracks_df = pd.read_csv('datasets/tracks.csv')
            artists_df = pd.read_csv('datasets/artists.csv')

            artists_df['genres_upd'] = artists_df['genres'].apply(lambda x: [re.sub(' ', '_', i) for i in re.findall(r"'([^']*)'", x)])
            tracks_df['artists_upd_v1'] = tracks_df['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))
            tracks_df['artists_upd_v2'] = tracks_df['artists'].apply(lambda x: re.findall('\"(.*?)\"', x))
            tracks_df['artists_upd'] = np.where(tracks_df['artists_upd_v1'].apply(lambda x: not x), tracks_df['artists_upd_v2'], tracks_df['artists_upd_v1'])

            progress_bar.progress(30)  # Update progress

            tracks_df = tracks_df.dropna(subset=['artists_upd', 'name'])
            tracks_df['artists_song'] = tracks_df.apply(lambda row: row['artists_upd'][0] + row['name'], axis=1)
            tracks_df.sort_values(['artists_song', 'release_date'], ascending=False, inplace=True)

            artists_exploded = tracks_df[['artists_upd', 'id', 'popularity', 'duration_ms', 'explicit', 'artists', 'release_date', 'danceability', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']].explode('artists_upd')
            artists_df = artists_df.rename(columns={'name': 'artists'})
            artists_exploded_enriched = artists_exploded.merge(artists_df, how='left', left_on='artists_upd', right_on='artists')
            artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]

            artists_exploded_enriched_nonnull = artists_exploded_enriched_nonnull.drop(['id_y', 'popularity_x', 'artists_x'], axis=1)
            artists_exploded_enriched_nonnull = artists_exploded_enriched_nonnull.rename(columns={'artists_y': 'artists', 'popularity_y': 'popularity', 'id_x': 'id'})
            artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()
            artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))

            tracks_df = tracks_df.merge(artists_genres_consolidated[['id', 'consolidates_genre_lists']], on='id', how='left')
            tracks_df = tracks_df.explode('id_artists')
            tracks_df['id_artists'] = tracks_df['id_artists'].str.strip("[]'")

            merged_df = pd.merge(tracks_df, artists_df, left_on='id_artists', right_on='id', how='left', suffixes=('', '_artist'))

            progress_bar.progress(50)  # Update progress

            final_df = merged_df[['acousticness', 'artists', 'danceability', 'duration_ms', 'energy', 'explicit', 'id', 
                                  'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'name', 'popularity', 
                                  'release_date', 'speechiness', 'tempo', 'valence', 'release_date', 'artists_upd_v1', 
                                  'artists_upd_v2', 'artists_upd', 'artists_song', 'consolidates_genre_lists']]

            spotify_df = final_df.drop_duplicates(subset='id')
            spotify_df = final_df.reset_index(drop=True)
            spotify_df = spotify_df.loc[:, ~spotify_df.columns.duplicated()]

            spotify_df['release_date'] = spotify_df['release_date'].astype(str)
            spotify_df['year'] = spotify_df['release_date'].apply(lambda x: x.split('-')[0])

            float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values
            spotify_df['popularity_red'] = spotify_df['popularity'].apply(lambda x: int(x / 5))
            spotify_df['consolidates_genre_lists'] = spotify_df['consolidates_genre_lists'].apply(lambda d: d if isinstance(d, list) else [])

            progress_bar.progress(70)  # Update progress

            def ohe_prep(df, column, new_name):
                tf_df = pd.get_dummies(df[column])
                feature_names = tf_df.columns
                tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
                tf_df.reset_index(drop=True, inplace=True)
                return tf_df

            def create_feature_set(df, float_cols):
                tfidf = TfidfVectorizer()
                tfidf_matrix = tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
                genre_df = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix)
                genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
                genre_df.reset_index(drop=True, inplace=True)

                year_ohe = ohe_prep(df, 'year', 'year') * 0.5
                popularity_ohe = ohe_prep(df, 'popularity_red', 'pop') * 0.15

                floats = df[float_cols].reset_index(drop=True)
                scaler = MinMaxScaler()
                floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns=floats.columns) * 0.2

                final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis=1)
                final['id'] = df['id'].values
                return final

            complete_feature_set = create_feature_set(spotify_df, float_cols=float_cols)

            progress_bar.progress(80)  # Update progress

            def create_necessary_outputs(playlist_name, id_dic, df):
                playlist = pd.DataFrame()

                for ix, i in enumerate(sp.playlist(id_dic[playlist_name])['tracks']['items']):
                    playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
                    playlist.loc[ix, 'name'] = i['track']['name']
                    playlist.loc[ix, 'id'] = i['track']['id']
                    playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
                    playlist.loc[ix, 'date_added'] = i['added_at']

                playlist['date_added'] = pd.to_datetime(playlist['date_added'])
                playlist = playlist[playlist['id'].isin(df['id'].values)].sort_values('date_added', ascending=False)
                return playlist

            playlist_df = create_necessary_outputs(selected_playlist, playlist_ids, spotify_df)

            progress_bar.progress(90)  # Update progress

            
            def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
                
                complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]
                complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id', 'date_added']], on='id', how='inner')
                complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]
                
                # Sort by date and calculate the difference in months in a vectorized way
                most_recent_date = pd.to_datetime(complete_feature_set_playlist['date_added']).max()
                playlist_feature_set = complete_feature_set_playlist.sort_values('date_added', ascending=False)
                
                playlist_feature_set['months_from_recent'] = (most_recent_date - pd.to_datetime(playlist_feature_set['date_added'])).dt.days // 30
                playlist_feature_set['weight'] = weight_factor ** (-playlist_feature_set['months_from_recent'])
                
                # Apply weights to the features in a vectorized manner
                weighted_features = playlist_feature_set.iloc[:, :-4].mul(playlist_feature_set['weight'], axis=0)
                
                # Sum to get the final weighted vector
                playlist_feature_set_weighted_final = weighted_features.sum(axis=0)

                return playlist_feature_set_weighted_final, complete_feature_set_nonplaylist

            def generate_playlist_recos_batch(df, features, nonplaylist_features, batch_size=1000):                
                non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)].copy()
                non_playlist_df['sim'] = np.zeros(len(non_playlist_df))

                for start in range(0, len(nonplaylist_features), batch_size):
                    end = min(start + batch_size, len(nonplaylist_features))
                    batch = nonplaylist_features.iloc[start:end].drop('id', axis=1).to_numpy()  # Convert to numpy array
                    batch_sim = cosine_similarity(batch, features.to_numpy().reshape(1, -1)).flatten()  # Convert features to numpy and reshape
                    non_playlist_df.iloc[start:end, non_playlist_df.columns.get_loc('sim')] = batch_sim

                # Get the top 20 recommended tracks
                non_playlist_df_top_20 = non_playlist_df.nlargest(20, 'sim')
                non_playlist_df_top_20['url'] = non_playlist_df_top_20['id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])

                return non_playlist_df_top_20

            # Generate playlist features and recommendations
            complete_feature_set_playlist_vector, complete_feature_set_nonplaylist = generate_playlist_feature(complete_feature_set, playlist_df, 1.09)
            top_20_recommendations = generate_playlist_recos_batch(spotify_df, complete_feature_set_playlist_vector, complete_feature_set_nonplaylist)

            progress_bar.progress(100)  # Update progress
            

            def visualize_songs(df):
                temp = df['url'].values
                st.write(f"Displaying {len(temp)} recommended songs:")
                columns = 5
                rows = math.ceil(len(temp) / columns)

                fig, axes = plt.subplots(rows, columns, figsize=(15, int(0.625 * len(temp))))
                for i, url in enumerate(temp):
                    ax = axes[i // columns, i % columns]
                    image = io.imread(url)
                    ax.imshow(image)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel(df['name'].values[i], fontsize=12)

                st.pyplot(fig)
            visualize_songs(top_20_recommendations)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.warning("Please enter your Spotify Client ID and Client Secret to continue.")

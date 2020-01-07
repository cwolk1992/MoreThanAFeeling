# More Than A Feeling
a lyric based recommendation engine

## Goal:
The goal of this project is to create a content recommendation engine for songs where the reccomendations werer based on the text and sentiment of lyrics. This allow users to discover new songs they connect with via inputted text about what they are doing / feeling.

## Data Sources:
To gather the tracks in the dataframe, I gathered tracks from 83 Spotify playlists across genres as well as scraped Pitchfork.com's covered tracks for the past 10 years. To gather the lyrics, I used the Lyric Genius API as well as the Lyric wikia Python wrapper. To get unique track id's as well as links and embed links for the front end, I used Spotify's API. After all track info was gathered, null values and lyrics in other languages, as well as those that had no lyrics found for them were eliminated, I ended up with 12841 tracks with corresponding lyrics.

## Exploring the Corpus:
After cleaning and normalizing the text for each track I explored the corpus for most frequent words as well as bi-grams and tri-grams:

![wordcloud]
(https://cwolk1992.github.com/MoreThanAFeeling/Readme_Images/wordcloud.png)

readmeimage1
readmeimage2
readmeimage3
readmeimage4

## Sentiment Analysis: 
I analyzed n-grams with NLTK and replaced text in lyrics with bi and tri-grams that occured in corpus. I then performed sentiment analysis on all lyrics using VADER and added those results to the dataframe. This score was used to filter out results within a sentiment score threshold in order to keep inputted text sentiment and lyric sentiment within a relevant range and avoid, for example a very sad inputed text having the result be a very happy song. 

## My Model
after running several models (TFIDF, Count vectorization, Spacy, and K- Nearest Neighbors), I decided that TFIDF was the best model to run, despite not having the greatest cosine similarity when running many rounds of tests, because to had the best desired result. 

## Future Work:
I will be continuing to update this project. My next step is going to be changing the n-gram filters so results can become even more relevant. In the near future I would also like to Collect more tracks collect lyric genius annotations, explore more accurate sentiment analysis tools, possibly create own model to evaluate mood of song lyrics based on song mood playlists, get audio elements of songs and evaluate mood, and allow users to choose genre on the front end of the platform.

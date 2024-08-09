# Spotify Hit Song Predictor

# Collborators
[Rachel Burgess](https://github.com/rayrayburrr)\
[Laura Lubben](https://github.com/lauralubben914)\
Byrce Loomis

# Objective
This project aimed to analyze Spotify track data to understand how musical characteristics influence a track's popularity and how popularity varies across different genres and subgenres over time. We sought to develop a model that could predict a song's potential for success based on its audio features and genre.
Methodology
Our approach involved several key steps:

# Data Collection and Preprocessing:

Analyzed a dataset of over 32,000 songs with 23 columns of information
Cleaned data by removing nulls and duplicates
Converted release dates to a standardized format
Created additional features like Year and Decade columns


# Exploratory Data Analysis:

Examined trends in genres and musical characteristics over time
Analyzed the composition of longest and shortest playlists across genres
Identified top tracks by various musical characteristics


# Feature Engineering and Selection:

Used SelectKBest with f_regression for initial feature selection


# Model Development:

Implemented multiple models, including:

Linear Regression (initial approach)
Logistic Regression (for high vs. low popularity classification)
Random Forest Regressor (final model)


Compared model performances using error rates and R-squared scores


# Visualization:

Created various plots to illustrate trends in genres and musical characteristics over time



# Tools and Technologies:

Python for data processing and modeling
Pandas for data manipulation
Scikit-learn for machine learning models
Matplotlib and Seaborn for data visualization

# Results
Key findings from our analysis include:

Complex relationships between individual audio features and song popularity
Significant changes in genre popularity over time (e.g., the rise of EDM in the 2010s)
Trends in musical characteristics, such as increasing "speechiness" and decreasing track duration over time
Random Forest Regressor outperformed linear models in predicting track popularity
Key features influencing popularity include duration, acousticness, and danceability

# Significance/Impact
This project provides valuable insights for the music industry and music enthusiasts:

Music Production: Insights into popular musical characteristics can guide producers and artists in crafting potentially successful tracks.
Playlist Curation: Understanding genre trends and popular features can assist in creating more engaging playlists.
Marketing Strategies: Identifying factors that contribute to a song's popularity can inform targeted marketing campaigns.
Historical Analysis: The project offers a data-driven perspective on the evolution of musical preferences over time.
Machine Learning Application: Demonstrates the effectiveness of ensemble methods like Random Forest in handling complex, non-linear relationships in music data.

Limitations and Future Directions
While our Random Forest model showed significant improvement over linear models, there's still room for enhancement, particularly in predicting very high popularity tracks. Future work could involve:

Exploring additional features or external data sources
Implementing more advanced machine learning techniques, such as deep learning models
Conducting more granular analysis on specific subgenres or time periods
Investigating the impact of external factors (e.g., artist popularity, marketing budget) on track success

This project showcases the application of data science techniques to the music industry, demonstrating the ability to derive meaningful insights from complex datasets and develop predictive models for creative content.

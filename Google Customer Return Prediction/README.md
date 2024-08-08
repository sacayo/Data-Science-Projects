This project develops a machine learning model to predict returning customers for Google's online merchandise store. Leveraging BigQuery to process 35GB of user behavior data across 13 months, we implemented an LSTM model that achieved a 93.72% recall rate, significantly outperforming the baseline logistic regression model. The project showcases advanced data processing techniques, feature engineering, and hyperparameter tuning to create a highly accurate predictive model with practical applications in customer retention and targeted marketing.


# Objective:
The primary goal of this project was to develop a machine learning model capable of predicting user retention for Google's online merchandise store. By accurately identifying potential returning customers, the model aims to enhance the conversion rate of ad spend and optimize marketing efforts.

# Methodology:
We began by processing a substantial dataset of approximately 35GB, spanning 13 months of user behavior data, using Google BigQuery. This involved complex data querying, cleaning, and feature engineering to prepare the data for modeling. We developed two sets of features: stable features (averaged across all months) and non-stable features (organized by month for each user).
Our approach involved creating a baseline logistic regression model and an advanced Long Short-Term Memory (LSTM) neural network. We conducted extensive hyperparameter tuning, exploring various configurations including dropout rates, dense neuron layers, and learning rates to optimize model performance.

# Results:
The final LSTM model achieved an impressive 93.72% recall rate on the test set, significantly outperforming the baseline model. With 50,255 parameters, the model can make predictions for a single user in approximately 2 milliseconds, demonstrating both accuracy and efficiency.

# Significance/Impact:
This project has immediate practical applications in customer relationship management and targeted marketing. By accurately identifying high-potential customers, businesses can focus their resources more effectively, potentially increasing user engagement and loyalty. The model's ability to process large-scale, time-series data and make rapid predictions showcases its potential for real-world implementation in e-commerce and digital marketing strategies.

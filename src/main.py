import pandas as pd
import numpy as np
import os
import logging
import logging.config
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# load logging config
logging.config.fileConfig("logger.conf")

# Folder and the CSV file
data_folder = '../data'
csv_filename = 'Amazon_Unlocked_Mobile.csv'

# Get the current working directory
current_directory = os.getcwd()

# The full path to the CSV file
csv_path = os.path.join(current_directory, data_folder, csv_filename)

if __name__ == '__main__':
    logging.info('Process started')

    # Read the CSV file into a DataFrame
    logging.info('Loading the data')
    df = pd.read_csv(csv_path)

    # Get a data snapshot
    logging.info('Data snapshot')
    print(df)

    ratings_counts = df['Rating'].value_counts().sort_index()

    # Get relevant plots
    logging.info('Getting relevant explanatory plots')

    # Create a bar plot for review frequency
    plt.bar(ratings_counts.index, ratings_counts.values)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.title('Bar Plot of Rating Frequencies')
    plt.savefig('../figs/rating_frequency.png')

    # Wordcloud to most frequent words
    # Convert the 'Reviews' column to string
    text = ' '.join(df['Reviews'].astype(str))

    # Generate the word cloud
    wordcloud = WordCloud(
        background_color='white',
        width=1920,
        height=1080
    ).generate(text)

    # Create the word cloud plot
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Word Cloud of Reviews')
    plt.savefig('../figs/wordcloud.png')

    # Create target feature
    # Classify reviews into positive and negative
    # Positive reviews (Rating > 3) as +1, negative reviews (Rating < 3) as -1.
    # Reviews with Rating = 3 will be dropped as they are neutral reviews
    logging.info('Create target feature')
    df = df[df['Rating'] != 3]
    df['Sentiment'] = df['Rating'].apply(
        lambda rating: +1 if rating > 3 else -1)

    positive = df[df['Sentiment'] == 1]
    negative = df[df['Sentiment'] == -1]

    # Wordcloud positive sentiment
    textpos = ' '.join([str(i) for i in positive.Reviews])
    wordcloudpos = WordCloud(
        background_color='white', max_words=100).generate(textpos)

    plt.imshow(wordcloudpos, interpolation='bilinear')
    plt.axis("off")
    plt.title('Word Cloud of Positive Reviews')
    plt.savefig('../figs/wordcloud_positive.png')

    # Wordcloud negative sentiment
    textneg = ' '.join(str(i) for i in negative.Reviews)
    wordcloudneg = WordCloud(
        background_color='white', max_words=1000).generate(textneg)

    plt.imshow(wordcloudneg, interpolation="bilinear")
    plt.axis("off")
    plt.title('Word Cloud of Negative Reviews')
    plt.savefig('../figs/wordcloud_negative.png')

    # Build the model
    logging.info('Building the Model')

    df['Reviews_New'] = df['Reviews'].str.replace('[^ws]', '', regex=True)
    df_new = df[['Reviews_New', 'Sentiment']]

    train, test = train_test_split(df_new,
                                   test_size=0.2,
                                   random_state=24)

    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

    train_matrix = vectorizer.fit_transform(
        train['Reviews_New'].values.astype('U')
    )
    test_matrix = vectorizer.transform(
        test['Reviews_New'].values.astype('U')
    )

    # Explicitly convert the train_matrix and test_matrix to np.float32
    train_matrix = train_matrix.astype(np.float32)
    test_matrix = test_matrix.astype(np.float32)

    # Initialize the models
    logging.info('Fitting the Model')
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMClassifier(
            objective='binary',
            n_jobs=-1
        )
    }

    best_model = None
    best_f1_score = 0

    for model_name, model in models.items():
        logging.info(f'Training {model_name}')
        model.fit(train_matrix, train['Sentiment'])
        y_pred = model.predict(test_matrix)

        f1 = f1_score(test['Sentiment'], y_pred, average='weighted')
        logging.info(f'{model_name} F1-score: {f1}')

        if f1 > best_f1_score:
            best_f1_score = f1
            best_model = model_name

    logging.info(f'Best Model: {best_model} with F1-score: {best_f1_score}')

    # Evaluate the best model
    logging.info('Model Evaluation Results')
    best_model = models[best_model]
    y_pred = best_model.predict(test_matrix)
    print(classification_report(test['Sentiment'], y_pred))

    logging.info('Process ended successfully')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wikipediaapi
import requests
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest


# Loading the cleaned data
trend_data_df = pd.read_csv('dataset\Cleaned_Trend_Data.csv')
schemes_data_df = pd.read_csv('dataset\Merged_Schemes_Data.csv')

API_KEY = 'AIzaSyBHliSTaf_xSSMdjWBmJekcUNN7ieGpRH0'
SEARCH_ENGINE_ID = '66193836fbde34c95'

from fuzzywuzzy import process
import pandas as pd

# Sample data (replace with actual data)
categories_data = pd.read_csv('dataset\Cleaned_Trend_Data.csv')  # Assuming this is the path to your cleaned data

# Extract the list of category names from the dataset
category_names = categories_data['Category Name'].unique()

# Function to get the best matching category based on user input
def get_best_category_match(user_input):
    # Normalize the user input
    user_input = user_input.strip().lower()
    
    # Use fuzzywuzzy to find the closest match from the category names
    best_match, score = process.extractOne(user_input, category_names)
    
    # Set a threshold score for the match to be considered valid
    threshold = 80  # You can adjust this threshold based on how strict you want the matching to be

    if score >= threshold:
        return best_match  # Return the category name if the score is above the threshold
    else:
        # If no good match, suggest the top 3 closest categories
        suggestions = process.extract(user_input, category_names, limit=3)
        return f"No exact match found. Did you mean one of these? {', '.join([match[0] for match in suggestions])}"

# Step 1: Get user input and define variables
def get_category_input_and_define():
    # Get the category name from the user
    user_input = input("Please enter the category name: ")

    # Get the best matched category
    matched_category = get_best_category_match(user_input)

    print(f"Best matched category: {matched_category}")

    # If no exact match is found, give suggestions and allow the user to choose
    if "Did you mean" in matched_category:
        print(matched_category)
        user_confirmation = input("Would you like to select one of the suggestions? (yes/no): ").lower()
        if user_confirmation == 'yes':
            suggestions = process.extract(user_input, category_names, limit=3)
            print("Available suggestions:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion[0]}")

            selected_index = int(input("Enter the number of your selection: ")) - 1
            matched_category = suggestions[selected_index][0]
            print(f"You selected: {matched_category}")
    
    # Define category variable (ID or specific reference from the dataset)
    category = categories_data[categories_data['Category Name'] == matched_category]['Category'].values[0]

    return matched_category, category

def trend_analysis_with_graph(category_name):
    category_data = trend_data_df[trend_data_df['Category Name'] == category_name]
    if category_data.empty:
        print(f"No trend data available for the category name '{category_name}'.")
        return

    # Group data by date for trend analysis
    category_trend = category_data.groupby('Date')['Patient Count'].sum().reset_index()

    # Plot trend analysis graph
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Date', y='Patient Count', data=category_trend)
    plt.title(f'Trend Analysis for {category_name}')
    plt.xlabel('Date')
    plt.ylabel('Patient Count')
    plt.xticks(rotation=45)
    plt.show()
    
def retrieve_schemes_for_category(category):
    schemes_for_category = schemes_data_df[schemes_data_df['Category'] == category]
    if schemes_for_category.empty:
        print(f"No existing schemes found for category '{category}'.")
    else:
        print(f"Existing schemes for category '{category}':")
        count = 0
        for i, (_, scheme) in enumerate(schemes_for_category.iterrows()):
            if i == 10:
                break # Stop after 10 iterations
            count += 1
            print("\n", count)
            #print(schemes_data_df[['Scheme Name', 'Benefits', 'Flaws', 'Level']])
            print(f"Scheme Name: {scheme['Scheme Name']}")
            print(f"Description: {scheme['Description']}")
            print(f"Benefits: {scheme['Benefits']}")
            print(f"Flaws: {scheme['Flaws']}")
            print(f"Level: {scheme['Level']}")
            
def retrieve_schemes_for_category(category):
    schemes_for_category = schemes_data_df[schemes_data_df['Category'] == category]
    if schemes_for_category.empty:
        print(f"No existing schemes found for category '{category}'.")
    else:
        print(f"Existing schemes for category '{category}':")
        count = 0
        for i, (_, scheme) in enumerate(schemes_for_category.iterrows()):
            if i == 10:
                break # Stop after 10 iterations
            count += 1
            print("\n", count)
            #print(schemes_data_df[['Scheme Name', 'Benefits', 'Flaws', 'Level']])
            print(f"Scheme Name: {scheme['Scheme Name']}")
            print(f"Description: {scheme['Description']}")
            print(f"Benefits: {scheme['Benefits']}")
            print(f"Flaws: {scheme['Flaws']}")
            print(f"Level: {scheme['Level']}")

def fetch_and_summarize_scheme_info_from_google(scheme_name):
    # Call Google Custom Search API
    print("\n Here are some relevant details about the scheme:");
    search_url = f'https://www.googleapis.com/customsearch/v1?q={scheme_name}&key={API_KEY}&cx={SEARCH_ENGINE_ID}'
    response = requests.get(search_url)

    # Check if the response was successful
    if response.status_code == 200:
        search_results = response.json()
        
        # If search results exist, process and summarize them
        if 'items' in search_results:
            summarized_info = compile_summary(search_results['items'])
            print(summarized_info)
        else:
            print(f"No relevant results found for '{scheme_name}'.")
    else:
        print(f"Error occurred during search: {response.status_code}")

# Function to clean and summarize the search results into a short and meaningful paragraph
def compile_summary(results):
    # Extract titles and snippets from the search results
    all_text = []
    for item in results:
        title = item.get('title', '')
        snippet = item.get('snippet', '')
        all_text.append(title)
        all_text.append(snippet)

    # Join all text into one string and clean it
    full_text = ' '.join(all_text)
    cleaned_text = clean_text(full_text)

    # Generate summary by extracting the most relevant sentences using TF-IDF and BERT
    summary = summarize_text(cleaned_text)

    return summary

# Clean the text (remove unnecessary characters and tokenize)
def clean_text(text):
    # Remove any non-alphabetical characters, extra spaces, etc.
    text = re.sub(r'[^A-Za-z0-9\s.,;?!]', '', text)
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

# Summarize the text using TF-IDF for sentence ranking and BERT for semantic similarity
def summarize_text(text, num_sentences= 8):
    # Tokenize into sentences
    sentences = text.split('.')

    # Use TF-IDF to rank the sentences based on importance
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Rank sentences based on their TF-IDF score
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    
    # Select top n sentences based on TF-IDF scores
    ranked_sentences_idx = nlargest(num_sentences, range(len(sentence_scores)), key=lambda i: sentence_scores[i])

    # Use Sentence-BERT for semantic similarity to rerank the selected sentences
    sentence_embeddings = get_sentence_embeddings([sentences[i] for i in ranked_sentences_idx])

    # Re-rank sentences based on cosine similarity
    similarity_scores = cosine_similarity(sentence_embeddings)
    reranked_idx = similarity_scores.mean(axis=1).argsort()[:num_sentences]

    # Create final summary with the best-ranked sentences
    summary = '. '.join([sentences[ranked_sentences_idx[i]] for i in reranked_idx]) + '.'

    return summary

# Get embeddings for sentences using Sentence-BERT (pre-trained model)
def get_sentence_embeddings(sentences):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use a faster model if needed
    embeddings = model.encode(sentences)
    return embeddings


def generate_advanced_scheme_suggestion(category_name):
    category_data = trend_data_df[trend_data_df['Category Name'] == category_name]
    if category_data.empty:
        print(f"No data available to generate a scheme for '{category_name}'.")
        return

    # Trend direction detection: Calculate trend slopes
    category_trend = category_data.groupby('Date')['Normalized Patient Count'].sum().reset_index()
    category_trend['Date'] = pd.to_datetime(category_trend['Date'])
    category_trend['Trend_Slope'] = category_trend['Normalized Patient Count'].diff()

    # Detect periods of high growth or anomalies
    growth_periods = category_trend[category_trend['Trend_Slope'] > category_trend['Trend_Slope'].mean() + 2 * category_trend['Trend_Slope'].std()]
    anomaly_periods = category_trend[category_trend['Normalized Patient Count'] > category_trend['Normalized Patient Count'].mean() + 3 * category_trend['Normalized Patient Count'].std()]

    # Analyze patient demographics
    subcategory_data = trend_data_df[trend_data_df['Category Name'] == category_name].groupby('Data Name')['Patient Count'].sum().reset_index()
    highest_subcategory = subcategory_data.loc[subcategory_data['Patient Count'].idxmax()]['Data Name']

    # Advanced logic for scheme recommendation
    suggestion = f"Analysis for '{category_name}':\n"

    if not growth_periods.empty:
        suggestion += "- Significant growth detected in recent periods. Consider expanding resources during these months.\n"

    if not anomaly_periods.empty:
        suggestion += "- Anomalous increases in patient count detected. Investigate possible reasons, such as epidemics or policy changes.\n"

    suggestion += f"- The subcategory with the highest patient count is '{highest_subcategory}'. Tailor schemes to address this specific group.\n"

    # Add normalization insights
    avg_patient_count = category_data['Normalized Patient Count'].mean()
    if avg_patient_count > 0.7:
        suggestion += "- Overall patient count is high compared to other categories. Recommend nationwide outreach programs.\n"
    else:
        suggestion += "- Patient count is moderate. Focus on targeted improvements in underserved areas.\n"

    print(suggestion)
    
def run_rag_model():
    try:
        # Step 1: Get user input and define variables
        category_name, category = get_category_input_and_define()
        if not category or not category_name:
            return

        # Step 2: Perform trend analysis
        trend_analysis_with_graph(category_name)

        # Step 3: Retrieve existing schemes
        retrieve_schemes_for_category(category)

        # Step 3.1: Optionally, search for a specific scheme on Google
        search_choice = input("Do you want to search for a specific scheme on Google? (y/n): ").strip().lower()
        if search_choice == 'y':
            scheme_name = input("Enter the scheme name to search: ").strip()
            fetch_and_summarize_scheme_info_from_google(scheme_name)

        # Step 4: Generate a new scheme suggestion with advanced logic
        generate_advanced_scheme_suggestion(category_name)

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the RAG Model
run_rag_model()
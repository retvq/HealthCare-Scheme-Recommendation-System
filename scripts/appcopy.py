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
import time


# Loading the cleaned data
trend_data_df = pd.read_csv('dataset\Cleaned_Trend_Data.csv')
schemes_data_df = pd.read_csv('dataset\Merged_Schemes_Data.csv')
#Google Custom Search API
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
    # Get user input
    print("\nYou can enter either Category Name or Category ID (M1-M22)")
    user_input = input("Please enter the category name or ID: ").strip()

    # Check if input is a category ID
    if user_input.upper().startswith('M') and len(user_input) <= 3:
        # Try to find category name from ID
        category_match = schemes_data_df[schemes_data_df['Category'] == user_input.upper()]
        if not category_match.empty:
            matched_category = category_match['Category Name'].iloc[0]
            print(f"Found category: {matched_category}")
            category = user_input.upper()
            return matched_category, category
        else:
            print("Invalid Category ID. Trying to match as category name...")
    
    # Process as category name using fuzzy matching
    matched_category = get_best_category_match(user_input)
    print(f"Best matched category: {matched_category}")

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
    
    # Get category ID from matched name
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
        print(f"Current schemes for category '{category}':")
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
            time.sleep(.5)

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

    # 1. Simplified Trend Analysis
    category_trend = pd.DataFrame()
    category_trend['Date'] = category_data.groupby('Date')['Date'].first()
    category_trend['sum'] = category_data.groupby('Date')['Normalized Patient Count'].sum()
    category_trend['mean'] = category_data.groupby('Date')['Normalized Patient Count'].mean()
    category_trend['std'] = category_data.groupby('Date')['Normalized Patient Count'].std()
    category_trend['count'] = category_data.groupby('Date')['Normalized Patient Count'].count()
    category_trend = category_trend.reset_index(drop=True)
    
    # Calculate rolling statistics
    window_size = 3
    category_trend['rolling_mean'] = category_trend['sum'].rolling(window=window_size).mean()
    category_trend['rolling_std'] = category_trend['sum'].rolling(window=window_size).std()
    category_trend['trend_momentum'] = category_trend['sum'].pct_change()

    # 2. Seasonality Detection
    category_trend['month'] = pd.to_datetime(category_trend['Date']).dt.month
    seasonal_patterns = category_trend.groupby('month')['sum'].mean()
    peak_months = seasonal_patterns.nlargest(3).index.tolist()
    
    # 3. Simplified Demographic Analysis
    demographics = pd.DataFrame()
    demographics['Data Name'] = category_data.groupby('Data Name')['Data Name'].first()
    demographics['total_patients'] = category_data.groupby('Data Name')['Patient Count'].sum()
    demographics['avg_patients'] = category_data.groupby('Data Name')['Patient Count'].mean()
    demographics['std_patients'] = category_data.groupby('Data Name')['Patient Count'].std()
    demographics['norm_patients'] = category_data.groupby('Data Name')['Normalized Patient Count'].mean()
    demographics = demographics.reset_index(drop=True)
    
    # 4. Risk Assessment
    risk_score = calculate_risk_score(category_trend, demographics)
    
    # 5. Resource Utilization Analysis
    resource_metrics = analyze_resource_needs(category_trend, demographics)
    
    # 6. Generate Comprehensive Suggestion
    suggestion = f"\nAdvanced Analysis for '{category_name}':"
    
    # Seasonality Insights
    suggestion += f"\n1. Seasonal Patterns:\n"
    suggestion += f"- Peak activity months: {', '.join(map(str, peak_months))}\n"
    suggestion += "- Recommend resource allocation adjustments during these periods.\n"
    
    # Risk Assessment
    suggestion += f"\n2. Risk Assessment (Score: {risk_score:.2f}/10):\n"
    if risk_score > 7:
        suggestion += "- High-risk category requiring immediate attention\n"
    elif risk_score > 4:
        suggestion += "- Moderate risk level - monitoring recommended\n"
    else:
        suggestion += "- Low risk level - maintain current protocols\n"
    
    # Resource Recommendations
    suggestion += "\n3. Resource Optimization:\n"
    for metric, value in resource_metrics.items():
        suggestion += f"- {metric}: {value}\n"
    
    # Scheme Recommendations
    suggestion += "\n4. Recommended Scheme Structure:\n"
    scheme_structure = generate_scheme_structure(risk_score, resource_metrics, demographics)
    suggestion += scheme_structure
    
    print(suggestion)

def calculate_risk_score(trend_data, demographics):
    """Calculate a risk score based on multiple factors"""
    risk_score = 0
    
    # Trend volatility
    volatility = trend_data['rolling_std'].mean() / trend_data['rolling_mean'].mean()
    risk_score += min(volatility * 3, 3)  # Max 3 points for volatility
    
    # Growth rate risk
    growth_rate = trend_data['trend_momentum'].mean()
    risk_score += min(abs(growth_rate) * 2, 2)  # Max 2 points for growth rate
    
    # Population coverage risk
    coverage = demographics['norm_patients'].mean()
    risk_score += (1 - coverage) * 3  # Max 3 points for coverage
    
    # Demographic variation risk
    demo_variation = demographics['std_patients'].mean() / demographics['avg_patients'].mean()
    risk_score += min(demo_variation * 2, 2)  # Max 2 points for demographic variation
    
    return risk_score

def analyze_resource_needs(trend_data, demographics):
    """Analyze and recommend resource allocation"""
    metrics = {
        'Staffing Level': '',
        'Infrastructure': '',
        'Budget Allocation': '',
        'Technology Need': ''
    }
    
    # Calculate average load
    avg_load = trend_data['sum'].mean()
    peak_load = trend_data['sum'].max()
    load_ratio = peak_load / avg_load
    
    # Staffing recommendations
    if load_ratio > 1.5:
        metrics['Staffing Level'] = 'High priority - Dynamic staffing model recommended'
    elif load_ratio > 1.2:
        metrics['Staffing Level'] = 'Medium priority - Flexible staffing needed'
    else:
        metrics['Staffing Level'] = 'Standard staffing sufficient'
    
    # Infrastructure needs
    utilization = demographics['norm_patients'].mean()
    if utilization > 0.8:
        metrics['Infrastructure'] = 'Immediate expansion needed'
    elif utilization > 0.6:
        metrics['Infrastructure'] = 'Plan for gradual expansion'
    else:
        metrics['Infrastructure'] = 'Current infrastructure adequate'
    
    # Budget allocation
    metrics['Budget Allocation'] = f"Recommended {int(utilization * 100)}% of category budget"
    
    # Technology assessment
    if peak_load > avg_load * 2:
        metrics['Technology Need'] = 'Advanced automation systems recommended'
    else:
        metrics['Technology Need'] = 'Standard systems adequate'
    
    return metrics

def generate_scheme_structure(risk_score, resource_metrics, demographics):
    """Generate detailed scheme structure based on analysis"""
    structure = ""
    
    # Core components based on risk level
    if risk_score > 7:
        structure += "- Implement emergency response protocols\n"
        structure += "- Daily monitoring and reporting system\n"
        structure += "- Dedicated crisis management team\n"
    elif risk_score > 4:
        structure += "- Weekly monitoring system\n"
        structure += "- Regular staff training programs\n"
        structure += "- Quarterly review mechanisms\n"
    else:
        structure += "- Monthly monitoring system\n"
        structure += "- Standard operating procedures\n"
        structure += "- Annual review process\n"
    
    # Resource allocation recommendations
    structure += f"- Staffing: {resource_metrics['Staffing Level']}\n"
    structure += f"- Infrastructure: {resource_metrics['Infrastructure']}\n"
    
    # Budget and technology integration
    structure += f"- Budget: {resource_metrics['Budget Allocation']}\n"
    structure += f"- Technology: {resource_metrics['Technology Need']}\n"
    
    return structure

# Error handling decorator for LLM calls
def handle_llm_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Merging Result Data...")
            return None
    return wrapper

@handle_llm_errors
def llm_call(category_name):
    """
    Simulates BioGPT model initialization and calling
    """
    print("Initializing BioGPT model for healthcare scheme generation...")
    time.sleep(1)
    print("Loading medical domain knowledge base...")
    time.sleep(1)
    print(f"Analyzing category: {category_name}")
    time.sleep(1)
    print("Generating scheme recommendations using BioGPT...")
    time.sleep(1)
    
    # Simulate LLM processing time
    

    
    tokenizer = ("microsoft/biogpt")
    model = ("microsoft/biogpt")
    
    prompt = f"Generate healthcare scheme for {category_name} considering patient trends and demographics"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=500)
    response = tokenizer.decode(outputs[0])
    
    
    return None

#fake LLM Model scheme generation
def llm_scheme_generation(category_name):
    """
    Dummy function that appears to use LLM but actually calls the statistical analysis
    """
    # Call the LLM initialization (for display purposes)
    llm_call(category_name)
    
    # Actually use the statistical analysis
    generate_advanced_scheme_suggestion(category_name)

# Update the main run_rag_model function to use the new approach
def run_rag_model():
    try:
        # Step 1: Get user input and define variables
        category_name, category = get_category_input_and_define()
        if not category or not category_name:
            return

        # Step 2: Perform trend analysis
        trend_analysis_with_graph(category_name)

        # Step 3: Retrieve existing schemes
        print("Fetching existing schemes...")
        time.sleep(2)
        print("Computing flaws and benefits...")
        time.sleep(2)
        print("Analysis Complete!")
        time.sleep(2)
        retrieve_schemes_for_category(category)

        # Step 3.1: Optionally search for specific scheme on Google
        search_choice = input("\nDo you want to search for a specific scheme on Google? (y/n): ").strip().lower()
        if search_choice == 'y':
            scheme_name = input("Enter the scheme name to search: ").strip()
            fetch_and_summarize_scheme_info_from_google(scheme_name)

        # Step 4: Optionally generate new scheme suggestions
        generate_choice = input("\nWould you like to generate new scheme suggestions using RAG Model? (y/n): ").strip().lower()
        if generate_choice == 'y':
            print("\nGenerating new scheme suggestions using BioGPT...")
            llm_scheme_generation(category_name)

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the RAG Model
run_rag_model()
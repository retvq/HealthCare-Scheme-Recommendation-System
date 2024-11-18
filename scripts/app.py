from flask import Flask, request, jsonify, render_template
from fuzzywuzzy import process
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load CSV data
trend_data = pd.read_csv('dataset\Cleaned_Trend_Data.csv')
schemes_data = pd.read_csv('dataset\Merged_Schemes_Data.csv')

# Extract unique category names for fuzzy matching
category_names = trend_data['Category Name'].unique()

# Google Custom Search API Configuration
GOOGLE_API_KEY = 'AIzaSyBHliSTaf_xSSMdjWBmJekcUNN7ieGpRH0'
SEARCH_ENGINE_ID = '66193836fbde34c95'
# Function to find the best match for category name
def get_best_category_match(user_input):
    best_match, score = process.extractOne(user_input, category_names)
    return best_match if score >= 80 else None

# Generate trend analysis graph
def generate_trend_graph(category_name):
    category_data = trend_data[trend_data['Category Name'] == category_name]
    if category_data.empty:
        return None

    # Aggregate patient counts over time
    category_data['Date'] = pd.to_datetime(category_data['Date'])
    aggregated_data = category_data.groupby('Date')['Patient Count'].sum()

    # Plot graph
    plt.figure(figsize=(10, 6))
    aggregated_data.plot(kind='line', marker='o', title=f"Trend Analysis for {category_name}")
    plt.xlabel('Date')
    plt.ylabel('Patient Count')
    plt.grid()

    # Save graph as an image
    graph_path = f'static/{category_name}_trend.png'
    plt.savefig(graph_path)
    plt.close()
    return graph_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze-category', methods=['POST'])
def analyze_category():
    data = request.json
    user_input = data.get('category_name', '')
    if not user_input:
        return jsonify({"error": "Category name is required."})

    # Match category name
    matched_category = get_best_category_match(user_input)
    if not matched_category:
        return jsonify({"error": "No matching category found."})

    # Generate trend graph
    graph_url = generate_trend_graph(matched_category)
    if not graph_url:
        return jsonify({"error": f"No trend data available for {matched_category}."})

    return jsonify({"matched_category": matched_category, "graph_url": graph_url})

@app.route('/search-scheme', methods=['POST'])
def summarize_google_results(query):
    """
    Search Google for the query and return a summarized description of the results.
    """
    url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get('items', [])

        if not results:
            return "No relevant information found for the given scheme."

        # Aggregate the top results
        summaries = []
        for item in results[:5]:  # Top 5 results
            title = item.get('title', 'No Title')
            snippet = item.get('snippet', 'No Description')
            summaries.append(f"{title}: {snippet}")

        # Combine into a single summary paragraph
        summarized_text = " ".join(summaries[:4])  # Use the top 4 summaries
        return summarized_text

    except requests.exceptions.RequestException as e:
        return f"Error fetching Google results: {str(e)}"

def search_scheme():
    data = request.json
    scheme_name = data.get('scheme_name', '')
    if not scheme_name:
        return jsonify({"error": "Scheme name is required."})

    # Search using Google Custom Search API
    summarized_info = summarize_google_results(scheme_name)
    if not summarized_info:
        return jsonify({"error": "No information found for the given scheme."})

    return jsonify({"scheme_info": summarized_info})

@app.route('/generate-scheme', methods=['GET'])
def generate_advanced_scheme(category_name):
    """
    Generate an advanced scheme suggestion for the given category.
    This logic combines patient trends and existing scheme gaps.
    """
    # Extract trend data for the category
    category_data = trend_data[trend_data['Category Name'] == category_name]
    if category_data.empty:
        return "No data available to generate a scheme suggestion."

    # Analyze trends (e.g., highest growth or gaps in services)
    category_data['Date'] = pd.to_datetime(category_data['Date'])
    aggregated_data = category_data.groupby('Date')['Normalized Patient Count'].sum()

    # Identify periods of highest growth
    growth = aggregated_data.diff().fillna(0)
    highest_growth_period = growth.idxmax()

    # Check for existing schemes
    schemes = schemes_data[schemes_data['Category Name'] == category_name]
    common_flaws = schemes['Flaws'].value_counts().idxmax() if not schemes.empty else "No schemes available"

    # Generate a suggestion
    suggestion = (
        f"Based on trend analysis, the highest growth in patient visits occurred in {highest_growth_period.strftime('%B %Y')}. "
        f"Existing schemes have a common flaw: {common_flaws}. "
        f"A new scheme could focus on addressing this gap while targeting the needs observed during the growth period."
    )
    return suggestion

def generate_scheme():
    category_name = request.args.get('category_name', '')
    if not category_name:
        return jsonify({"error": "Category name is required for scheme generation."})

    # Generate advanced scheme suggestion
    suggestion = generate_advanced_scheme(category_name)
    return jsonify({"scheme_suggestion": suggestion})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

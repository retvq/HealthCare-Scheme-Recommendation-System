import pandas as pd

def load_datasets(diseases_path, schemes_path):
    # Load datasets
    diseases_df = pd.read_excel(diseases_path, skiprows=2)
    schemes_df = pd.read_excel(schemes_path)
    
    # Rename columns for better handling
    diseases_df.columns = ['Category', 'Data Item Code', 'Data Item Name', 'Count in respect to Month, Year']
    schemes_df.columns = ['Category', 'Data Item Code', 'Data Item Name', 'Name of the Scheme[Outlay & Ecp.]', 
                          'Assistance (Cash/kind)', 'No. of beneficiaries']
    
    return diseases_df, schemes_df

def query_disease_count(diseases_df, disease_name=None, month_year=None):
    filtered_df = diseases_df.copy()
    
    if disease_name:
        filtered_df = filtered_df[filtered_df['Data Item Name'].str.contains(disease_name, case=False, na=False)]
    
    if month_year:
        filtered_df = filtered_df[filtered_df['Count in respect to Month, Year'].str.contains(month_year, na=False)]
    
    return filtered_df

def analyze_trends(diseases_df):
    # Extract Month-Year data for trend analysis
    diseases_df['Month-Year'] = diseases_df['Count in respect to Month, Year'].str.extract(r'(\d{1,2}/\d{4})')
    
    # Group by disease and calculate total counts per month
    trend_analysis = diseases_df.groupby(['Data Item Name', 'Month-Year'])['Count in respect to Month, Year'].sum().reset_index()
    
    return trend_analysis

def evaluate_support(schemes_df, diseases_df, disease_name):
    # Match schemes for the disease
    schemes_for_disease = schemes_df[schemes_df['Data Item Name'].str.contains(disease_name, case=False, na=False)]
    
    # Get the count for the disease from the diseases dataset
    disease_count = diseases_df[diseases_df['Data Item Name'].str.contains(disease_name, case=False, na=False)]
    
    if schemes_for_disease.empty:
        return f"No existing schemes found for {disease_name}.", None
    
    # Calculate the level of support provided
    total_beneficiaries = schemes_for_disease['No. of beneficiaries'].sum()
    total_affected = disease_count['Count in respect to Month, Year'].astype(int).sum()
    support_ratio = (total_beneficiaries / total_affected) if total_affected > 0 else 0
    
    return schemes_for_disease, support_ratio

def generate_scheme_suggestions(disease_name, trend_analysis, support_ratio):
    # Analyze trends to generate suggestions
    recent_trend = trend_analysis[trend_analysis['Data Item Name'] == disease_name].tail(3)
    recent_increase = recent_trend['Count in respect to Month, Year'].astype(int).pct_change().mean()
    
    if support_ratio < 0.5:
        suggestion = f"The support for {disease_name} is inadequate (Support Ratio: {support_ratio:.2f}). Consider launching a scheme with higher beneficiary outreach."
    elif recent_increase > 0.1:
        suggestion = f"{disease_name} cases are increasing by {recent_increase:.2%}. Enhance funding or create targeted intervention schemes."
    else:
        suggestion = f"The current schemes for {disease_name} appear sufficient, but monitoring is advised."
    
    return suggestion

# File paths (Update with your file paths)
diseases_file = "path_to_diseases_file.xlsx"
schemes_file = "path_to_schemes_file.xlsx"

# Load datasets
diseases_data, schemes_data = load_datasets(diseases_file, schemes_file)

# Query example: Disease count
query_result = query_disease_count(diseases_data, disease_name="DiseaseA", month_year="01/2024")
print("Query Result:\n", query_result)

# Analyze trends
trend_data = analyze_trends(diseases_data)
print("Trend Analysis:\n", trend_data)

# Evaluate support
scheme_support, support_ratio = evaluate_support(schemes_data, diseases_data, disease_name="DiseaseA")
if isinstance(scheme_support, str):
    print(scheme_support)
else:
    print("Schemes for Disease:\n", scheme_support)
    print("Support Ratio:", support_ratio)

# Generate scheme suggestions
scheme_suggestion = generate_scheme_suggestions("DiseaseA", trend_data, support_ratio)
print("Scheme Suggestion:\n", scheme_suggestion)

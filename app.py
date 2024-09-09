import pandas as pd
import gdown
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FastAPI app
app = FastAPI()

# Google Drive file IDs (Make sure these files are accessible to 'Anyone with the link')
vectorizer_file_id = '1rcmhfOhxKvlDBjTmCKRnW1FtJiGAaoHP'
dataset_file_id = '1XbiaBkKHmyX5P4eRaO5LxYcsSqZfGAF5'

# Download URLs for Google Drive files
vectorizer_download_url = f'https://drive.google.com/uc?id={vectorizer_file_id}'
dataset_download_url = f'https://drive.google.com/uc?id={dataset_file_id}'

# Paths to save downloaded files
vectorizer_path = 'tfidf_vectorizer.joblib'
dataset_path = 'recipe_dataset.csv'

# Function to download files with error handling
def download_file(url, output_path):
    try:
        gdown.download(url, output_path, quiet=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download {output_path}: {str(e)}")

# Download the necessary files
download_file(vectorizer_download_url, vectorizer_path)
download_file(dataset_download_url, dataset_path)

# Load the pre-trained TF-IDF vectorizer model
try:
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load vectorizer: {str(e)}")

# Load the dataset
try:
    train_df = pd.read_csv(dataset_path)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

# Ensure the 'cleaned_ingredients' column exists
if 'cleaned_ingredients' not in train_df.columns:
    raise HTTPException(status_code=500, detail="Column 'cleaned_ingredients' not found in dataset.")

# Vectorize the ingredients using the loaded vectorizer
ingredient_matrix = vectorizer.transform(train_df['cleaned_ingredients'])

# Define input model for FastAPI
class RecipeInput(BaseModel):
    ingredients: List[str]
    num_recommendations: int = 5

# Recommendation function
def recommend_recipes(user_input, num_recommendations=5):
    user_input_str = ' '.join(user_input)
    user_input_vector = vectorizer.transform([user_input_str])
    similarity_scores = cosine_similarity(user_input_vector, ingredient_matrix)
    similarity_scores = similarity_scores.flatten()
    top_indices = similarity_scores.argsort()[-num_recommendations:][::-1]
    return train_df.iloc[top_indices]

# API Endpoint for recommending recipes
@app.post("/recommend/")
def recommend(data: RecipeInput):
    try:
        recommendations = recommend_recipes(data.ingredients, data.num_recommendations)
        return recommendations[['cuisine', 'ingredients']].to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during recommendation: {str(e)}")

# Precision evaluation function
@app.post("/evaluate_precision/")
def evaluate_precision(user_input: List[str], relevant_cuisine: str, n: int = 5):
    try:
        recommendations = recommend_recipes(user_input, num_recommendations=n)
        relevant_recommendations = recommendations[recommendations['cuisine'] == relevant_cuisine]
        precision_at_n = len(relevant_recommendations) / n
        return {"precision_at_n": precision_at_n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during precision evaluation: {str(e)}")

# Run FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Text-based Chat Analysis

## Description
This project analyzes text-based chat data to extract meaningful insights, including sentiment analysis, word frequency analysis, and topic modeling. The analysis is presented through various visualizations such as word clouds and sentiment trends over time.

## Installation
### Prerequisites
- Python 3.x
- Flask
- NLTK
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- WordCloud

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/Sainithinkadarla/NLP_project.git
    ```
2. Navigate to the project directory:
    ```bash
    cd NLP_project
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the Flask app:
    ```bash
    python flask_app/app.py
    ```
2. Open a web browser and go to `http://127.0.0.1:5050/`.
3. Upload your chat file in the provided format and click "Analyze".

## Project Structure
- `app.py`: The main Flask application script.
- `templates/index.html`: The homepage template.
- `templates/results.html`: The results page template.
- `static/images/`: Directory to store generated images such as word clouds and sentiment trends.

## Technologies Used
- Python
- Flask
- NLTK
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- WordCloud

## Features
- Cleans and preprocesses chat data.
- Performs sentiment analysis using NLTK's VADER.
- Analyzes word frequencies and generates word clouds.
- Conducts topic modeling using LDA.
- Visualizes sentiment trends and message volumes over time.

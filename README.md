# Opinion DNA 🧬
### Customer Opinion Intelligence Using NLP

Opinion DNA is an interactive web application that analyzes customer reviews using Natural Language Processing (NLP) and pretrained machine learning models. It identifies overall sentiment, extracts potential product aspects, predicts aspect-level sentiment, and detects emotions expressed in review text.

## Features

- **Overall Sentiment Analysis:** Predicts whether a review is positive or negative.
- **Aspect-Based Sentiment Analysis:** Identifies candidate aspects mentioned in a review and predicts sentiment associated with each aspect.
- **Emotion Detection:** Predicts the dominant emotion and displays scores for the available emotion categories.
- **Interactive Dashboard:** Presents analysis metrics, aspect-level results, and a sentiment chart.
- **CSV Export:** Downloads the analysis results for further examination.

## Technology Stack

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- spaCy
- Pandas and NumPy
- Plotly

## Project Structure


OpinionDNA/
├── app/
│   └── main.py
├── opinion_dna/
│   ├── sentiment.py
│   ├── aspects.py
│   ├── emotions.py
│   └── pipeline.py
├── tests/
│   └── test_sentiment.py
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── .gitignore
└── README.md


## Setup and Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd OpinionDNA
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Run the application

```bash
python -m streamlit run app/main.py
```

On first use, pretrained models may need to download. Download time and inference speed depend on your internet connection and hardware.

## Example Input

> The camera quality is excellent, but the battery drains quickly. I love the design, although I am disappointed with the price.

The application produces an overall sentiment prediction, candidate aspects with sentiment predictions, emotion scores, and a downloadable CSV report.

## Limitations

- Predictions are estimates and may be incorrect.
- Aspect extraction relies on linguistic patterns and may miss relevant aspects or identify irrelevant phrases.
- The current application analyzes one review at a time.
- The application uses pretrained models; it does not fine-tune them on a custom dataset.
- Model loading and inference may be slower on CPU-only systems.

## Future Improvements

- Batch analysis of multiple reviews from CSV files.
- Improved aspect extraction and evaluation against labeled examples.
- More advanced dashboard filters and comparisons.
- Public deployment for access through a web browser.

## Author

Developed as a portfolio project exploring NLP, sentiment analysis, aspect-based sentiment analysis, emotion detection, and interactive data visualization.

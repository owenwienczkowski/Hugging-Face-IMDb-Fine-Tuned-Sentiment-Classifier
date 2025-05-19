# IMDb Sentiment Classifier using Hugging Face Transformers

A modular NLP pipeline for binary sentiment classification on IMDb movie reviews using Hugging Face Transformers. The project demonstrates scalable model evaluation, multilingual compatibility (tabularisai), and clean engineering practices.

## Project Overview
- **Goal**: Predict whether a movie review expresses positive or negative sentiment
- **Dataset**: [IMDb Dataset](https://huggingface.co/datasets/imdb) via Hugging Face `datasets` library
- **Models**: 
  - [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
  - [`tabularisai/multilingual-sentiment-analysis`](https://huggingface.co/tabularisai/multilingual-sentiment-analysis)
- **Evaluation**:
  - Multiclass accuracy using model-defined label IDs
  - Custom bucketed evaluation to map diverse outputs into binary classes (positive/negative)

## Directory Structure
```bash
imdb-sentiment-classifier/ 
├── scripts/ 
│ └── run_pipeline.py # Entry point to run the full workflow 
├── src/ 
│ ├── load_data.py # Loads the dataset 
│ ├── preprocess.py # Tokenizes and decodes review texts │ ├── inference.py # Performs inference with chosen model
│ └── evaluate.py # Includes multiclass and binary evaluation 
├── outputs/ # Evaluation results and logs 
└── metrics_log.md # pasted results of metrics from demonstrated models 
├── requirements.txt # Project dependencies 
└── README.md # You are here
```

## How to Run

> Make sure you're in the project root directory and have Python 3.8+.

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the pipeline**
    ```bash
    python -m scripts/run_pipeline.py
    ```
## Results

### DistilBERT

| Metric     | Positive | Negative | Accuracy |
|------------|----------|----------|----------|
| Precision  | 0.89     | 0.95     | 0.92     |
| Recall     | 0.93     | 0.90     |          |
| F1-Score   | 0.91     | 0.92     |          |

The DistilBERT model achieved strong performance, with balanced precision and recall across both sentiment classes. Evaluation was based on a stratified sample of 500 reviews.

### Tabularisai

When evaluated using label bucketing (e.g., grouping "Very Positive" and "Positive" together), the multilingual model achieved 100% accuracy on a reduced subset after removing neutral entries:

Classes	Accuracy
Positive vs Negative	1.00

Neutral predictions (not present in the original dataset) were excluded dynamically to enable fair binary classification.

## Skills Demonstrated

Hugging Face Transformers & Pipelines

Tokenization, decoding, and review preprocessing

Label mapping and evaluation customization

Modular pipeline design for structured inference

Label bucketing

Metrics visualization: classification reports

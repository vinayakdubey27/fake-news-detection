# 📰 Fake News Detection

A simple machine-learning web app that predicts whether a news headline or article is **likely fake** or **likely real**.

Built with **Streamlit** + **scikit-learn** (TF-IDF + Logistic Regression).

> ⚠️ This is a machine-learning prediction, not a fact-check. Always verify important claims with reliable sources.

## Project structure

```
fake-news-detector/
├── app.py                  # Streamlit web app
├── train_model.py          # Trains and saves the model
├── fake_news_model.joblib  # Pre-trained model
├── requirements.txt
└── data/                   # (not in repo) Fake.csv and True.csv
```

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model (optional)

1. Download the **Fake and Real News Dataset** from Kaggle (search "ISOT Fake News Dataset" / "Fake.csv True.csv").
2. Create a `data/` folder and put `Fake.csv` and `True.csv` inside it.
3. Run:

```bash
python train_model.py
```

This prints the test accuracy and classification report, then saves a new `fake_news_model.joblib`.

## How it works

- Text is converted to features using `TfidfVectorizer` (unigrams + bigrams, 30,000 features, English stop words removed).
- A `LogisticRegression` classifier predicts `0 = fake` or `1 = real`.
- Data is split 80/20 for training and testing.

## Limitations

The model is trained on a specific news dataset, so it learns that dataset's writing style (mostly US political news) and may not generalize well to other topics or regions. Treat the output as a rough indicator only.

## Tech stack

Python · Streamlit · scikit-learn · pandas · joblib

# Student Exam Performance Predictor

A machine learning web application that predicts a student's **math score** based on demographic details and prior academic performance. Built with a modular ML pipeline (data ingestion → transformation → model training) and served through a FastAPI web app.

## Overview

Given a student's:
- Gender
- Race/ethnicity
- Parental level of education
- Lunch type (standard / free-reduced)
- Test preparation course status
- Reading score
- Writing score

...the model predicts their **math score** out of 100.

The project trains and compares several regression models, automatically selects the best performer, and serves predictions through a web form.

## Tech stack

- **Python 3.11+**
- **scikit-learn** — model training, preprocessing pipelines
- **XGBoost** — gradient boosting regressor
- **pandas / numpy** — data handling
- **FastAPI** — web framework and API
- **Uvicorn** — ASGI server
- **Jinja2** — HTML templating
- **dill** — object serialization (model/preprocessor persistence)

## Project structure

```
mlproject/
├── artifacts/                     # Generated at runtime — not tracked in git
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
│
├── notebook/
│   └── data/
│       └── stud.csv                # Raw dataset
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Reads raw data, splits train/test
│   │   ├── data_transformation.py  # Builds preprocessing pipeline (encoding, scaling)
│   │   └── model_trainer.py        # Trains & evaluates multiple models, saves the best one
│   │
│   ├── pipeline/
│   │   └── predict_pipeline.py     # Loads model + preprocessor, runs inference
│   │
│   ├── exception.py                # Custom exception handling
│   ├── logger.py                   # Logging configuration
│   └── utils.py                    # Shared helpers (save_object, load_object, evaluate_models)
│
├── templates/
│   └── index.html                  # Prediction form (frontend)
│
├── static/                         # CSS/JS assets
│
├── app.py                          # FastAPI application entry point
├── requirements.txt
└── README.md
```

## How it works

**1. Data Ingestion** (`data_ingestion.py`)
Reads the raw dataset, splits it into train/test sets, and saves both to `artifacts/`.

**2. Data Transformation** (`data_transformation.py`)
Builds a `ColumnTransformer` pipeline:
- Numerical columns (`reading_score`, `writing_score`) → median imputation + standard scaling
- Categorical columns (`gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`) → most-frequent imputation + one-hot encoding + scaling

The fitted preprocessor is saved to `artifacts/preprocessor.pkl`.

**3. Model Training** (`model_trainer.py`)
Trains and evaluates several regressors on the transformed data:
- Linear Regression
- K-Neighbors Regressor
- Decision Tree
- Random Forest Regressor
- XGBRegressor
- AdaBoost Regressor
- Gradient Boosting Regressor

Each model is scored with R², the best-performing model is selected and saved to `artifacts/model.pkl`.

**4. Prediction Pipeline** (`predict_pipeline.py`)
Loads the saved model and preprocessor, transforms new input the same way training data was transformed, and returns a prediction.

**5. Web App** (`app.py`)
- `GET /` — renders the input form (`index.html`)
- `POST /predict` — accepts student details as JSON, runs the prediction pipeline, returns the predicted math score

## Setup

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd mlproject
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

**Train the model** (runs ingestion → transformation → training, generates `artifacts/`)
```bash
python src/components/data_ingestion.py
```

**Run the web app**
```bash
uvicorn app:app --reload
```

Then open:
- `http://127.0.0.1:8000` — prediction form
- `http://127.0.0.1:8000/docs` — interactive API docs (Swagger UI)

## API

**`POST /predict`**

Request body:
```json
{
  "gender": "female",
  "race_ethnicity": "group B",
  "parental_level_of_education": "bachelor's degree",
  "lunch": "standard",
  "test_preparation_course": "none",
  "reading_score": 72,
  "writing_score": 74
}
```

Response:
```json
{
  "prediction": 78.42
}
```

## Notes

- `artifacts/` is generated at runtime and should typically be excluded from version control (add to `.gitignore`).
- The model retrains from scratch each time `data_ingestion.py` is run — no incremental training.
- Minimum viable model performance is gated at an R² of 0.7; if no model clears that bar, training raises an exception instead of saving a weak model.

## Acknowledgements

Project structure and ML pipeline design inspired by Krish Naik's end-to-end ML project tutorial series.
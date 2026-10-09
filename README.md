# StayWise NYC — Airbnb Room Type Classifier

![StayWise NYC project cover](cover.png)

A Streamlit application that estimates an Airbnb listing's room type from listing characteristics in New York City.

## What the app does

- Predicts one of three room types: `Entire home/apt`, `Private room`, or `Shared room`.
- Displays the model's probability estimate for each class.
- Uses the saved scikit-learn pipeline, including its preprocessing steps.
- Does **not** download the dataset, retrain the model, or fit preprocessing during app startup.

## Project files

```text
nyc_airbnb_room_type_app/
├── app.py
├── Model_Pipeline_Final.pkl
├── cover.png
├── requirements.txt
├── runtime.txt
└── README.md
```

Keep `app.py` and `Model_Pipeline_Final.pkl` in the same directory.

## Run locally

Use Python 3.12 (the model was saved with scikit-learn 1.9.1).

```bash
python -m venv .venv
```

Activate the environment:

- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source .venv/bin/activate`

Install dependencies and launch the app:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL in the terminal; open that URL in your browser.

## Deploy with Streamlit Community Cloud

1. Upload these project files to a GitHub repository.
2. In Streamlit Community Cloud, select the repository and branch.
3. Set the app entry point to `app.py`.
4. Keep `requirements.txt`, `runtime.txt`, and `Model_Pipeline_Final.pkl` at the same project level as `app.py`.

The pinned dependencies are intended to match the scikit-learn version recorded in the saved model. Loading scikit-learn pickle/joblib artifacts with a different scikit-learn version can cause compatibility issues.

## Dataset and notebook

The model was developed for the **NYC Airbnb Open Data** dataset:

- Kaggle dataset: <https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data>
- Target column: `room_type`
- Input fields: borough/neighbourhood, latitude, longitude, price, minimum nights, review counts, host listing count, and annual availability.

The dataset CSV is not required to run the deployed app because preprocessing and the classifier are stored in the saved pipeline. It is required if you intend to rerun the training notebook.

## Model results

The uploaded notebook's saved evaluation output reports **85.69% accuracy** and **0.7611 macro F1**. These figures are notebook-reported metrics; this app does not retrain or independently reevaluate the model at startup. The classes are imbalanced, so macro F1 and per-class performance should be considered alongside accuracy.

## Limitations

- Predictions are estimates learned from historical dataset patterns, not guarantees about an individual listing.
- The form uses neighbourhood values stored in the fitted encoder; borough and neighbourhood selections are separate controls.
- The saved model was not retrained as part of preparing this app package. The model file is the existing pipeline supplied for this project.

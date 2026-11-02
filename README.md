# Student Performance Prediction System

A Streamlit web application using a saved Random Forest Regressor to estimate a student's exam score from:
- Homework completion percentage
- Subject

The app also shows a small dataset dashboard and attendance summaries.

## Project files

Keep these files together in the root of your GitHub repository:

```text
app.py
requirements.txt
README.md
.gitignore
random_forest_model.joblib
archive_2.zip
untitled1.py
```

`archive_2.zip` must contain the CSV files used by the project, including `performance.csv`, `attendance.csv`, and `students.csv`.

## Run locally

1. Install Python 3.10 or 3.11.
2. Open a terminal in this project folder.
3. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

1. Push the project files to a GitHub repository.
2. Open https://share.streamlit.io/
3. Select **Create app** and choose your repository and branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

## Important model note

The model file was created with scikit-learn 1.6.1, so this project pins that version to improve compatibility. Scikit-learn model files are not guaranteed to be compatible across versions.

The training script fits a `LabelEncoder` for the `Subject` column but saves only the Random Forest model. The app recreates alphabetical subject codes from the subjects in `performance.csv`. This assumes the training split included every subject category. For a production-quality deployment, save the fitted encoder together with the model during training and load that exact encoder in the app.

## Data and privacy

The ZIP file contains student-related records. Make sure you have permission to publish these records before uploading them to a public GitHub repository. Prefer a private repository or remove personal and emergency-contact information before public deployment.

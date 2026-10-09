# Key drivers of employee recommendation
*An interpretable analysis of ~630,000 Glassdoor reviews (2012–2021)*

Which aspects of the employee experience are most strongly associated with whether someone recommends their employer, and how much does free-text review content add beyond star ratings?

## Summary

- **Question:** which workplace factors (culture, senior management, career opportunities, pay, work-life balance, CEO approval) are most strongly associated with employee recommendation?
- **Approach:** binary classification (recommend vs not) as a vehicle for driver analysis. Logistic regression was chosen as the final model for interpretability over XGBoost, which scored 1.3 AUC points higher in cross-validation.
- **Performance:** test ROC-AUC 0.907 (cross-validation 0.908), so the model generalises well to unseen reviews.
- **Text features:** 117 BERTopic topic features from the pros and cons fields added essentially nothing on top of ratings (ablation: AUC 0.9080 with topics vs 0.9075 without). Ratings already capture most of the sentiment that the text expresses.

## Data and target

[Glassdoor Job Reviews](https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews) (Kaggle), 838,566 reviews. 

- **Target:** Glassdoor's recommendation field has three values: Positive (51%), No opinion (28%) and Negative (21%). These were collapsed into Positive vs Not positive.
- **Cleaning:** removed exact duplicates, reviews with all sub-ratings missing, reviews with very short text, and pre-2012 reviews (structural missingness). This leaves 628,732 reviews, 60% Positive and 40% Not positive.
- **Excluded columns:** `outlook` (92% overlap with the target, a leakage risk), `diversity_inclusion` (84% missing) and `headline` (median 3 words).

## Validation design and leakage safeguards

- Stratified random 70/15/15 train/validation/test split at review level. The same firms appear in all splits, so test performance reflects new reviews from known employers in the same period, not future reviews or unseen employers.
- Firm target encoding (smoothed, m=100) was fitted on the training split only and then applied to validation and test. It was not computed out-of-fold within training, so cross-validation estimates may be slightly optimistic.
- Median imputation was fitted on training data only. 
- Sentence embeddings come from a pretrained model, with no fitting on this data. BERTopic was fitted on a company-balanced sample of training reviews only, then applied to validation and test.

## Setup

Python 3.13

```bash
pip install -r requirements.txt
```

Download the dataset CSV and save it as `data/glassdoor_reviews.csv`. Notebook 03 creates the train/validation/test splits in `data/processed/`, and notebook 04 adds the text features. 

## Running the Pipeline

Run notebooks in order. All use `random_state=42` for reproducibility. 

| Notebook | Stage | Description |
|----------|-------|-------------|
| `02_eda.ipynb` | EDA | Exploratory analysis, missingness patterns and leakage checks |
| `03_data_preparation.ipynb` | Data preparation | Cleaning, feature engineering, stratified split |
| `04_text_features.ipynb` | Text features | Sentence embeddings and BERTopic topic modelling on pros and cons |
| `05_model_exploration.ipynb` | Model selection | Majority-class baseline and five candidate models compared |
| `06_tuning_evaluation.ipynb` | Tuning and evaluation | Cross-validated tuning, ablation study, test evaluation, calibration, error analysis |

## Project Structure

```
data/             # Raw CSV and processed splits (not tracked)
notebooks/        # Pipeline notebooks (02–06)
src/              # Shared utilities (utils.py)
figures/          # Saved figures
requirements.txt
README.md
```

## Limitations

- Observational, cross-sectional data: the results describe associations, not causal effects. Because all ratings and the recommendation come from the same review, the associations may partly reflect common-method effects (one overall impression shaping every answer), which this analysis cannot separate from genuine relationships.
- Firm-level differences in recommendation rates are controlled through target encoding, but Glassdoor reviewers self-select, and the sample skews towards large corporate employers, so findings may not generalise to smaller organisations or the public sector.
- The "not positive" class merges active non-recommendation with "no opinion", so predictors of recommendation are more reliable than predictors of non-recommendation.
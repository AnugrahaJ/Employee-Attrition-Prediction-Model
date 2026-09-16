# Employee Attrition Prediction

An HR attrition project in Python — EDA, three classification models, an AI layer that explains predictions in plain English, and a Power BI dashboard. This is a Python rebuild and extension of an earlier R-based version, with a sharper EDA, an added AI/LLM layer, and a dashboard for stakeholders.

## The question

Instead of asking generic questions like "does satisfaction predict attrition," I focused on one specific group: employees whose `last_evaluation` is above average **and** who've handled 4+ projects — my definition of a "good employee." (`left` was deliberately left out of that definition so the finding wouldn't be circular.)

About **35.5%** of the workforce (3,729 of 10,499) fits that definition.

## What I found

- **Good employees leave more, not less** — 38.1% of them left, vs. 24.4% of everyone else, a 13.7-point gap
- **Their attrition has a tenure "danger zone"** — it spikes in years 4–6, peaking at year 5 (~72%). Regular employees' attrition is much flatter and peaks earlier (year 3, ~35%), with no comparable surge
- **The leavers show signs of burnout** — they averaged ~248 hours/month vs. ~207 for those who stayed, and reported lower satisfaction (0.49 vs. 0.66)

Put together: top performers leave at a higher rate than average, mostly around the 4–6 year mark, and the ones who leave look overworked and less satisfied than the ones who stay. Correlational, not causal — but a pattern worth flagging to HR.

## Models

Logistic regression, a decision tree, and a random forest, all trained/tested on the same 80/20 split for a fair comparison.

| Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---|---|---|
| Accuracy | 71.6% | 79.5% | **86.2%** |
| Precision (left) | 0.55 | 0.65 | **0.84** |
| Recall (left) | 0.22 | 0.66 | 0.66 |
| AUC | 0.73 | 0.76 | **0.83** |

Random forest wins on every metric, and unlike the standalone decision tree (98% train vs. 79.5% test accuracy), it doesn't show the same overfitting gap in practice. Hyperparameter tuning was skipped on purpose.

**Worth being upfront about:** even the best model only catches 66% of employees who actually leave. Useful for flagging risk, not a fully solved problem.

## AI layer

Two small tools built on top of the random forest model, using SHAP for explainability and an LLM (via Groq) to turn model output into something non-technical:

- **Individual predictor** — enter one employee's stats, get a leave probability plus a SHAP-driven explanation of *why*, written up in plain language. Useful for something like a 1:1 or performance review.
- **Segment risk tool** — filter the dataset by any condition (department, satisfaction below a threshold, etc.) and get back an aggregate risk summary for that group, also narrated by the LLM.

Together they mean someone in HR can ask "why is this person at risk?" or "how bad is attrition in sales?" and get a written answer, no Python or dashboard-reading required.

## Power BI dashboard

Handles the simpler, single-variable and departmental views, leaving the deeper multivariable analysis to Python.

![Employee Attrition Dashboard](images/attrition_dashboard.png)

A few things worth noting:
- Overall attrition sits at **29.3%** — right between the good-employee rate (~38%) and the general rate (~24%) from the Python EDA, a nice sanity check
- Tenure-based attrition peaks at year 5 (~55%), consistent with the Python finding
- Project count barely differs between leavers and stayers at the whole-population level (3.86 vs. 3.79) — it only becomes meaningful once you narrow in on the "good employee" subgroup, which is exactly why that deeper analysis needed Python rather than the dashboard

## Stack

Python (pandas, scikit-learn), SHAP, Groq LLM API, Power BI Desktop.

---
*Python rebuild and extension of an earlier R-based attrition analysis, completed independently.*

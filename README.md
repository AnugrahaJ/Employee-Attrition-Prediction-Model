# Employee Attrition Prediction

An end-to-end HR attrition analysis and prediction project in Python — covering exploratory data analysis, three classification models, a SHAP-explained LLM layer for HR-facing narratives, and a Power BI dashboard. This is a Python port and extension of an earlier R-based project, rebuilt from scratch with a deeper EDA, a proper model comparison, and two new AI-powered tools that go beyond what either the models or the dashboard can do alone.

## Overview

The dataset covers ~10,500 employees, with features on satisfaction, performance, workload, tenure, salary, and department, plus whether the employee left (`left`). The goal was to move past a generic "what predicts attrition" analysis and toward a sharper, more specific question: **do an organization's best performers leave at a different rate — and if so, why?**

The project has four parts:

1. **Data Cleaning & EDA** — cleaning the data and investigating a specific, non-obvious question about high-performing employees
2. **Model Building** — logistic regression, decision tree, and random forest, compared on the same train/test split
3. **AI Layer** — two tools that use the trained model plus SHAP and an LLM to turn predictions into plain-language HR narratives
4. **Power BI Dashboard** — a stakeholder-facing view built around simpler, single-variable and departmental breakdowns

## 1. Exploratory Data Analysis

### Defining a "good employee"

Rather than asking generic questions like "does satisfaction correlate with leaving," the EDA centers on a specific, self-defined segment: employees whose `last_evaluation` is above the dataset mean **and** who have completed **4 or more projects**. `left` was deliberately excluded from this definition — including it would have made "good employee" partly defined by the outcome being studied, which would bias every downstream finding.

By this definition, **~35.5% of employees (3,729 of 10,499)** qualify as "good employees."

### Findings

**Good employees leave more often, not less.**
- Good employees: **38.1%** left
- Regular employees: **24.4%** left
- Gap: **~13.7 percentage points**

**Their attrition is concentrated in a tenure "danger zone."**
Among good employees, attrition spikes sharply in years 4–6 of tenure, **peaking at year 5 (~71.7%)**. Regular employees show a much milder pattern, peaking earlier (year 3, ~34.9%) with no dramatic surge — the danger zone is specific to top performers, not the workforce generally.

**Good employees who left show signs of overwork.**
- Left: averaged **~248 hours/month**
- Stayed: averaged **~207 hours/month**
- Gap: **~42 hours/month (~10 extra hours/week)**

**Good employees who left were less satisfied — but not dramatically so.**
- Left: satisfaction averaged **0.49**
- Stayed: satisfaction averaged **0.66**
- Gap: **~0.17** — a meaningful but mid-scale difference, not a collapse in morale

### Narrative

Good employees leave at a significantly higher rate than the rest of the workforce, with attrition sharply concentrated around years 4–6 of tenure. Those who leave show both a strong overwork signal and moderately lower satisfaction than good employees who stay — together pointing toward **burnout as a plausible, though not definitively proven, driver of attrition among top performers.**

*(These are correlational EDA findings, not causal claims.)*

## 2. Model Building

Three models were built and compared on an identical 80/20 train/test split (`random_state=121`), reused across all three for a fair comparison.

**Preprocessing:**
- A phantom fully-blank row (index 10499) was found and dropped
- `sales` (department) and `salary` were one-hot encoded with `pd.get_dummies(..., drop_first=True)` to avoid the dummy variable trap
- Features were scaled with `StandardScaler` — fit on training data only, applied to test data — for logistic regression only; the tree-based models used unscaled features, since split-based models aren't affected by feature scale

**Note:** an early version had a naming collision — the `sales` department dummy column shared its name with the original `sales` categorical column, silently dropping the department dummy from the training data. Fixed by renaming the raw column to `department` and retraining; results did not change significantly.

### Results

| Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---|---|---|
| Accuracy | 71.6% | 79.5% | **86.2%** |
| Precision (left) | 0.55 | 0.65 | **0.84** |
| Recall (left) | 0.22 | 0.66 | 0.66 |
| AUC | 0.73 | 0.76 | **0.83** |

**Logistic regression** serves as the baseline. It only catches 22% of employees who actually leave — insufficient for the practical goal of flagging at-risk employees.

**Decision tree** improves substantially on every metric, especially recall (22% → 66%), but shows clear overfitting: 98.0% training accuracy vs. 79.5% test accuracy.

**Random forest** is the strongest model on every metric — best or tied-best — without the decision tree's overfitting gap (its 98.0% training accuracy reflects rows with identical features but different outcomes, which no tree depth can separate, not overfitting; test performance is what matters, and it clearly favors random forest). Hyperparameter tuning was intentionally skipped.

**Limitation:** even the best model still misses about a third of employees who actually leave (recall 0.66). This is a real constraint worth stating plainly — the model is a useful risk-flagging tool, not a fully solved prediction problem.

This mirrors the outcome of the original R-based version of this project, where random forest was likewise the only model that met the bar.

## 3. AI Layer

Two tools built on top of the random forest model, using [SHAP](https://shap.readthedocs.io/) for explainability and an LLM (via Groq) to translate model output into plain-language narratives an HR stakeholder can read without any data science background.

### Individual Employee Predictor

Given one employee's raw stats, predicts their probability of leaving and explains *why* in plain English.

**Flow:**
1. Employee data goes in as a Python dict → converted to a one-row DataFrame matching the model's training columns
2. `model.predict_proba()` returns a leave probability
3. `shap.TreeExplainer` scores which features pushed that specific prediction up or down, and by how much
4. The probability, SHAP values, and the employee's actual attribute values are fed into a prompt sent to an LLM, which returns a plain-language narrative — useful for individual conversations like performance reviews

Built interactively (`input()`-based) so it works for any employee, not just a hardcoded example.

### Segment Risk Query Tool

Instead of one employee, filters the whole dataset to a segment (e.g. "everyone in HR," "everyone with `last_evaluation` < 0.5") and returns an aggregate risk narrative for that group.

**Flow:**
1. `attrition_by_filter(column, symbol, value)` — a single reusable function using Python's `operator` module to apply any comparison (`<`, `>`, `==`, etc.) to any column
2. Filters the dataframe, runs `predict_proba()` on the matching group, and buckets each employee into high (≥70%), medium (30–70%), or low (<30%) risk
3. Returns the average risk and tier counts
4. An interactive version lets a user type in the column, operator, and value live
5. Same as the individual tool — aggregate stats are fed into an LLM prompt to produce an HR-readable segment summary: overall risk level, concentration, and retention implications

Together, these mean an HR stakeholder — no Python, no dashboard-reading skills required — can ask "why is this person at risk?" or "how bad is attrition in the sales team?" and get a written answer instead of raw numbers or a chart to interpret.

## 4. Power BI Dashboard

Built to handle the simpler, single-variable and departmental relationships, leaving the deeper multivariable analysis (like the "good employee" segment) to Python — see *Findings* below for why that division mattered.

**Setup:** loaded the same dataset, removed the same phantom blank row via Power Query, and applied a custom theme (slate blue, terracotta accent, cream background).

**Measures (DAX):**
- `Attrition Percentage = DIVIDE(SUM(hr_train[left]), COUNTROWS(hr_train))` — one measure, reused across five category breakdowns (salary, department, tenure, work accident, promotion) via filter context rather than writing a separate formula for each
- `Avg Satisfaction`, `Average Hours Spent`, `Average Projects Done` — simple averages

**Calculated columns:** `Status`, `Work_accident`, and an equivalent promotion column, each converting a raw 0/1 into a readable label for chart axes.

**Dashboard structure** — 10 questions mapped to visuals: overall attrition rate, department/salary slicers, attrition by salary band, attrition by department, satisfaction/hours/projects (stayed vs. left), attrition by tenure, attrition by work accident, and attrition by promotion history.

### Findings

- **Overall attrition rate: 29.3%** — sits neatly between the "good employee" rate (~38%) and the general rate (~24%) found in the Python EDA, a useful sanity check across the two halves of the project
- **Project count barely differed** between leavers (3.86) and stayers (3.79) at the whole-population level — in contrast to the Python EDA, where project count mattered specifically within the "good employee" subgroup. This is a clear illustration of why the deeper, segmented analysis needed to live in Python rather than the dashboard: the signal only appears once you condition on the right subgroup
- **Tenure-based attrition peaked at year 5 (55.48%)** — consistent with the Python EDA's good-employee attrition spike

## Tech Stack

- **Python** — pandas, scikit-learn (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `StandardScaler`, `train_test_split`)
- **SHAP** — model explainability for individual predictions
- **Groq (LLM API)** — plain-language narrative generation, accessed via `.env` + `python-dotenv` (no keys committed)
- **Power BI Desktop** — stakeholder dashboard, DAX measures and calculated columns

## Key Takeaways

- Attrition risk isn't evenly distributed — it's sharpest among an organization's best performers, and concentrated in a specific tenure window (years 4–6)
- A model's overall accuracy can be misleading; recall on the minority class (employees who actually leave) is the metric that matters for a retention use case, and even the best model here misses a third of true leavers
- Prediction alone isn't the deliverable — pairing a model with SHAP and an LLM turns a probability score into something an HR stakeholder can actually act on and understand
- Simple, single-variable patterns (dashboard) and deeper, segmented multivariable patterns (Python) can tell different stories from the same data — both are needed

---

*This project is a Python rebuild and extension of an earlier R-based attrition analysis, completed independently.*

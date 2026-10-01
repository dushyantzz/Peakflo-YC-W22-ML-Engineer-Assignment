# Peakflo Expense Account Classification — ML Engineer Take-Home

## 1. Project Overview

This project builds a leakage-safe multiclass ML system to predict the correct `accountName` for expense transactions.

The dataset contains **4,894 transactions, 337 vendors, and 103 account-name classes**. The objective is to reduce repetitive manual accounting categorization while keeping the system auditable and safe for production use.

The final model is a **LinearSVC** pipeline using word-level TF-IDF, character-level TF-IDF, vendor information, and numerical/text-structure features.

### Final verified performance

| Metric | Result |
|---|---:|
| Group-aware CV accuracy | **85.63%** |
| Final test accuracy | **85.28%** |
| Weighted F1 | **84.23%** |
| Macro F1 | **71.64%** |
| Balanced accuracy | **74.13%** |

The final test accuracy exceeds Peakflo's required **85% threshold**.

---

# 2. Evaluation Criteria — How This Submission Addresses Them

## A. Performance — 40%

### Does the model meet the 85% accuracy threshold?

**Yes.** The final model achieves **85.28% accuracy on the untouched test set**, exceeding the required 85% threshold.

The development-stage group-aware cross-validation accuracy is **85.63%**, and the difference between CV and final test accuracy is only **0.35 percentage points**. This provides a consistent estimate of model generalization rather than relying on a single favorable split.

### Are the evaluation methods appropriate and rigorous?

**Yes.** The evaluation is designed specifically to reduce leakage from related transactions.

- The final test set is separated before model selection.
- `_id` is used as a grouping variable rather than as a predictive feature.
- Related groups are prevented from crossing validation boundaries.
- Group-aware cross-validation is used during model selection.
- `accountId` is excluded because it directly identifies the accounting target.
- Target-derived information is fitted only within the appropriate training partition.
- The final test set is used only for the final performance estimate.

This makes the reported 85.28% substantially more meaningful than an accuracy obtained from a simple random row-level split.

### Quality of performance metrics and analysis

Accuracy is reported together with:

- Weighted precision: **84.95%**
- Weighted recall: **85.28%**
- Weighted F1: **84.23%**
- Macro F1: **71.64%**
- Balanced accuracy: **74.13%**

This is important because the dataset contains **103 account classes with significant class imbalance**. Macro F1 and balanced accuracy provide additional visibility into performance across smaller classes instead of allowing large classes to dominate the evaluation.

---

# B. Approach & Methodology — 30%

## Soundness of the technical approach

The problem is primarily text-heavy, multiclass, and relatively small in sample size. A sparse linear model is therefore a strong fit for the problem.

The final pipeline combines:

- Word-level TF-IDF for meaningful words and phrases.
- Character-level TF-IDF for abbreviations, spelling variations, identifiers, and semi-structured text.
- Vendor features for recurring supplier/account relationships.
- Log-transformed transaction amount features.
- Text length, token-count, and digit-count features.

**LinearSVC** was selected because it handles high-dimensional sparse text representations efficiently and provides a strong baseline without requiring a very large training dataset.

## Feature engineering creativity and effectiveness

The feature engineering intentionally combines three types of information available at prediction time:

**Semantic information**
- `itemName`
- `itemDescription`
- word and character TF-IDF

**Business/entity information**
- `vendorId`

**Transaction structure**
- transaction amount
- text length
- token counts
- digit counts

This combination allows the model to use both the language of an expense and its structured characteristics.

The use of both word and character TF-IDF is particularly useful for expense data because descriptions often contain short business phrases, abbreviations, product names, plan identifiers, and inconsistent formatting.

## Handling data challenges: imbalance, missing values, duplicates, and leakage

The analysis identified significant class imbalance across the 103 account categories. Rare classes were retained rather than removed because removing them would change the original business problem.

Missing descriptions were handled safely rather than dropping transactions.

Duplicate and repeated transaction groups were investigated. `_id` is used only for group-aware validation so related records do not artificially inflate validation performance.

Most importantly, `accountId` is deliberately excluded because it is an alternative representation of the target and would create direct target leakage.

## Validation and overfitting control

The model-selection process uses development data and group-aware cross-validation. The final test set remains untouched until final evaluation.

The final diagnostic reports:

- Training accuracy: **98.78%**
- Group-aware CV accuracy: **85.63%**
- Test accuracy: **85.28%**
- Train-test gap: **13.50 percentage points**

The gap is explicitly reported rather than hidden. This demonstrates that the next improvement target is **generalization**, not simply maximizing training accuracy.

---

# C. Code Quality — 15%

## Clarity and organization

The notebook is organized as an end-to-end ML workflow:

1. Problem definition
2. Data loading
3. Data quality analysis
4. Exploratory analysis
5. Leakage checks
6. Feature engineering
7. Group-aware data splitting
8. Model development
9. Validation
10. Final evaluation
11. Error analysis
12. Deployment considerations

Each major stage contains explanatory Markdown and comments so that the modeling decisions can be reviewed independently.

## Documentation and comments

The notebook documents why important decisions are made rather than only showing what was executed.

Examples include:

- Why `accountId` is excluded.
- Why `_id` is used only for grouping.
- Why group-aware validation is required.
- Why both word and character TF-IDF are used.
- Why accuracy alone is insufficient.
- Why the final test set must remain untouched.

## Reproducibility

The notebook uses fixed random seeds where applicable and keeps model selection separate from final test evaluation.

The required Python dependencies are provided in `requirements.txt`.

The notebook is intended to be run sequentially from top to bottom using the supplied `accounts-bills.json`.

## Best practices

The implementation follows several practical ML best practices:

- No target identifier is used as a feature.
- Evaluation data is protected from model selection.
- Group leakage is explicitly checked.
- Multiple metrics are reported.
- Rare classes are not silently removed.
- Generalization gaps are measured.
- Production limitations are documented.

---

# D. Communication — 15%

## Clarity of written description

The accompanying report explains the problem from both a machine-learning and business perspective. It covers the data, methodology, model choice, validation strategy, results, limitations, and deployment approach without relying on the code to explain the reasoning.

## Quality of insights from data analysis

The analysis does more than report a single accuracy number.

Key observations include:

- The dataset contains **103 account classes**, making this a challenging multiclass problem.
- The target distribution is strongly imbalanced.
- Several account categories have very small support.
- Related `_id` groups make ordinary random splitting unsafe.
- The final CV and test results are closely aligned.
- The model has a measurable train-test generalization gap.
- Category-level metrics show that overall accuracy does not represent every account equally well.

These observations directly influence the modeling and improvement strategy.

## Critical thinking about limitations and improvements

The current model is treated as a strong baseline, not as the final ceiling.

The main limitation is the **13.50 percentage-point train-test gap**. Rather than responding by simply increasing model complexity, the improvement plan focuses on generalization.

Potential next steps include:

1. Stronger semantic text representations.
2. A vendor-independent fallback model.
3. Confidence-aware prediction and human review.
4. Hierarchical classification.
5. Targeted modeling of recurring confusion pairs.
6. Active learning for rare and difficult categories.
7. Additional labeled examples for underrepresented accounts.
8. Temporal validation and model-drift monitoring.

The objective is to move toward **90%+ performance without introducing leakage or sacrificing reliability**.

## Appropriate use of visualizations

The notebook uses visual analysis where it supports a decision, including class-distribution analysis, data-quality exploration, and model/error diagnostics.

Visualizations are used to communicate patterns rather than as decoration. The written report intentionally excludes visualizations and code because Peakflo requested those separately from the 4–6 page written description.

---

# 3. How This Model Can Help Peakflo

The model can be integrated as a decision-support layer in the expense-processing workflow:

```text
Expense ingestion
       ↓
Feature extraction
       ↓
Account prediction
       ↓
Confidence / risk check
       ↓
 ┌─────┴─────┐
 ↓           ↓
High       Low
confidence confidence
 ↓           ↓
Auto       Finance
process    review
       ↓
Feedback / corrections
       ↓
Periodic retraining
```

This design avoids treating ML predictions as unconditional accounting decisions.

High-confidence, repetitive expenses can be automated, while uncertain transactions can be reviewed by finance specialists. Reviewer corrections can then become new labeled examples for future model iterations.

A useful production KPI would therefore be **automation coverage at an agreed precision/error threshold**, rather than accuracy alone.

---

# 4. Roadmap Toward 90%+

The current **85.28% final test accuracy** already satisfies the assignment requirement. The next objective is to determine whether the remaining error can be reduced through better representations and validation rather than overfitting.

The strongest areas for further experimentation are:

### 1. Better semantic text understanding

Explore richer text normalization, domain-specific phrase handling, and compact semantic embeddings to capture equivalent expense descriptions written in different ways.

### 2. Vendor-independent fallback

Develop a dedicated model using transaction text and amount without depending on vendor identity. This provides robustness when a supplier is new or vendor history is insufficient.

### 3. Confidence-aware routing

Use prediction confidence to distinguish straightforward classifications from ambiguous cases. High-confidence cases can be automated, while uncertain cases can be routed to a reviewer.

### 4. Hierarchical classification

First predict a broader accounting family and then classify within that smaller group. This may make the 103-class problem easier to learn.

### 5. Targeted confusion-pair modeling

Identify recurring account confusions and introduce specialized features or decision logic for those specific account pairs.

### 6. Active learning

Prioritize human review for uncertain or information-rich transactions so that the next training cycle receives the most valuable new labels.

### 7. More data for rare classes

Several categories have very few examples. Increasing their labeled support may provide more benefit than increasing model complexity.

### 8. Temporal validation and drift monitoring

Once reliable transaction dates are available, evaluate future-period performance and monitor changes in vendors, text patterns, class distributions, confidence, and reviewer corrections.

---

# 5. Reproducibility

## Dataset

Place the supplied file in the same directory as the notebook:

```text
accounts-bills.json
```

## Environment

Recommended Python version:

```text
Python 3.11
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Then open the final notebook and execute the cells sequentially.

## Submission package

The recommended submission package is:

```text
.
├── accounts-bills.json
├── peakflo_expense_account_classification_leakage_safe.ipynb
├── README.md
├── requirements.txt
├── Peakflo_ML_Engineer_Take_Home_Report.md
└── Peakflo_ML_Engineer_Take_Home_Report.pdf
```

---

# 6. Final Takeaway

This submission is designed to demonstrate more than a model that crosses an accuracy threshold. It demonstrates a complete ML workflow: understanding the business problem, identifying leakage risks, handling class imbalance, engineering useful features, validating rigorously, measuring generalization, documenting limitations, and defining a practical path toward further improvement.

The current model achieves **85.28% accuracy on the untouched test set**, exceeding the required threshold, while the evaluation methodology provides a defensible estimate of generalization.

The next goal is not simply to make the model more complex. It is to make it **more reliable, more generalizable, and more useful to Peakflo's finance workflow**, with a clear engineering path toward 90%+ performance.
#

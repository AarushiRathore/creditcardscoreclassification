"""
Credit Score Classification: Data Mining ETL & Pre-processing Pipeline

Source: https://www.kaggle.com/datasets/parisrohan/credit-score-classification
Input:  credit_score.csv (keep in the same folder as this script)
Output: credit_score_cleaned.csv

Install dependencies:
    pip install pandas numpy matplotlib scikit-learn

Run:
    python Credit_Score_ETL_Pipeline.py

This script is the .py equivalent of the assignment Jupyter Notebook.
"""


# Data Mining ETL & Pre-processing Pipeline
# # Data Mining ETL & Pre-processing Pipeline# ## Credit Score Classification Dataset## **Course:** Data Mining and Warehousing  # **Dataset source:** [Kaggle – Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)  # **Local input file:** `credit_score.csv`  # **Target variable:** `Credit_Score` (Good, Standard, Poor)## ### Aim# To extract, profile, clean, transform, reduce, summarize and load a customer credit-score dataset using Python.## > Run this notebook from top to bottom. Keep `credit_score.csv` in the same folder as this notebook. The notebook deliberately shows the calculations and intermediate outputs needed for the handwritten report.

# 1. Import libraries and extract the data
# ## 1. Import libraries and extract the data## Pandas handles tabular data, NumPy handles numerical operations, and Matplotlib creates plots. `read_csv()` is the **Extract** step of ETL.

# ---- Code cell 3 ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

INPUT_FILE = Path("credit_score.csv")
OUTPUT_FILE = Path("credit_score_cleaned_final.csv")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Place credit_score.csv in the same folder as this notebook."
    )

raw = pd.read_csv(INPUT_FILE, low_memory=False)
df = raw.copy()

print("Extraction successful.")
print("Initial shape (rows, columns):", raw.shape)
display(raw.head())


# 2. Dataset profiling and initial audit
# ## 2. Dataset profiling and initial audit## We inspect dimensions, data types, missing cells, exact duplicate rows, target classes and numerical ranges **before cleaning**. Repeated `Customer_ID` values are expected because each customer has multiple monthly records; they are not automatically duplicate rows.

# ---- Code cell 5 ----
print("Dataset dimensions:", raw.shape)
print("\nData types:")
display(raw.dtypes.astype(str).value_counts().rename("Number of columns").to_frame())

profile = pd.DataFrame({
    "Data type": raw.dtypes.astype(str),
    "Missing values": raw.isna().sum(),
    "Missing %": (raw.isna().mean() * 100).round(2),
    "Unique values": raw.nunique(dropna=True)
}).sort_values("Missing values", ascending=False)

display(profile)
print("Exact duplicate rows:", raw.duplicated().sum())
print("\nTarget class counts:")
display(raw["Credit_Score"].value_counts(dropna=False).to_frame("Records"))
print("\nNumerical description before cleaning:")
display(raw.describe(include="all").T.head(20))


# Initial audit findings
# ### Initial audit findings## Examples of issues to investigate:# 1. Missing values already represented as NaN.# 2. Placeholder strings such as `_`, `_______`, `NM` and `!@9#%8`.# 3. Numeric-looking values stored as text, sometimes with underscore characters.# 4. Impossible or implausible values (for example, negative ages or extreme account counts).# 5. Extreme numerical observations that should be checked using IQR.## The dataset has 100,000 rows and 28 columns in the supplied file. Confirm this using the output above.

# 3. Transform: standardise text and missing-value markers
# ## 3. Transform: standardise text and missing-value markers## Whitespace is removed so `" Good "` and `"Good"` do not become separate categories. Placeholder strings are converted to `NaN`, allowing Pandas to recognise them as missing values. The special string `__10000__` in monthly investment is treated as a sentinel (a placeholder code), not as a genuine investment amount.

# ---- Code cell 8 ----
# Trim leading/trailing spaces in text columns
text_cols = df.select_dtypes(include=["object", "string"]).columns
for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

# Preserve and identify the special sentinel before general replacement
investment_sentinel = df["Amount_invested_monthly"].eq("__10000__")

placeholders = ["", "_", "_______", "!@9#%8", "NM", "Not Specified"]
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].replace(placeholders, pd.NA)

df.loc[investment_sentinel, "Amount_invested_monthly"] = pd.NA

print("Missing cells before placeholder treatment:", int(raw.isna().sum().sum()))
print("Missing cells after placeholder treatment:", int(df.isna().sum().sum()))


# 4. Transform: convert columns to appropriate numeric types
# ## 4. Transform: convert columns to appropriate numeric types## Numeric data stored as text cannot be reliably used in arithmetic or statistical algorithms. We remove formatting underscores and use `to_numeric(errors="coerce")`. Invalid text becomes missing (`NaN`) rather than stopping the notebook with an error.

# ---- Code cell 10 ----
numeric_cols = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col].astype("string").str.replace("_", "", regex=False),
        errors="coerce"
    )

# Convert "22 Years and 1 Months" into total months (22*12 + 1 = 265)
history_parts = df["Credit_History_Age"].astype("string").str.extract(
    r"(?P<years>\d+) Years and (?P<months>\d+) Months"
)
df["Credit_History_Months"] = (
    pd.to_numeric(history_parts["years"], errors="coerce") * 12
    + pd.to_numeric(history_parts["months"], errors="coerce")
)
df = df.drop(columns=["Credit_History_Age"])

print("Numeric conversion complete.")
display(df[numeric_cols + ["Credit_History_Months"]].dtypes.to_frame("Data type"))


# 5. Data cleaning: invalid values and sentinels
# ## 5. Data cleaning: invalid values and sentinels## The rules below are broad data-validation rules for this coursework, not official lending rules. They turn clearly implausible values into missing values, which are imputed later. We do not delete entire customer rows merely because one field is invalid.## For this assignment, ages outside 14–100 and negative or exceptionally large count values are flagged. The chosen limits must be stated as assumptions in the report.

# ---- Code cell 12 ----
cleaning_log = {}

def set_invalid_to_missing(column, condition, reason):
    condition = condition.fillna(False)
    cleaning_log[reason] = int(condition.sum())
    df.loc[condition, column] = np.nan

set_invalid_to_missing(
    "Age", (df["Age"] < 14) | (df["Age"] > 100), "Age outside 14–100"
)
set_invalid_to_missing(
    "Num_Bank_Accounts",
    (df["Num_Bank_Accounts"] < 0) | (df["Num_Bank_Accounts"] > 50),
    "Bank account count outside 0–50"
)
set_invalid_to_missing(
    "Num_Credit_Card",
    (df["Num_Credit_Card"] < 0) | (df["Num_Credit_Card"] > 50),
    "Credit card count outside 0–50"
)
set_invalid_to_missing(
    "Interest_Rate",
    (df["Interest_Rate"] < 0) | (df["Interest_Rate"] > 100),
    "Interest rate outside 0–100"
)
set_invalid_to_missing(
    "Num_of_Loan",
    (df["Num_of_Loan"] < 0) | (df["Num_of_Loan"] > 50),
    "Loan count outside 0–50"
)
set_invalid_to_missing(
    "Num_of_Delayed_Payment",
    (df["Num_of_Delayed_Payment"] < 0) |
    (df["Num_of_Delayed_Payment"] > 100),
    "Delayed payment count outside 0–100"
)
set_invalid_to_missing(
    "Num_Credit_Inquiries",
    (df["Num_Credit_Inquiries"] < 0) |
    (df["Num_Credit_Inquiries"] > 100),
    "Credit inquiries outside 0–100"
)
set_invalid_to_missing(
    "Monthly_Balance",
    df["Monthly_Balance"] < -1_000_000,
    "Extreme negative monthly-balance sentinel"
)

print("Invalid-value counts:")
display(pd.Series(cleaning_log, name="Values changed to missing").to_frame())


# 6. Outlier detection using the IQR method
# ## 6. Outlier detection using the IQR method## For a numerical feature:# - IQR = Q3 − Q1# - Lower fence = Q1 − 1.5 × IQR# - Upper fence = Q3 + 1.5 × IQR## Values outside these fences are flagged as potential outliers. For this assignment, we **cap** values at the fences (winsorisation) for selected continuous attributes rather than deleting customer-month rows. This reduces the influence of extreme values while retaining observations. It is a preprocessing choice, not proof that every flagged value is an error.

# ---- Code cell 14 ----
continuous_features = [
    "Annual_Income", "Monthly_Inhand_Salary", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance"
]

iqr_rows = []
for col in continuous_features:
    s = df[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((s < lower) | (s > upper)).sum())
    iqr_rows.append([col, q1, q3, iqr, lower, upper, count])

iqr_table = pd.DataFrame(
    iqr_rows,
    columns=["Feature", "Q1", "Q3", "IQR", "Lower fence", "Upper fence", "Outliers"]
)
display(iqr_table.round(3))

# Cap potential outliers at the IQR fences
for _, row in iqr_table.iterrows():
    col = row["Feature"]
    df[col] = df[col].clip(lower=row["Lower fence"], upper=row["Upper fence"])

print("IQR outlier capping completed for", len(continuous_features), "features.")


# Manual IQR calculation example
# ### Manual IQR calculation example## Use the `Outstanding_Debt` row in the IQR table above and copy its actual Q1 and Q3 values.## 1. IQR = Q3 − Q1# 2. Lower fence = Q1 − (1.5 × IQR)# 3. Upper fence = Q3 + (1.5 × IQR)# 4. A value below the lower fence or above the upper fence is flagged.# 5. In this notebook, such values are capped at the relevant fence.## **Illustration only:** If Q1 = 500 and Q3 = 1,000, then IQR = 500; lower fence = 500 − 750 = −250; upper fence = 1,000 + 750 = 1,750. Replace these illustrative numbers with the actual notebook output when writing your report.

# 7. Missing-value handling (imputation)
# ## 7. Missing-value handling (imputation)## Numerical columns are filled with their median because the median is less sensitive to extreme observations than the mean. Categorical columns are filled with `"Unknown"` so that missing categories remain visible rather than being guessed. Rows with a missing target label are removed because a supervised classification model cannot learn a target class that is unknown.

# ---- Code cell 17 ----
rows_before_target_filter = len(df)
df = df.dropna(subset=["Credit_Score"]).copy()
rows_removed_target = rows_before_target_filter - len(df)

# Keep a list of numeric columns that still exist
numeric_existing = [c for c in numeric_cols + ["Credit_History_Months"] if c in df.columns]

# Median imputation
imputation_medians = {}
for col in numeric_existing:
    median_value = df[col].median()
    imputation_medians[col] = median_value
    df[col] = df[col].fillna(median_value)

# Categorical imputation
categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

print("Rows removed due to missing target:", rows_removed_target)
print("Missing cells after imputation:", int(df.isna().sum().sum()))
print("Rows retained:", len(df))


# 8. Data transformation: Min-Max normalization and Z-score standardization
# ## 8. Data transformation: Min-Max normalization and Z-score standardization## Min-Max scales values to the interval [0, 1]. Z-score expresses a value as the number of standard deviations from the mean.## - Min-Max: x' = (x − minimum) / (maximum − minimum)# - Z-score: z = (x − mean) / standard deviation## We demonstrate both on `Outstanding_Debt`. These are added as new columns so the original cleaned feature is retained.

# ---- Code cell 19 ----
feature = "Outstanding_Debt"
x = df[feature].astype(float)
x_min = x.min()
x_max = x.max()
x_mean = x.mean()
x_std = x.std(ddof=0)

df[feature + "_MinMax"] = (x - x_min) / (x_max - x_min)
df[feature + "_Zscore"] = (x - x_mean) / x_std

sample_value = float(x.iloc[0])
sample_minmax = (sample_value - x_min) / (x_max - x_min)
sample_zscore = (sample_value - x_mean) / x_std

print(f"Feature: {feature}")
print(f"Sample raw value: {sample_value:.4f}")
print(f"Minimum: {x_min:.4f}; Maximum: {x_max:.4f}")
print(f"Mean: {x_mean:.4f}; Standard deviation: {x_std:.4f}")
print(f"Min-Max result: ({sample_value:.4f} - {x_min:.4f}) / "
      f"({x_max:.4f} - {x_min:.4f}) = {sample_minmax:.6f}")
print(f"Z-score result: ({sample_value:.4f} - {x_mean:.4f}) / "
      f"{x_std:.4f} = {sample_zscore:.6f}")
display(df[[feature, feature + "_MinMax", feature + "_Zscore"]].head())


# 9. Discretization using equal-width binning
# ## 9. Discretization using equal-width binning## Discretization converts a continuous numerical variable into a small number of categories. Here, age is divided into five equal-width intervals using `pd.cut()`. This can make age groups easier to summarize and compare.

# ---- Code cell 21 ----
df["Age_Bin"] = pd.cut(
    df["Age"].astype(float),
    bins=5,
    labels=["Very young", "Young", "Middle", "Older", "Oldest"],
    include_lowest=True
)
print("Age-bin counts:")
display(df["Age_Bin"].value_counts(sort=False).to_frame("Records"))


# 10. Data reduction: feature selection
# ## 10. Data reduction: feature selection## Data reduction decreases the number of attributes while retaining useful information. We remove direct identifiers (`ID`, `Name`, `SSN`, `Customer_ID`) and the raw month label from the analytical feature set. `Customer_ID` is retained separately only to demonstrate a group-aware train/test split, then excluded from model features. We do not merge external tables because no secondary dataset was supplied; therefore no join is required for this single-source ETL task.

# ---- Code cell 23 ----
identifier_columns = ["ID", "Name", "SSN", "Customer_ID", "Month"]
existing_identifiers = [c for c in identifier_columns if c in df.columns]

before_reduction_columns = df.shape[1]
model_ready = df.drop(columns=[c for c in existing_identifiers if c in df.columns]).copy()
after_reduction_columns = model_ready.shape[1]

print("Columns before feature reduction:", before_reduction_columns)
print("Identifier/time columns removed:", existing_identifiers)
print("Columns after feature reduction:", after_reduction_columns)
print("Dimensionality reduced by:", before_reduction_columns - after_reduction_columns)


# 11. Statistical summary: before and after cleaning
# ## 11. Statistical summary: before and after cleaning## For a fair comparison, use `Annual_Income` because it is a key numerical attribute. The "before" column is converted to numeric for measurement only; the "after" column reflects validation, IQR capping and median imputation. The table reports mean, median, variance, standard deviation and quartiles.

# ---- Code cell 25 ----
before_income = pd.to_numeric(
    raw["Annual_Income"].astype("string").str.replace("_", "", regex=False),
    errors="coerce"
)
after_income = df["Annual_Income"].astype(float)

def summary_stats(series, label):
    s = series.dropna().astype(float)
    return pd.Series({
        "Mean": s.mean(),
        "Median": s.median(),
        "Variance": s.var(ddof=1),
        "Standard deviation": s.std(ddof=1),
        "Q1 (25%)": s.quantile(0.25),
        "Q2 (50%)": s.quantile(0.50),
        "Q3 (75%)": s.quantile(0.75),
        "Valid observations": s.count()
    }, name=label)

comparison = pd.concat([
    summary_stats(before_income, "Before cleaning"),
    summary_stats(after_income, "After cleaning")
], axis=1)
display(comparison.round(3))


# 12. Class discrimination analysis: Good versus Poor
# ## 12. Class discrimination analysis: Good versus Poor## We compare the mean and median of selected financial attributes for Good and Poor credit-score classes. This is descriptive analysis: differences show association in this dataset, not proof that one attribute causes a credit score.

# ---- Code cell 27 ----
comparison_features = [
    "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Num_of_Delayed_Payment", "Interest_Rate"
]
class_data = df[df["Credit_Score"].isin(["Good", "Poor"])].copy()

class_summary = class_data.groupby("Credit_Score")[comparison_features].agg(
    ["mean", "median"]
).T
display(class_summary.round(3))

class_counts = df["Credit_Score"].value_counts().to_frame("Records")
display(class_counts)

# Plot class distribution
df["Credit_Score"].value_counts().plot(kind="bar", title="Credit Score Class Distribution")
plt.xlabel("Credit score class")
plt.ylabel("Number of customer-month records")
plt.tight_layout()
plt.show()

# Plot average outstanding debt by class
df.groupby("Credit_Score")["Outstanding_Debt"].mean().plot(
    kind="bar", title="Mean Outstanding Debt by Credit Score"
)
plt.xlabel("Credit score class")
plt.ylabel("Mean outstanding debt")
plt.tight_layout()
plt.show()


# 13. Load phase: export the processed dataset
# ## 13. Load phase: export the processed dataset## The **Load** phase writes the processed table to a CSV file. CSV is portable and can be reopened in Python, Excel, Power BI or a database import tool.

# ---- Code cell 29 ----
# Export the cleaned, transformed and reduced dataset (without direct identifiers)
model_ready.to_csv(OUTPUT_FILE, index=False)

print("Export completed:", OUTPUT_FILE.resolve())
print("Final dataset dimensions:", model_ready.shape)
print("Remaining missing cells:", int(model_ready.isna().sum().sum()))
print("Final duplicate rows:", int(model_ready.duplicated().sum()))
print("File size (MB):", round(OUTPUT_FILE.stat().st_size / (1024**2), 2))


# 14. Data mining readiness and conclusion
# ## 14. Data mining readiness and conclusion## The processed dataset can support **classification** using `Credit_Score` as the target. It can also support exploratory clustering after suitable encoding and scaling. Before training a predictive model, use a preprocessing pipeline fitted on training data only to avoid leakage.## Because each customer has multiple monthly records, split by `Customer_ID` groups so that one customer's records do not appear in both training and test sets.

# ---- Code cell 31 ----
# Demonstrate a group-aware split using original customer identifiers.
# Customer_ID is used only for grouping and is not a model predictor.
groups = raw.loc[df.index, "Customer_ID"].astype(str)
target = df["Credit_Score"].astype(str)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(df, target, groups=groups))

print("Training records:", len(train_idx))
print("Testing records:", len(test_idx))
print("Customers in training:", groups.iloc[train_idx].nunique())
print("Customers in testing:", groups.iloc[test_idx].nunique())
print("Customer overlap:",
      len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])))

print("\nConclusion:")
print("The raw credit-score data was extracted, profiled, cleaned, transformed,")
print("reduced, summarized and exported. It is ready for further classification")
print("experiments after train-only encoding and scaling.")


# ---
# ---# ## Submission checklist## - [ ] Run every cell from top to bottom without errors.# - [ ] Keep the source CSV beside this notebook.# - [ ] Submit this `.ipynb` file as the executable code.# - [ ] Submit the exported `credit_score_cleaned.csv` if requested or as supporting output.# - [ ] Handwrite the 2–3 page report using the actual outputs from Sections 2, 6, 8, 11 and 12.# - [ ] In the report, clearly state that no secondary dataset was merged.# - [ ] Do not copy illustrative IQR numbers; use the actual `Outstanding_Debt` row from the generated IQR table.
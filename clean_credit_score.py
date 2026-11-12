"""
Clean the credit_score.csv dataset.

Requirements:
    pip install pandas numpy

Run:
    python clean_credit_score.py

Keep this script in the same folder as credit_score.csv, or edit INPUT_FILE.
"""

from pathlib import Path
import pandas as pd
import numpy as np

INPUT_FILE = Path("credit_score.csv")
OUTPUT_FILE = Path("credit_score_cleaned.csv")


def clean_credit_data(input_file, output_file):
    # 1. Load every column as text first. This avoids mixed-type warnings
    #    and lets us clean each column deliberately.
    data = pd.read_csv(input_file, dtype="string", low_memory=False)

    original_shape = data.shape
    missing_before = int(data.isna().sum().sum())
    duplicates_before = int(data.duplicated().sum())

    # 2. Remove leading/trailing spaces.
    for column in data.select_dtypes(include="string").columns:
        data[column] = data[column].str.strip()

    # Save masks for special sentinel values before removing underscores.
    amount_sentinel = data["Amount_invested_monthly"].eq("__10000__")

    # 3. Replace known placeholder text with proper missing values.
    placeholders = ["", "_", "_______", "!@9#%8", "NM", "Not Specified"]
    for column in data.select_dtypes(include="string").columns:
        data[column] = data[column].replace(placeholders, pd.NA)

    missing_after_placeholder_cleanup = int(data.isna().sum().sum())

    # In the loan list, "Not Specified" may appear alongside a real loan type.
    # Remove only the unknown item and keep any actual loan types.
    def clean_loan_types(value):
        if pd.isna(value):
            return pd.NA
        parts = []
        for item in value.split(","):
            item = item.strip()
            if item.lower().startswith("and "):
                item = item[4:].strip()
            if item and item.lower() != "not specified" and item not in parts:
                parts.append(item)
        return ", ".join(parts) if parts else pd.NA

    data["Type_of_Loan"] = data["Type_of_Loan"].apply(clean_loan_types).astype("string")

    # Convert "22 Years and 1 Months" into 265 total months.
    history_parts = data["Credit_History_Age"].str.extract(
        r"(?P<years>\d+) Years and (?P<months>\d+) Months"
    )
    data["Credit_History_Months"] = (
        pd.to_numeric(history_parts["years"], errors="coerce") * 12
        + pd.to_numeric(history_parts["months"], errors="coerce")
    )
    data = data.drop(columns=["Credit_History_Age"])

    # 4. Convert columns that should contain numbers.
    numeric_columns = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
        "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Total_EMI_per_month",
        "Amount_invested_monthly", "Monthly_Balance",
        "Credit_History_Months"
    ]

    for column in numeric_columns:
        # Underscores are formatting noise in values such as "14388.79_"
        # and "__10000__". Remove them before numeric conversion.
        cleaned_text = data[column].astype("string").str.replace("_", "", regex=False)
        data[column] = pd.to_numeric(cleaned_text, errors="coerce")

    cleaning_actions = {}

    def make_invalid(condition, column, reason):
        """Turn values that fail a stated validation rule into missing data."""
        condition = condition.fillna(False)
        cleaning_actions[reason] = int(condition.sum())
        data.loc[condition, column] = np.nan

    # 5. Apply broad plausibility rules.
    # These are project assumptions, not universal financial regulations.
    make_invalid(
        (data["Age"] < 14) | (data["Age"] > 100),
        "Age", "Age outside 14-100"
    )
    make_invalid(
        (data["Num_Bank_Accounts"] < 0) | (data["Num_Bank_Accounts"] > 50),
        "Num_Bank_Accounts", "Bank accounts outside 0-50"
    )
    make_invalid(
        (data["Num_Credit_Card"] < 0) | (data["Num_Credit_Card"] > 50),
        "Num_Credit_Card", "Credit cards outside 0-50"
    )
    make_invalid(
        (data["Interest_Rate"] < 0) | (data["Interest_Rate"] > 100),
        "Interest_Rate", "Interest rate outside 0-100"
    )
    make_invalid(
        (data["Num_of_Loan"] < 0) | (data["Num_of_Loan"] > 50),
        "Num_of_Loan", "Loans outside 0-50"
    )
    make_invalid(
        (data["Num_of_Delayed_Payment"] < 0) |
        (data["Num_of_Delayed_Payment"] > 100),
        "Num_of_Delayed_Payment", "Delayed payments outside 0-100"
    )
    make_invalid(
        (data["Num_Credit_Inquiries"] < 0) |
        (data["Num_Credit_Inquiries"] > 100),
        "Num_Credit_Inquiries", "Credit inquiries outside 0-100"
    )

    # This exact text pattern is a sentinel in this file, not a normal amount.
    make_invalid(
        amount_sentinel,
        "Amount_invested_monthly", "Amount invested sentinel __10000__"
    )

    # The source contains an extremely large negative Monthly_Balance sentinel.
    # Keep ordinary negative balances if they occur; only reject extreme values.
    make_invalid(
        data["Monthly_Balance"] < -1_000_000,
        "Monthly_Balance", "Extreme negative monthly-balance sentinel"
    )

    # Annual income should be broadly comparable with 12 months of
    # take-home salary. A ratio above 5 is flagged as suspicious.
    income_ratio = (
        data["Annual_Income"] / (data["Monthly_Inhand_Salary"] * 12)
    )
    make_invalid(
        income_ratio > 5,
        "Annual_Income", "Annual income > 5x annualised in-hand salary"
    )

    # 6. Do not keep direct personal identifiers in the analysis dataset.
    # ID and Customer_ID remain for traceability, but should not be model inputs.
    data = data.drop(columns=["Name", "SSN"])

    # The target is essential for supervised learning; do not invent its value.
    rows_before_target_check = len(data)
    data = data.dropna(subset=["Credit_Score"])
    target_rows_removed = rows_before_target_check - len(data)

    # 7. Fill numerical missing values using the median.
    for column in numeric_columns:
        if column in data.columns:
            data[column] = data[column].fillna(data[column].median())

    # Count-like columns should be whole numbers after median imputation.
    count_columns = [
        "Age", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
        "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries"
    ]
    for column in count_columns:
        data[column] = data[column].round().astype("Int64")

    # 8. Fill missing categorical values with an explicit "Unknown" label.
    categorical_columns = data.select_dtypes(include="string").columns
    for column in categorical_columns:
        data[column] = data[column].fillna("Unknown")

    # 9. Export the cleaned file.
    data.to_csv(output_file, index=False)

    # 10. Display a compact cleaning report.
    print("========== CREDIT DATA CLEANING REPORT ==========")
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape:  {data.shape}")
    print(f"Missing cells initially recognised by pandas: {missing_before}")
    print(f"Missing cells after placeholder cleanup: {missing_after_placeholder_cleanup}")
    print(f"Missing cells after imputation: {int(data.isna().sum().sum())}")
    print(f"Duplicate rows before cleaning: {duplicates_before}")
    print(f"Rows removed due to missing target: {target_rows_removed}")
    print("\nInvalid values changed to missing, then imputed:")
    for reason, count in cleaning_actions.items():
        print(f"- {reason}: {count}")
    print(f"\nSaved cleaned file to: {output_file.resolve()}")
    print("\nFirst five cleaned rows:")
    print(data.head().to_string(index=False))

    return data


if __name__ == "__main__":
    clean_credit_data(INPUT_FILE, OUTPUT_FILE)

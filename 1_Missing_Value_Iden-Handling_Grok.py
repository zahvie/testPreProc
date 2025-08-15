'''
Created on Apr 8, 2025

@author: MSII
'''
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro
from pathlib import Path
import os


# Default: local file path (Windows)
local_path = r"D:\Python\Thesis\ExSTraCS\test\DataSets\Real\Data\Loan Approval Data\loan_approval_data.csv"

# Alternative: Colab path
colab_path = "/content/data/loan_approval_data.csv"

# Decide which one to use
if os.path.exists(colab_path):
    file_path = colab_path
else:
    file_path = local_path

# Decide which one to use
if os.path.exists(colab_path):
    file_path = colab_path
else:
    file_path = local_path
    
def determine_imputation_strategy(df_column, total_records):
    num_missing = df_column.isnull().sum()
    missing_percentage = (num_missing / total_records) * 100
    if missing_percentage > 50:
        return {"Column Name": df_column.name, "Best Imputation Strategy": "Drop", "Best Value": None}
    
    if df_column.dtype == 'object':
        best_strategy = "Mode"
        best_value = df_column.mode()[0] if not df_column.mode().empty else "Missing"
    else:
        min_value = df_column.min()
        max_value = df_column.max()
        mean_value = df_column.mean()
        median_value = df_column.median()
        stat, p_value = shapiro(df_column.dropna()) if df_column.dropna().shape[0] > 50 else (None, 0)
        best_strategy = "Mean" if p_value > 0.05 else "Median"
        best_value = mean_value if p_value > 0.05 else median_value
    
    return {
        "Column Name": df_column.name,
        "Total Records": total_records,
        "Number of Missing Values": num_missing,
        "Percentage of Missing Values": f"{missing_percentage:.2f}%",
        "Best Imputation Strategy": best_strategy,
        "Best Value for Imputation": best_value
    }

data = pd.read_csv(file_path)
total_records = data.shape[0]
missing_value_columns = [col for col in data.columns if data[col].isnull().sum() > 0]

report = []
for col in missing_value_columns:
    result = determine_imputation_strategy(data[col], total_records)
    report.append(result)
    if result["Best Imputation Strategy"] == "Drop":
        data.drop(columns=col, inplace=True)
    elif result["Best Imputation Strategy"] == "Mode":
        data[col] = data[col].fillna(result["Best Value for Imputation"])
    else:
        imputer = SimpleImputer(strategy=result["Best Imputation Strategy"].lower())
        data[col] = imputer.fit_transform(data[[col]])

report_df = pd.DataFrame(report)
report_file_path = file_path.replace(".csv", "_missing_values_report.csv")
report_df.to_csv(report_file_path, index=False)
data.to_csv(file_path.replace(".csv", "_imputed.csv"), index=False)

print(f"Missing values report saved to: {report_file_path}")
print(f"Imputed dataset saved to: {file_path.replace('.csv', '_imputed.csv')}")
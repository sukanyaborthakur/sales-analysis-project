import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv('Sales.csv')
print(df)

#Print the Dataframe df and determine its dimension
print(df.shape)
rows = df.shape[0]
columns = df.shape[1]

print(f"The Dimensions are rows : {rows} and columns : {columns}")

#Check for missing values in the DataFrame
#Verify that there are no non-missing values in the DataFrame
missing_values= df.isnull().sum()
print("Missing values for each column : ")
print(missing_values)


print(df['Date'].isnull().sum())#for individual checking
print(df['Time'].isnull().sum())

missing_percentage =  ( df.isnull().sum()  / len(df))*100
print("Missing values percentage : ")
for col,pct in missing_percentage.items():
   print(f" {col} : {pct:.2f}%")


# Verify that there are no non-missing values in the DataFrame
total_missing = df.isnull().sum().sum()

if total_missing == 0 :

    print('No missing Values in the DataFrame')

else:

    print(f'{total_missing} missing values found')


#Quick Data Exploration
print('\nQuick Data Exploration : ')
print(df.dtypes)
df['Date'] = pd.to_datetime(df['Date'])
print(df.dtypes)
print(f'Date range : {df['Date'].min() } to {df['Date'].max()}')
print(f'States : {df['State'].unique()} ')
print(f'No of States : {df['State'].nunique()} ')
print(f'Customer Groups : {list(df['Group'].unique())} ')
for g in list(df['Group'].unique()):
    print(g)
print(f'Time :',list(df['Time'].unique()))
print(f'Units sold : {df['Unit'].min()} Units to {df['Unit'].max()} Units.' )
print(f'Sales Range :${df['Sales'].min()} to ${df['Sales'].max()}')



#2.Normalize Data for Analysis
#Create a new DataFrame called df_dataonly from the existing df object

df_dataonly = df[['Unit','Sales']].copy()

print(df_dataonly)

#Normalize
normalize= MinMaxScaler()
normalize_data = normalize.fit_transform(df_dataonly )
print(f'Normalize Data Shape : {normalize_data.shape }')
print(f'Type of Normalize Data : {type(normalize_data)}')

print(normalize_data)
print('min:',normalize_data.min())
print('max:',normalize_data.max())



#  3 Visualize Overall Trend
# Identify date column
# Detect date-like columns
date_candidates = [c for c in df.columns if "date" in c.lower()]
if not date_candidates:
    raise ValueError("No column name containing 'date' found.")

date_col = date_candidates[0]

# Convert to datetime safely
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# Confirm
print(f"Detected date column: {date_col}")
print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")



# Detect Unit and Sales columns
possible_unit_cols = [c for c in df.columns if c.lower() in ("unit","units","quantity","qty")]
possible_sales_cols = [c for c in df.columns if c.lower() in ("sales","amount","sale","total","revenue")]
if not possible_unit_cols or not possible_sales_cols:
    raise ValueError("Could not detect unit or sales columns automatically. Please rename columns or specify them.")
unit_col = possible_unit_cols[0]
sales_col = possible_sales_cols[0]
print("Using unit_col:", unit_col, "sales_col:", sales_col)




# plt.subplot(r,c,1)

# time series aggregation
df_time = df.dropna(subset=[date_col]).copy()

#Keep day resolution as datetime type
df_time["date_only"] = df_time[date_col].dt.date

# Daily aggregation
daily = df_time.groupby("date_only").agg({unit_col:"sum", sales_col:"sum"}).reset_index()
daily["date_only"] = pd.to_datetime(daily["date_only"])
r=2
c=2

# plt.subplot(r,c,1)
plt.figure(figsize=(10,4))
plt.plot(daily["date_only"], daily[unit_col], linewidth=1)
plt.title("Daily Units Sold")
plt.xlabel("Date")
plt.ylabel("Units")
plt.tight_layout()
plt.show()



# plt.subplot(r,c,3)
# plt.subplot(r,c,2)
plt.figure(figsize=(10,4))
plt.plot(daily["date_only"], daily[sales_col], linewidth=1)
plt.title("Daily Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()




# plt.subplot(r,c,2)
# Monthly analysis(Analyze Unit data)

df_time["year_month"] = df_time[date_col].dt.to_period("M").astype(str)
monthly_agg = df_time.groupby("year_month").agg({unit_col:"sum", sales_col:"sum"}).reset_index()

#(Unit analysis)
plt.figure(figsize=(8,4))
plt.bar(monthly_agg["year_month"], monthly_agg[unit_col])
plt.title("Monthly Units Sold")
plt.xlabel("Year-Month")
plt.ylabel("Units")
plt.tight_layout()
plt.show()



#(Sales analysis)
# plt.subplot(r,c,4)
plt.figure(figsize=(8,4))
plt.bar(monthly_agg["year_month"], monthly_agg[sales_col])
plt.title("Monthly Sales")
plt.xlabel("Year-Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()
print(df_time)


# 5. Describe Data
print('October Describe')
df_oct = df_time[df_time['year_month']=='2020-10']
print(df_oct.describe())
print('November Describe')
df_nov = df_time[df_time['year_month']=='2020-11']
print(df_nov.describe())
print('December Describe')
df_dec = df_time[df_time['year_month']=='2020-12']
print(df_dec.describe())


# plt.subplot(r,c,5)
plt.figure(figsize=(10,4))
plt.plot(monthly_agg["year_month"], monthly_agg[sales_col], marker="o")
plt.title("Consolidated Monthly Sales Trend")
plt.xlabel("Year-Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# plt.subplot(r,c,6)
# KPI Snapshot(Obtain a Comprehensive Snapshot)
total_sales = df_time[sales_col].sum()
total_units = df_time[unit_col].sum()
num_orders = df_time.shape[0]
avg_order_value = total_sales / num_orders if num_orders else 0
unique_states = df_time["State"].nunique() if "State" in df_time.columns else None
median_order_value = df_time[sales_col].median()
top_group = df_time["Group"].value_counts().idxmax() if "Group" in df_time.columns else None

print("KPI Snapshot")
print("Total Sales:", total_sales)
print("Total Units:", total_units)
print("Number of Orders:", num_orders)
print("Avg Order Value:", avg_order_value)
print("Unique States:", unique_states)
print("Median Order Value:", median_order_value)
print("Top Group:", top_group)


# Statewise analysis
if "State" in df_time.columns:
    statewise = df_time.groupby("State").agg({sales_col:"sum", unit_col:"sum"}).reset_index().sort_values(by=sales_col, ascending=False)
    print(statewise.head(10))
    plt.figure(figsize=(10,4))
    plt.bar(statewise["State"].head(10), statewise[sales_col].head(10))
    plt.title("Top 10 States by Sales")
    plt.xlabel("State")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    

# plt.subplot(r,c,7)
# Group analysis
if "Group" in df_time.columns:
    group_summary = df_time.groupby("Group").agg({sales_col:"sum", unit_col:"sum"}).reset_index().sort_values(by=sales_col, ascending=False)
    print(group_summary)
    plt.figure(figsize=(8,4))
    plt.bar(group_summary["Group"], group_summary[sales_col])
    plt.title("Sales by Group")
    plt.xlabel("Group")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.show()
    

# plt.subplot(r,c,8)
# Weekday analysis
df_time["weekday"] = df_time[date_col].dt.day_name()
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
weekday_agg = df_time.groupby("weekday").agg({sales_col:"sum", unit_col:"sum"}).reindex(weekday_order).reset_index()
print(weekday_agg)
plt.figure(figsize=(8,4))
plt.plot(weekday_agg["weekday"], weekday_agg[sales_col], marker="o")
plt.title("Sales by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()
# SETUP
# Library Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# File Ingest
csv_file = 'service_channel_FY24FY25_consolidated_raw_dataset.csv'
df = pd.read_csv(csv_file)

# ------------------------------------------------------------------------------------------------------------
# DATA PREPROCESSING

# Standardize column headers: lowercase and replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(' ', '_')
print("Standardized columns:", df.columns.tolist())

# Display the first few rows of the DataFrame
print(df.head())
print("Columns:", df.columns.tolist())
print("\nDataFrame Info:")
print(df.info())

# ------------------------------------------------------------------------------------------------------------
# INTITTIAL DATA ANALYSIS

# List unique values in the 'equipment' column
print("\nUnique values in 'equipment' column:")
print(df['equipment'].unique())

# Rank the volume of work orders by equipment Type
def count_and_rank_equipment(df):
    equipment_counts = df['equipment'].value_counts()
    print("\nRecord count per equipment type (ranked):")
    print(equipment_counts)

count_and_rank_equipment(df)

# Extract rows where Equipment is "AC/EVAPORATIVE COOLER"
ac_evap_df = df[df['equipment'] == "AC/EVAPORATIVE COOLER"]

print("\nRows with equipment == 'AC/EVAPORATIVE COOLER':")
print(ac_evap_df.head())

# List unique values in the 'region' column
print("\nUnique values in 'region' column:")
print(ac_evap_df['region'].unique())

# Rank the volume of work orders by region Type
def count_and_rank_region(ac_evap_df):
    region_counts = ac_evap_df['region'].value_counts()
    print("\nRecord count per region type (ranked):")
    print(region_counts)

count_and_rank_region(ac_evap_df)

# ------------------------------------------------------------------------------------------------------------
# DRILL DOWN INTO REGIONAL WEEKLY PERFORMANCE

# Ensure the date column is in datetime format (replace 'Date' with your actual date column name)
ac_evap_df['created_date'] = pd.to_datetime(ac_evap_df['created_date'])

# Aggregate weekly work order counts by Region
weekly_counts = ac_evap_df.groupby([pd.Grouper(key='created_date', freq='W'), 'region']).size().reset_index(name='WorkOrderCount')
print("\nWeekly HVAC work order counts by region:")
print(weekly_counts.head())

# Visualization
# Plot weekly HVAC work order counts by region, each region as a subplot
regions = weekly_counts['region'].unique()
num_regions = len(regions)

fig, axes = plt.subplots(num_regions, 1, figsize=(10, 4 * num_regions), sharex=True, sharey=True)
if num_regions == 1:
    axes = [axes]  # Ensure axes is iterable

# Plot each region's data
for i, region in enumerate(sorted(regions)):
    region_data = weekly_counts[weekly_counts['region'] == region]
    axes[i].scatter(region_data['created_date'], region_data['WorkOrderCount'], color='b')
    axes[i].set_title(f"Region: {region}")
    axes[i].set_ylabel('Work Orders')
    axes[i].grid(True)

axes[-1].set_xlabel('Week')
plt.suptitle('Weekly HVAC Work Order Counts by Region', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Save the plot as image file
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("hvac_weekly_work_orders.png")  # Save the plot as an image
print("Plot saved as hvac_weekly_work_orders.png")
# plt.show()  # You can comment this out if running headless

# ------------------------------------------------------------------------------------------------------------
# DRILLING DOWN FURTHER INTO REGIONAL DATA

# Isolate out the MIDSOUTH region data into a new dataframe
midsouth_df = ac_evap_df[ac_evap_df['region'] == '1-MIDSOUTH']
print("\nRows with region == 'MIDSOUTH':")
print(midsouth_df.head())

# Aggregate weekly work order counts by District within the MIDSOUTH region
weekly_midsouth_counts = midsouth_df.groupby([pd.Grouper(key='created_date', freq='W'), 'district']).size().reset_index(name='WorkOrderCount')
print("\nWeekly HVAC work order counts by district in MIDSOUTH region:")
print(weekly_midsouth_counts.head())

# Visualization: Plot weekly HVAC work order counts by district, each district as a subplot
districts = weekly_midsouth_counts['district'].unique()
num_districts = len(districts)

fig, axes = plt.subplots(num_districts, 1, figsize=(10, 4 * num_districts), sharex=True, sharey=True)
if num_districts == 1:
    axes = [axes]  # Ensure axes is iterable

for i, district in enumerate(sorted(districts)):
    district_data = weekly_midsouth_counts[weekly_midsouth_counts['district'] == district]
    axes[i].scatter(district_data['created_date'], district_data['WorkOrderCount'], color='g')
    axes[i].set_title(f"District: {district}")
    axes[i].set_ylabel('Work Orders')
    axes[i].grid(True)

axes[-1].set_xlabel('Week')
plt.suptitle('Weekly HVAC Work Order Counts by District in MIDSOUTH Region', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("midsouth_district_weekly_work_orders.png")
print("Plot saved as midsouth_district_weekly_work_orders.png")

# ------------------------------------------------------------------------------------------------------------
# BASIC HOLT_WINTERS FORECASTING

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- FILTER TO ONLY MIDSOUTH REGION ---
# Standardize region column for matching
ac_evap_df['region'] = ac_evap_df['region'].str.strip().str.upper()
midsouth_df = ac_evap_df[ac_evap_df['region'] == '1-MIDSOUTH']

if midsouth_df.empty:
    print("No data found for region '1-MIDSOUTH' with equipment 'AC/EVAPORATIVE COOLER'.")
else:
    print("\nRows with region == '1-MIDSOUTH':")
    print(midsouth_df.head())

    # Aggregate weekly work order counts by District within the MIDSOUTH region
    weekly_midsouth_counts = midsouth_df.groupby([pd.Grouper(key='created_date', freq='W'), 'district']).size().reset_index(name='WorkOrderCount')
    print("\nWeekly HVAC work order counts by district in MIDSOUTH region:")
    print(weekly_midsouth_counts.head())

    # Aggregate weekly work order counts for the whole MIDSOUTH region
    midsouth_weekly = midsouth_df.groupby(pd.Grouper(key='created_date', freq='W')).size().reset_index(name='WorkOrderCount')
    midsouth_weekly = midsouth_weekly.sort_values('created_date')

    # Holt-Winters Forecast for next week
    y = midsouth_weekly['WorkOrderCount'].values
    if len(y) >= 10:
        model = ExponentialSmoothing(y, trend='add', seasonal=None, initialization_method="estimated")
        fit = model.fit()
        forecast = fit.forecast(1)
        print(f"\nMIDSOUTH Region | Next week's forecast: {forecast[0]:.2f}")
    else:
        print("Not enough data to forecast for MIDSOUTH region.")

    # Visualization: Plot weekly HVAC work order counts for MIDSOUTH region
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    # Set axis and label colors to white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    # Set background to transparent
    plt.gcf().patch.set_alpha(0)
    ax.set_facecolor('none')

    # Scatter Dots - Slightly less neon green
    less_neon_green = '#32FF57'  # Softer than pure neon green
    plt.scatter(midsouth_weekly['created_date'], midsouth_weekly['WorkOrderCount'], color=less_neon_green, label='Actual', zorder=3)
    if len(y) >= 10:
        # Forecast Line - Powder Blue
        plt.plot(midsouth_weekly['created_date'], fit.fittedvalues, color='#B0E0E6', label='Fitted', zorder=2)
        # Forecast Dot - Forest Green
        plt.scatter([midsouth_weekly['created_date'].max() + pd.Timedelta(weeks=1)], [forecast[0]], color='#228B22', label='Forecast', s=100, zorder=4)
    plt.title('Weekly HVAC Work Order Counts - MIDSOUTH Region')
    plt.xlabel('Week')
    plt.ylabel('Work Orders')
    plt.legend(facecolor='none', edgecolor='white', labelcolor='white')

    # Add horizontal grid lines
    ax.yaxis.grid(True, color='white', alpha=0.3, linestyle='--', linewidth=1, zorder=0)

    plt.tight_layout()
    # Remove all spines (the black outline)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.savefig("midsouth_weekly_hvac_forecast.png", transparent=True, bbox_inches='tight', pad_inches=0)
    print("Plot saved as midsouth_weekly_hvac_forecast.png")
    # plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

file_path = '/kaggle/input/turkey-earthquakes-1915-2024-feb/turkey_earthquakes(1915-2024_feb).csv'
earthquakes_df = pd.read_csv(file_path)

earthquakes_df['Olus_datetime'] = pd.to_datetime(earthquakes_df['Olus tarihi'] + ' ' + earthquakes_df['Olus zamani'], errors='coerce')

earthquakes_df.drop(columns=['Olus tarihi', 'Olus zamani'], inplace=True)

features = ['xM', 'MD', 'ML', 'Ms', 'Mb']
target = 'Mw'

train_df = earthquakes_df.dropna(subset=[target])
test_df = earthquakes_df[earthquakes_df[target].isna()]

X_train = train_df[features].fillna(0)
y_train = train_df[target]

model = LinearRegression()
model.fit(X_train, y_train)

X_test = test_df[features].fillna(0)
predicted_mw = model.predict(X_test)

earthquakes_df.loc[earthquakes_df[target].isna(), target] = predicted_mw

remaining_nans_after_fill = earthquakes_df.isna().sum()

plt.style.use('seaborn-darkgrid')

plt.figure(figsize=(10, 6))
plt.hist(earthquakes_df['Mw'], bins=50, edgecolor='k', alpha=0.7)
plt.title('Distribution of Earthquake Magnitudes (Mw)')
plt.xlabel('Magnitude (Mw)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(earthquakes_df['Derinlik'], bins=50, edgecolor='k', alpha=0.7)
plt.title('Distribution of Earthquake Depths')
plt.xlabel('Depth (km)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
earthquakes_df['Olus_datetime'].groupby(earthquakes_df['Olus_datetime'].dt.year).count().plot(kind='bar')
plt.title('Frequency of Earthquakes Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Earthquakes')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(earthquakes_df['Boylam'], earthquakes_df['Enlem'], alpha=0.5, c=earthquakes_df['Mw'], cmap='viridis')
plt.colorbar(label='Magnitude (Mw)')
plt.title('Geographical Distribution of Earthquakes')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

magnitudes = earthquakes_df[['xM', 'MD', 'ML', 'Mw', 'Ms', 'Mb']]
correlation_matrix = magnitudes.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Magnitudes')
plt.show()

plt.figure(figsize=(10, 6))
earthquakes_df['Tip'].value_counts().plot(kind='bar', color='c')
plt.title('Distribution of Earthquakes by Type')
plt.xlabel('Type')
plt.ylabel('Number of Earthquakes')
plt.show()

plt.figure(figsize=(10, 6))
earthquakes_df['Yer'].value_counts().head(10).plot(kind='bar', color='m')
plt.title('Top 10 Locations with the Most Earthquakes')
plt.xlabel('Location')
plt.ylabel('Number of Earthquakes')
plt.xticks(rotation=45, ha='right')
plt.show()

plt.figure(figsize=(10, 6))
earthquakes_df['Month'] = earthquakes_df['Olus_datetime'].dt.to_period('M')
monthly_counts = earthquakes_df['Month'].value_counts().sort_index()
monthly_counts.plot(kind='line', color='b')
plt.title('Monthly Frequency of Earthquakes Over the Years')
plt.xlabel('Month')
plt.ylabel('Number of Earthquakes')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(earthquakes_df['Derinlik'], earthquakes_df['Mw'], alpha=0.5, color='g')
plt.title('Magnitude vs. Depth of Earthquakes')
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude (Mw)')
plt.show()

plt.figure(figsize=(10, 6))
earthquakes_df.boxplot(column='Mw', by='Tip', grid=False, showfliers=False)
plt.title('Magnitude Distribution Over Different Types of Earthquakes')
plt.xlabel('Type')
plt.ylabel('Magnitude (Mw)')
plt.suptitle('')
plt.show()

significant_earthquakes = earthquakes_df[earthquakes_df['Mw'] > 5]

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
turkey = world[world.name == "Turkey"]

gdf = gpd.GeoDataFrame(
    significant_earthquakes, 
    geometry=gpd.points_from_xy(significant_earthquakes['Boylam'], significant_earthquakes['Enlem'])
)

fig, ax = plt.subplots(figsize=(10, 10))
turkey.boundary.plot(ax=ax)
gdf.plot(ax=ax, markersize=significant_earthquakes['Mw']*2, color='red', alpha=0.5, legend=True)
plt.title('Earthquakes in Turkey with Magnitudes Greater Than 5')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

plt.figure(figsize=(12, 12))
ax = plt.axes(projection=ccrs.Mercator())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='aqua')

ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

cities = {
    "Istanbul": (28.9784, 41.0082),
    "Ankara": (32.8597, 39.9334),
    "Izmir": (27.1428, 38.4237),
    "Antalya": (30.7133, 36.8969),
    "Diyarbakir": (40.2189, 37.9136)
}

for city, (lon, lat) in cities.items():
    ax.plot(lon, lat, 'bo', markersize=5, transform=ccrs.PlateCarree())
    plt.text(lon, lat, city, fontsize=12, ha='left', va='center', color='blue', transform=ccrs.PlateCarree())

lons = significant_earthquakes['Boylam'].values
lats = significant_earthquakes['Enlem'].values
magnitudes = significant_earthquakes['Mw'].values

scatter = ax.scatter(lons, lats, c=magnitudes, cmap='hot', alpha=0.6, edgecolors='k', marker='o', s=magnitudes**2*10, transform=ccrs.PlateCarree())

cbar = plt.colorbar(scatter, orientation='vertical', pad=0.1, aspect=50)
cbar.set_label('Magnitude')

plt.title('Earthquakes in Turkey with Magnitudes Greater Than 5', fontsize=16)
plt.show()

significant_earthquakes = earthquakes_df[earthquakes_df['Mw'] >= 5]

plt.figure(figsize=(12, 6))
significant_earthquakes['Year'] = significant_earthquakes['Olus_datetime'].dt.year
significant_earthquakes['Year'].value_counts().sort_index().plot(kind='bar', color='salmon')
plt.title('Frequency of Earthquakes with Magnitudes 5 and Above Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Earthquakes')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(significant_earthquakes['Derinlik'], significant_earthquakes['Mw'], alpha=0.6, color='darkred')
plt.title('Magnitude vs. Depth of Earthquakes with Magnitudes 5 and Above')
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude (Mw)')
plt.show()

earthquakes_df.set_index('Olus_datetime', inplace=True)
earthquakes_monthly = earthquakes_df['Mw'].resample('M').mean().fillna(0)

model = SARIMAX(earthquakes_monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

future_years = pd.date_range(start='2024-03-01', end='2200-12-31', freq='M')
forecast = results.get_forecast(steps=len(future_years))
forecast_mean = forecast.predicted_mean

plt.figure(figsize=(14, 7))
plt.plot(earthquakes_monthly, label='Observed')
plt.plot(future_years, forecast_mean, label='Forecast', color='red')
plt.title('Forecast of Earthquake Magnitudes in Turkey (2024-2200)')
plt.xlabel('Year')
plt.ylabel('Magnitude (Mw)')
plt.legend()
plt.show()

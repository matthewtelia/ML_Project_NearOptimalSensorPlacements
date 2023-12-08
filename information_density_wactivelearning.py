import matplotlib.pyplot as plt
from modAL.density import information_density
from modAL.models import ActiveLearner
from sklearn.datasets import make_blobs
import pandas as pd
import os
from datetime import datetime, timedelta
import requests
import numpy as np
import cartopy.feature as cfeature
import cartopy.crs as ccrs


# %% 
APIkey = "a6f1442a-b753-11ec-b909-0242ac120002"
list_sensor_url =f"https://api.breathelondon.org/api/ListSensors?key={APIkey}"
sensor_list = requests.get(list_sensor_url).json()
#print(sensor_list)


#print(sensor_list[0])
sensors_df = pd.DataFrame(sensor_list[0])
#sensors_df
sensors_df.to_csv(f'sensor_list.csv', encoding='utf-8', index=False)

def get_clarity_data(sensor_code, species, start_time, end_time, averaging):
    sensor_url = f"https://api.breathelondon.org/api/getClarityData/{sensor_code}/{species}/{start_time}/{end_time}/{averaging}?key={APIkey}"
    response = requests.get(sensor_url).json()
    return response

def parse_multiple_csv_files(base_filename,start_index,end_index,columns_to_append):

    padded_start_index = str(start_index).zfill(3)
    first_file_path = f'PM25/{base_filename}{padded_start_index}_IPM25_fulldayhourly.csv'
    # first_file_path = f'PM25/{base_filename}{padded_start_index}_IPM25_fulldayhourly.csv'
    first_df=pd.read_csv(first_file_path)
    
    datetime_col = first_df['DateTime']
    
    PM2_5_df = pd.DataFrame({'DateTime':datetime_col})
    
    for i in range(start_index, end_index + 1):
        padded_index = str(i).zfill(3)
        file_path = f'PM25/{base_filename}{padded_index}_IPM25_fulldayhourly.csv'
        # file_path = f'PM25/{base_filename}{padded_index}_IPM25_fulldayhourly.csv'

        # Read the CSV file into a DataFrame
        if os.path.exists(file_path) and os.path.getsize(file_path) > 2:
            df = pd.read_csv(file_path)
            if not df.empty:
               
                # Extract the desired columns from the DataFrame
                selected_columns = df[columns_to_append[2]]
                sitecode_col = df['SiteCode'].iloc[1]
                selected_columns = selected_columns.rename(sitecode_col)
                
                # Append the selected columns to the result DataFrame
                PM2_5_df = pd.concat([PM2_5_df, selected_columns], axis=1)
                

    return PM2_5_df

base_filename = "CLDP0"
start_index = 1
end_index = 300
sensor_PM25_data_columns = ['SiteCode','DateTime','ScaledValue']

#calling parse function
year_rushhour_25_data = parse_multiple_csv_files(base_filename,start_index,end_index,sensor_PM25_data_columns)

num_columns = 2
ind = 0
average_PM25_seasons = pd.DataFrame()

for i in range(0,num_columns,2):

    #split into seasons

    #spring is march through may
    spring_start ='2022-03-20T07:00:00.000Z'
    spring_end = '2022-06-20T08:00:00.000Z '
    springPM25_data = year_rushhour_25_data[(year_rushhour_25_data['DateTime'] >= spring_start) & (year_rushhour_25_data['DateTime'] <= spring_end)]
    springPM25_data = springPM25_data.reset_index(drop=True)
    
    #summer is june through through august
    summer_start ='2022-06-21T07:00:00.000Z'
    summer_end = '2022-08-20T08:00:00.000Z '
    summerPM25_data = year_rushhour_25_data[(year_rushhour_25_data['DateTime'] >= summer_start) & (year_rushhour_25_data['DateTime'] <= summer_end)]
    summerPM25_data = summerPM25_data.reset_index(drop=True)
    
    #fall is sept through november
    fall_start ='2022-09-21T07:00:00.000Z'
    fall_end = '2022-11-20T08:00:00.000Z '
    fallPM25_data = year_rushhour_25_data[(year_rushhour_25_data['DateTime'] >= fall_start) & (year_rushhour_25_data['DateTime'] <= fall_end)]

    #winter is dec through feburary
    winter_start ='2022-11-21T07:00:00.000Z'
    winter_end = '2023-03-19T08:00:00.000Z '
    winterPM25_data = year_rushhour_25_data[(year_rushhour_25_data['DateTime'] >= winter_start) & (year_rushhour_25_data['DateTime'] <= winter_end)]


for i in range(0,num_columns,2):

    #average value for the seasons
    avg_spring_PM25 = springPM25_data.iloc[:,1:].mean()
    avg_spring_PM25 = pd.concat([avg_spring_PM25, sensors_df.set_index('SiteCode').loc[avg_spring_PM25.index, ['Latitude', 'Longitude']]], axis=1)
    avg_spring_PM25 = avg_spring_PM25.rename(columns={avg_spring_PM25.columns[0]:'Average PM2.5'})
    
    avg_summer_PM25 = summerPM25_data.iloc[:,1:].mean()
    avg_summer_PM25 = pd.concat([avg_summer_PM25, sensors_df.set_index('SiteCode').loc[avg_summer_PM25.index, ['Latitude', 'Longitude']]], axis=1)
    avg_summer_PM25 = avg_summer_PM25.rename(columns={avg_summer_PM25.columns[0]:'Average PM2.5'})    
    
    avg_fall_PM25 = fallPM25_data.iloc[:,1:].mean()
    avg_fall_PM25 = pd.concat([avg_fall_PM25, sensors_df.set_index('SiteCode').loc[avg_fall_PM25.index, ['Latitude', 'Longitude']]], axis=1)
    avg_fall_PM25 = avg_fall_PM25.rename(columns={avg_fall_PM25.columns[0]:'Average PM2.5'}) 
   
    avg_winter_PM25 = winterPM25_data.iloc[:,1:].mean()
    avg_winter_PM25 = pd.concat([avg_winter_PM25, sensors_df.set_index('SiteCode').loc[avg_winter_PM25.index, ['Latitude', 'Longitude']]], axis=1)
    avg_winter_PM25 = avg_winter_PM25.rename(columns={avg_winter_PM25.columns[0]:'Average PM2.5'})
    
    avg_year_PM25 = year_rushhour_25_data.iloc[:,1:].mean()
    avg_year_PM25 = pd.concat([avg_year_PM25, sensors_df.set_index('SiteCode').loc[avg_year_PM25.index, ['Latitude', 'Longitude']]], axis=1)
    avg_year_PM25 = avg_year_PM25.rename(columns={avg_year_PM25.columns[0]:'Average PM2.5'})    
    
   
    #average value for the seasons
    max_spring_PM25 = springPM25_data.iloc[:,1:].max()
    max_spring_PM25 = pd.concat([max_spring_PM25, sensors_df.set_index('SiteCode').loc[max_spring_PM25.index, ['Latitude', 'Longitude']]], axis=1)
    max_spring_PM25 = max_spring_PM25.rename(columns={max_spring_PM25.columns[0]:'Max PM2.5'})
    
    max_summer_PM25 = summerPM25_data.iloc[:,1:].max()
    max_summer_PM25 = pd.concat([max_summer_PM25, sensors_df.set_index('SiteCode').loc[max_summer_PM25.index, ['Latitude', 'Longitude']]], axis=1)
    max_summer_PM25 = max_summer_PM25.rename(columns={max_summer_PM25.columns[0]:'Max PM2.5'})
    
    max_fall_PM25 = fallPM25_data.iloc[:,1:].max()
    max_fall_PM25 = pd.concat([max_fall_PM25, sensors_df.set_index('SiteCode').loc[max_fall_PM25.index, ['Latitude', 'Longitude']]], axis=1)
    max_fall_PM25 = max_fall_PM25.rename(columns={max_fall_PM25.columns[0]:'Max PM2.5'})
    
    max_winter_PM25 = winterPM25_data.iloc[:,1:].max()
    max_winter_PM25 = pd.concat([max_winter_PM25, sensors_df.set_index('SiteCode').loc[max_winter_PM25.index, ['Latitude', 'Longitude']]], axis=1)
    max_winter_PM25 = max_winter_PM25.rename(columns={max_winter_PM25.columns[0]:'Max PM2.5'})
    
    max_year_PM25 = year_rushhour_25_data.iloc[:,1:].max()
    max_year_PM25 = pd.concat([max_year_PM25, sensors_df.set_index('SiteCode').loc[max_year_PM25.index, ['Latitude', 'Longitude']]], axis=1)
    max_year_PM25 = max_year_PM25.rename(columns={max_year_PM25.columns[0]:'Max PM2.5'})
    
   
#generating list of sensors by site classification    
urban_background_list = sensors_df[sensors_df.SiteClassification=="Urban Background"]["SiteCode"].tolist()
road_side_list = sensors_df[sensors_df.SiteClassification=="Roadside"]["SiteCode"].tolist()
kerb_side_list = sensors_df[sensors_df.SiteClassification=="Kerbside"]["SiteCode"].tolist()

# fig.show()
def plot_pm_data(longitude,latitude,pm25):
    fig = plt.figure()
    plt.scatter(longitude, latitude, c=pm25, cmap='viridis')
    plt.colorbar()
    plt.show()
    
#plot average data
plot_pm_data(avg_year_PM25['Longitude'], avg_year_PM25['Latitude'], avg_year_PM25['Average PM2.5'])
# plot_pm_data(avg_spring_PM25['Longitude'], avg_spring_PM25['Latitude'], avg_spring_PM25['Average PM2.5'])
# plot_pm_data(avg_summer_PM25['Longitude'], avg_summer_PM25['Latitude'], avg_summer_PM25['Average PM2.5'])
# plot_pm_data(avg_fall_PM25['Longitude'], avg_fall_PM25['Latitude'], avg_fall_PM25['Average PM2.5'])
# plot_pm_data(avg_winter_PM25['Longitude'], avg_winter_PM25['Latitude'], avg_winter_PM25['Average PM2.5'])

plot_pm_data(max_year_PM25['Longitude'], max_year_PM25['Latitude'], max_year_PM25['Max PM2.5'])


# %% Original example

X, y = make_blobs(n_features=2, n_samples=1000, centers=3, random_state=0, cluster_std=0.7)


cosine_density = information_density(X, 'cosine')
euclidean_density = information_density(X, 'euclidean')

# visualizing the cosine and euclidean information density
with plt.style.context('ggplot'):
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.scatter(x=X[:, 0], y=X[:, 1], c=cosine_density, cmap='viridis', s=50)
    plt.title('The cosine information density')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.scatter(x=X[:, 0], y=X[:, 1], c=euclidean_density, cmap='viridis', s=50)
    plt.title('The euclidean information density')
    plt.colorbar()
    plt.show()

#plotting breath london data

# Calculate Euclidean density for the existing sensor coordinates
coordinates = np.array([avg_year_PM25['Longitude'].values, avg_year_PM25['Latitude'].values]).T
sensor_euclidean_density = information_density(coordinates, 'euclidean')


min_lon = min(avg_year_PM25['Longitude'].values)
min_lat = min(avg_year_PM25['Latitude'].values)
max_lon = max(avg_year_PM25['Longitude'].values)
max_lat = max(avg_year_PM25['Latitude'].values)

aspect_ratio = (max_lon - min_lon) / (max_lat - min_lat)

# Plot the existing sensor placement with Euclidean density
with plt.style.context('ggplot'):
    plt.figure(figsize=(14, 7))
    plt.scatter(avg_year_PM25['Longitude'], avg_year_PM25['Latitude'], c=sensor_euclidean_density, cmap='viridis', s=50)
    plt.title('Existing Sensor Placement with Euclidean Density')
    plt.colorbar()
    plt.show()
    

with plt.style.context('ggplot'):
    # plt.figure(figsize=(14, 12))
    # plt.subplot(1, 2, 1)
    # plt.scatter(avg_year_PM25['Longitude'], avg_year_PM25['Latitude'], c=sensor_cosine_density, cmap='viridis', s=50)
    # plt.title('The cosine information density')
    # plt.colorbar()
    fig, ax = plt.subplots(subplot_kw = {'projection': ccrs.PlateCarree()})
        
    ax.set_extent([min_lon - 0.1, max_lon + 0.1, min_lat - 0.1, max_lat + 0.1])
    ax.set_aspect(aspect_ratio)

    
    # ax.coastlines(resolution='50m', linewidth=1)
    # ax.add_feature(ccrs.feature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black'))
    # ax.add_wmts("http://maps.openweathermap.org/maps/2.0/weather/TA2/{z}/{x}/{y}?appid={API_KEY}", layer="clouds")
    ax.add_feature(cfeature.RIVERS, edgecolor='blue')
        
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Overlaying Data on a Map using Cartopy')
    plt.show()
    

# %%


# Active learning for optimized sensor placement
# Initial labeled data (randomly select a few points)
initial_idx = np.random.choice(range(len(coordinates)), size=5, replace=False)
X_initial = coordinates[initial_idx]
y_initial = sensor_euclidean_density[initial_idx]

# Define the Euclidean density query strategy
def query_strategy(classifier, X, y):
    query_idx, query_instance = max(
        enumerate(X),
        key=lambda x: information_density(np.array(x).reshape(1, -1), 'euclidean')
    )
    return query_idx, query_instance

# Create the active learner
learner = ActiveLearner(
    estimator=None,  # You need to specify your model for estimation
    X_training=X_initial,
    y_training=y_initial,
    query_strategy=query_strategy
)

# Number of queries to make
n_queries = 10

# Active learning loop
for _ in range(n_queries):
    query_idx, query_instance = learner.query(coordinates)
    learner.teach(coordinates[query_idx], sensor_euclidean_density[query_idx])

# Plotting the results
with plt.style.context('ggplot'):
    plt.figure(figsize=(14, 7))
    plt.scatter(avg_year_PM25['Longitude'], avg_year_PM25['Latitude'], c=sensor_euclidean_density, cmap='viridis', s=50)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='red', marker='x', label='Selected Sensors')
    plt.title('Optimized Sensor Placement with Active Learning')
    plt.colorbar()
    plt.legend()
    plt.show()

import math
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import pandas as pd
import numpy as np 

point_colors = [
    'red',
    'blue',
    'gray',
    'darkred',
    'darkpurple',
    'orange',
    'beige',
    'green',
    'black'
    'darkgreen',
    'lightred',
    'lightgreen',
    'darkblue',
    'pink',
    'lightblue',
    'purple',
    'cadetblue',
    'lightgray',
]




def print_boxplots(data):
    number_of_plots=data.shape[1]
    number_of_columns=5
    number_of_rows=math.ceil(number_of_plots/number_of_columns)
    columns=list(data.columns)
    fig, axes = plt.subplots(number_of_rows,number_of_columns, figsize=(25, 3*number_of_rows))
    plt.subplots_adjust(hspace = 0.5)
    fig.suptitle('DATSET BOX PLOTS')
   

    for i in range(0,number_of_rows):
        for j in range(0,number_of_columns):
            index=i*number_of_columns+j
            if(index<number_of_plots):
                sns.boxplot(ax=axes[i, j], data=data, x=columns[index])
            else:
                fig.delaxes(axes[i][j])



def make_map(dataframe,lat=48.750,long=2.25,width=1300, height=800,mag=9,number_of_points=-1):
    f = folium.Figure(width=width, height=height)
    map_ = folium.Map(location=[lat, long], zoom_start=mag,control_scale=True,zoom_control=False).add_to(f)

    if(number_of_points<0 or number_of_points>dataframe.shape[0]):
        number_of_points=dataframe.shape[0]
    
    gps_spots = dataframe[['lat','long']].values.tolist()
   
    for point in range(0, number_of_points):
        folium.Marker([gps_spots[point][0],gps_spots[point][1]]).add_to(map_)
        
    return map_


def draw_clusters(dataframe, lat=48.750, long=2.25, width=1300, height=800, mag=9):
    f = folium.Figure(width=1300, height=800)
    map_ = folium.Map(location=[lat, long], zoom_start=mag,control_scale=True,zoom_control=False).add_to(f)
    cluster_indexes=list(dataframe['cluster'].unique())
    cluster_indexes.remove(-1)
    
    for index in cluster_indexes:
        (dataframe.iloc[np.where((dataframe['cluster']==index))]).apply(lambda row: folium.Marker([row['lat'],row['long']],icon=folium.Icon(color=point_colors[index%len(point_colors)])).add_to(map_),axis=1)

    return map_

def filter_clusters_by_roadID(dataframe,min_cluster_size):
    clusters=dataframe['cluster'].unique()
    free_cluster=clusters.max()+1
  
    for cluster in clusters:
        streets=dataframe.loc[dataframe['cluster'] ==cluster]['voie'].unique()
        clu_numbers=list(range(free_cluster,free_cluster+len(streets)-1))
        clu_numbers.append(cluster)
        free_cluster+=len(streets)-1
        street_dict=dict(zip(streets,clu_numbers))
        for street, c in street_dict.items():
            dataframe['cluster']=np.where((dataframe['cluster']==cluster) & (dataframe['voie']==street), c, dataframe['cluster'])
        
    drop_cluster_dict=dataframe['cluster'].value_counts().ge(min_cluster_size).to_dict()
    dataframe=dataframe[dataframe['cluster'].replace(drop_cluster_dict).to_list()]

    return dataframe

def convert_column_to_string(dataframe,columns):
    for column in columns:
        dataframe[column]=dataframe[column].astype('int32').astype('str')
import math
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA

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

lum_dict = {
    '1': 'Full day',
    '2': 'Twilight or dawn',
    '3': 'Night without public lighting',
    '4': 'Night with public lighting not lit',
    '5': 'Night with public lighting on'
}

col_dict = {
    '1': 'Two vehicles - frontal',
    '2': 'Two vehicles - from the rear',
    '3': 'Two vehicles - by the side',
    '4': 'Three vehicles and more - in chain',
    '5': 'Three or more vehicles - multiple collisions',
    '6': 'Other collision',
    '7': 'Without collision'
}


surf_dict = {
    '1': 'Normal',
    '2': 'Wet',
    '3': 'Puddles',
    '4': 'Flooded',
    '5': 'Snow',
    '6': 'Mud',
    '7': 'Icy',
    '8': 'Fat - oil',
    '9': 'Other'
}

atm_dict = {
    '1': 'Normal',
    '2': 'Light rain',
    '3': 'Heavy rain',
    '4': 'Snow - hail',
    '5': 'Fog - smoke',
    '6': 'Strong wind - storm',
    '7': 'Dazzling weather',
    '8': 'Cloudy weather',
    '9': 'Other'
}

lum_icon = {
    '1': ['certificate', 'orange', '#F8FF00'],
    '2': ['sun-o', 'orange', '#F8FF00'],
    '3': ['moon-o', 'black', '#D1D1C0'],
    '4': ['lightbulb-o', 'black', '#D1D1C0'],
    '5': ['lightbulb-o', 'black', '#F8FF00']
}

atm_icon = {
    '1': ['minus', 'lightgray', '#009C17'],
    '2': ['umbrella', 'blue', '#001EB8'],
    '3': ['bolt', 'darkblue', '#F8FF00'],
    '4': ['asterisk', 'cadetblue', '#FFFFFF'],
    '5': ['eye-slash', 'darkred', '#CCCCCC'],
    '6': ['leaf', 'darkpurple', '#CCCCCC'],
    '7': ['certificate', 'orange', '#F8FF00'],
    '8': ['cloud', 'lightgreen', '#FFFFFF'],
    '9': ['question-circle', 'pink', '#FFFFFF'],
}

col_icon = {
    '1': ['arrow-left', 'orange', '#FFFFFF'],
    '2': ['arrow-right', 'orange', '#FFFFFF'],
    '3': ['arrow-down', 'orange', '#FFFFFF'],
    '4': ['link', 'red', '#FFFFFF'],
    '5': ['arrows-alt', 'red', '#FFFFFF'],
    '6': ['question', 'purple', '#FFFFFF'],
    '7': ['ban', 'green', '#FFFFFF']
}

surf_icon = {
    '1': ['check-circle-o', 'green', '#FFFFFF'],
    '2': ['tint', 'lightblue', '#000000'],
    '3': ['tint', 'blue', '#000000'],
    '4': ['tint', 'darkblue', '#000000'],
    '5': ['asterisk', 'cadetblue', '#FFFFFF'],
    '6': ['circle', 'beige', '#552D00'],
    '7': ['cube', 'lightgray', '#FFFFFF'],
    '8': ['eyedropper', 'darkgreen', '#FFFFFF'],
    '9': ['question', 'purple', '#FFFFFF']
}


def print_boxplots(data):
    number_of_plots = data.shape[1]
    number_of_columns = 5
    number_of_rows = math.ceil(number_of_plots/number_of_columns)
    columns = list(data.columns)
    fig, axes = plt.subplots(number_of_rows, number_of_columns, figsize=(25, 3*number_of_rows))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle('DATSET BOX PLOTS')

    for i in range(0, number_of_rows):
        for j in range(0, number_of_columns):
            index = i*number_of_columns+j
            if(index < number_of_plots):
                sns.boxplot(ax=axes[i, j], data=data, x=columns[index])
            else:
                fig.delaxes(axes[i][j])


def make_map(dataframe, lat=48.750, long=2.25, width=1300, height=800, mag=9, number_of_points=-1, color='black', icon_color='#FFFFFF', icon='circle', prefix='fa'):
    f = folium.Figure(width=width, height=height)
    map_ = folium.Map(location=[lat, long], zoom_start=mag,control_scale=True, zoom_control=False).add_to(f)

    if(number_of_points < 0 or number_of_points > dataframe.shape[0]):
        number_of_points = dataframe.shape[0]

    gps_spots = dataframe[['lat', 'long']].values.tolist()

    for point in range(0, number_of_points):
        folium.Marker([gps_spots[point][0], gps_spots[point][1]], icon=folium.Icon(color=color, icon_color=icon_color, icon=icon, prefix=prefix)).add_to(map_)

    return map_


def draw_clusters(dataframe, lat=48.750, long=2.25, width=1300, height=800, mag=9):
    f = folium.Figure(width=1300, height=800)
    map_ = folium.Map(location=[lat, long], zoom_start=mag,control_scale=True, zoom_control=False).add_to(f)
    cluster_indexes = list(dataframe['cluster'].unique())
    cluster_indexes.remove(-1)

    for index in cluster_indexes:
        (dataframe.iloc[np.where((dataframe['cluster'] == index))]).apply(lambda row: folium.Marker([row['lat'], row['long']], icon=folium.Icon(color=point_colors[index % len(point_colors)])).add_to(map_), axis=1)

    return map_


def filter_clusters_by_roadID(dataframe, min_cluster_size):
    clusters = dataframe['cluster'].unique()
    clusters = clusters[clusters != -1]
    free_cluster = clusters.max()+1

    for cluster in clusters:
        streets = dataframe.loc[dataframe['cluster']== cluster]['voie'].unique()
        clu_numbers = list(range(free_cluster, free_cluster+len(streets)-1))
        clu_numbers.append(cluster)
        free_cluster += len(streets)-1
        street_dict = dict(zip(streets, clu_numbers))
        for street, c in street_dict.items():
            dataframe['cluster'] = np.where((dataframe['cluster'] == cluster) & (dataframe['voie'] == street), c, dataframe['cluster'])

    drop_cluster_dict = dataframe['cluster'].value_counts().ge(min_cluster_size).to_dict()
    dataframe = dataframe[dataframe['cluster'].replace(drop_cluster_dict).to_list()]

    return dataframe


def convert_column_to_string(dataframe, columns):
    for column in columns:
        dataframe[column] = dataframe[column].astype('int32').astype('str')


def filter(val, base, threshold):
    if(val > base+threshold):
        return 1
    if(val < base-threshold):
        return -1
    return 0


def get_dictionary(attribute):
    if(attribute == 'lum'):
        return lum_dict
    if(attribute == 'col'):
        return col_dict
    if(attribute == 'surf'):
        return surf_dict
    return atm_dict


def analyse_attributes_values_in_clusters(dataframe, attribute, threshold=0.18):
    cluster_dict = dataframe.groupby(['cluster']).count().iloc[:, 0].to_dict()
    names = get_dictionary(attribute)
    frequences = []
    unique_attribute_values = dataframe[attribute].unique()

    for cluster in list(cluster_dict.keys()):
        cluster_values = np.zeros(int(max(unique_attribute_values))+1)
        cluster_values[0] = cluster
        values_dict = dataframe.groupby(['cluster', attribute]).agg({attribute: ['count']}).loc[cluster].T.to_dict('records')[0]

        for index, value in values_dict.items():
            cluster_values[int(index)] = (value/cluster_dict[cluster])

        frequences.append(cluster_values)

    freq_df = pd.DataFrame(frequences)
    freq_df=freq_df.loc[:, (freq_df != 0).any(axis=0)]
    freq_df.columns=['cluster']+sorted(list(unique_attribute_values))

    freq_df['cluster'] = freq_df['cluster'].astype('int32')
    freq_df.set_index('cluster', inplace=True)

    total_freq_dict = dataframe.groupby([attribute]).agg({attribute: ['count']}).div(len(dataframe.index)).T.to_dict('records')[0]

    anomalies_data = pd.DataFrame(index=freq_df.index, columns=freq_df.columns)
    for column in freq_df.columns:
        anomalies_data[column] = freq_df[column].apply(lambda x: filter(x, total_freq_dict[column], threshold))

    number_of_rows = len(unique_attribute_values)
    fig, axs = plt.subplots(number_of_rows, figsize=(25, number_of_rows*5))
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle(attribute, fontsize=26)

    plot_index = 0
    for index, value in total_freq_dict.items():
        freq_df[index].plot(ax=axs[plot_index], kind='bar', rot=0)
        axs[plot_index].axhline(min(1, value+threshold),c="red", linewidth=1, ls='--')
        axs[plot_index].axhline(max(value-threshold, 0),c="red", linewidth=1, ls='--')
        axs[plot_index].axhline(value, c="green", linewidth=2, ls='-')
        axs[plot_index].set_title(f'Attribute value:{index}     Attribute name:{names[index]}     Probability: {round(value,4)}', fontsize=16)
        axs[plot_index].set_xlabel('Cluster number', fontsize=12)
        axs[plot_index].set_ylabel('Probability', fontsize=12)

        plot_index += 1

    return anomalies_data


def get_icon_dict(attribute):
    if(attribute == 'lum'):
        return lum_icon
    if(attribute == 'col'):
        return col_icon
    if(attribute == 'surf'):
        return surf_icon
    return atm_icon


def create_tooltip_text(row):
    day = row['jour']
    month = row['mois']
    year = row['an']
    return (f'Date: {day}.{month}.20{year}')


def add_frequent_marker(row, attribute, anomalies_dataframe, map_,):
    cluster = row['cluster']
    attribute_value = str(row[attribute])
    anomaly = anomalies_dataframe.loc[cluster, attribute_value]
    if(anomaly == 1):
        icon_dict = get_icon_dict(attribute)
        folium.Marker([row['lat'], row['long']], icon=folium.Icon(icon=icon_dict[attribute_value][0], color=icon_dict[attribute_value][1], icon_color=icon_dict[attribute_value][2], prefix='fa'), tooltip=folium.Tooltip(create_tooltip_text(row))).add_to(map_)


def add_less_frequent_marker(lat, long, attribute, attribute_value, map_, cluster_number):
    icon_dict = get_icon_dict(attribute)
    folium.Marker([lat, long], icon=folium.Icon(icon=icon_dict[attribute_value][0], color='pink',icon_color=icon_dict[attribute_value][2], prefix='fa'), tooltip=folium.Tooltip(f'Cluster no: {cluster_number}')).add_to(map_)


def draw_more_frequent_anomalies_clusters(dataframe, anomalies_dataframe, attribute, lat=48.750, long=2.25, width=1300, height=800, mag=9):
    f = folium.Figure(width=1300, height=800)
    map_ = folium.Map(location=[lat, long], zoom_start=mag,control_scale=True, zoom_control=False).add_to(f)

    dataframe.loc[dataframe['cluster'] > -1].apply(lambda row: add_frequent_marker(row, attribute, anomalies_dataframe, map_,), axis=1)

    return map_


def draw_less_frequent_anomalies_clusters(dataframe, anomalies_dataframe, attribute, lat=48.750, long=2.25, width=1300, height=800, mag=9):
    f = folium.Figure(width=1300, height=800)
    map_ = folium.Map(location=[lat, long], zoom_start=mag,control_scale=True, zoom_control=False).add_to(f)

    for column in anomalies_dataframe.columns:
        clusters = anomalies_dataframe[anomalies_dataframe[column] == -1].index
        for cluster in clusters:
          
            data_ser = dataframe.loc[dataframe['cluster']== cluster].iloc[int(column)]
            add_less_frequent_marker(data_ser['lat'], data_ser['long'], attribute, column, map_, cluster)

    return map_


def plot_dendrogram(model, **kwargs):
    plt.rcParams["figure.figsize"] = (20,12)
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
                counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)

def plot_3D(data,labels):
    pca_ = PCA(n_components=3)
    X = pca_.fit_transform(data)

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2], c=labels/5, cmap='viridis',edgecolor='k', s=100, alpha = 0.8)


    ax.set_title("PCA directions")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.dist = 10
    plt.autoscale(enable=True, axis='x', tight=True)    

    plt.show()

def plot_feature_importances(importances,columns):
    plt.figure(figsize=(20,15))
    plt.barh([x for x in range(len(importances))], importances)
    plt.yticks(range(0,len(columns)-1),columns)
    plt.show()
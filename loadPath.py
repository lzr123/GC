import pandas as pd
import numpy as np
import json as js
import matplotlib.pyplot as plt
import requests
from NodeAndLink import *
import warnings

warnings.filterwarnings('ignore')

STD_PATH_FILE_PATH = './stdPath.csv'
DATA_FILE_PATH = './bus data.csv.xlsx'
STD_PATH_FILE_PATH_BAIDU = './std_path_m.csv'
DATA_FILE_PATH_BAIDU = './Source_data_baidu.csv'
STATION_DATA_PATH = './stations.csv'

API_URL = {'FORMAT_TRANS': 'http://api.map.baidu.com/geoconv/v1/?',
             'CORRECTION': 'http://yingyan.baidu.com/api/v3/track/getlatestpoint'}

API_KEY = 'a53lVKSRdvwoYmaVv7Dafe3Zx2nkFODD'

SOURCE_FORMAT = {  'WGS84': 1,
                  'SOUGOU': 2,
                   'GCJ02': 3,
                 'GCJ02_M': 4,
                  'BD09LL': 5,
                  'BD09MC': 6,
                  'MAPBAR': 7,
                   '51MAP': 8}

TARGET_FORMAT = {  'GCJ02': 3,
                 'GCJ02_M': 4,
                  'BD09LL': 5,
                  'BD09MC': 6}

def transFormatUsingAPI(url, lng, lat, point_count, src_format = 'WGS84', tgt_format='BD09LL'):
    longitude = []
    latitude = []
    i = 0
    while True:
        coord_peers = ''
        if i+100 > point_count:
            tail = point_count
        else:
            tail = i+100
        for x, y in zip(lng.iloc[i:tail], lat.iloc[i:tail]):
            coord_peers += (str(x) + ',' + str(y) + ';')
        coord_peers = coord_peers[:-1]
        params = {  'ak' : API_KEY,
                 'coords': coord_peers,
                  'from' : SOURCE_FORMAT[src_format],
                    'to' : TARGET_FORMAT[tgt_format],
                 'output': 'json'}
        result = requests.get(url, params=params)
        r = js.loads(result.text)
        for item in r['result']:
            longitude.append(item['x'])
            latitude.append(item['y'])
        i += 100
        if i > point_count:
            break
    return longitude, latitude

def trackCorrectionBAIDU(lng_src, lat_src, lng_src_m, lat_src_m):
    point_count = len(lng_src)
    i = 0
    while True:
        coord_peers = ''
        if i+100 > point_count:
            tail = point_count
        else:
            tail = i+100

        for x, y in zip(lng.iloc[i:tail], lat.iloc[i:tail]):
            coord_peers += (str(x) + ',' + str(y) + ';')

def linearizationLine(line_lng, line_lat, line_lng_m, line_lat_m, start_station, end_station):
    node_list = []
    link_list = []
    for x, y, x_m, y_m in zip(line_lng, line_lat, line_lng_m, line_lat_m):
        new_node = Node(x, y, x_m, y_m)
        node_list.append(new_node)
    for i in range(len(node_list) - 1):

        if not (node_list[i] == node_list[i+1]):
            new_link = Link(node_list[i], node_list[i + 1])
            link_list.append(new_link)

    line = Line(start_station, end_station, node_list, link_list)
    return line

def calcDistance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calcTerminalPos(lng_src_m, lat_src_m, lng_sta_m, lat_sta_m, station_count, threshold=50):
    station_info = np.empty([station_count, 2])
    for i in range(station_count):
        station_info[i, 0] = lng_sta_m.iloc[i]
        station_info[i, 1] = lat_sta_m.iloc[i]

    lng_arrive = []
    lat_arrive = []
    for x, y in zip(lng_src_m, lat_src_m):
        for j in range(station_count):
            distance = calcDistance((x, y), (station_info[j, 0], station_info[j, 1]))
            if distance <= threshold:
                lng_arrive.append(x)
                lat_arrive.append(y)
    return lng_arrive, lat_arrive



if __name__  == '__main__':
    # std_path = pd.read_csv(STD_PATH_FILE_PATH)
    #
    # lng_std = std_path.lng
    # lat_std = std_path.lat
    #
    # source_data = pd.read_excel(DATA_FILE_PATH)
    # source_data = source_data[(source_data['lng'] > 0) & (source_data['lat'] > 0)]
    # lng_src = source_data.lng
    # lat_src = source_data.lat
    # door_state = source_data.DoorState
    #
    # lng_src_baidu, lat_src_baidu = transFormatUsingAPI(API_URL['FORMAT_TRANS'], lng_src, lat_src, lng_src.shape[0],
    #                                                    src_format='WGS84', tgt_format='BD09LL')
    # lng_src_baidu_m, lat_src_baidu_m = transFormatUsingAPI(API_URL['FORMAT_TRANS'], lng_src, lat_src,
    #                                                        lng_src.shape[0], src_format='WGS84', tgt_format='BD09MC')
    # lng_std_m, lat_std_m = transFormatUsingAPI(API_URL['FORMAT_TRANS'], lng_std, lat_std,
    #                                            lng_std.shape[0], src_format='BD09LL', tgt_format='BD09MC')
    #
    #
    # source_data_baidu = source_data.copy()
    #
    # source_data_baidu['Lng_Baidu'] = lng_src_baidu.copy()
    # source_data_baidu['Lat_Baidu'] = lat_src_baidu.copy()
    #
    # source_data_baidu['Lng_Baidu_m'] = lng_src_baidu_m.copy()
    # source_data_baidu['Lat_Baidu_m'] = lat_src_baidu_m.copy()
    #
    # std_path['lng_m'] = lng_std_m.copy()
    # std_path['lat_m'] = lat_std_m.copy()
    #
    # source_data_baidu.to_csv('./Source_data_baidu.csv')
    # std_path.to_csv('./std_path_m.csv')

    # stations = pd.read_csv(STATION_DATA_PATH)
    # lng_sat_baidu = stations.lng
    # lat_sat_baidu = stations.lat
    #
    #
    # lng_sat_m, lat_sat_m = transFormatUsingAPI(API_URL['FORMAT_TRANS'], lng_sat_baidu, lat_sat_baidu,
    #                                            lng_sat_baidu.shape[0], src_format='BD09LL', tgt_format='BD09MC')
    #
    # stations['lng_m'] = lng_sat_m.copy()
    # stations['lat_m'] = lat_sat_m.copy()
    #
    # stations.to_csv(STATION_DATA_PATH)


    std_path = pd.read_csv(STD_PATH_FILE_PATH_BAIDU).iloc[:-1]

    lng_std_m = std_path['lng_m'].copy()
    lat_std_m = std_path['lat_m'].copy()

    lng_std = std_path['lng'].copy()
    lat_std = std_path['lat'].copy()

    source_data = pd.read_csv(DATA_FILE_PATH_BAIDU)
    lng_src_m = source_data['lng_m'].copy()
    lat_src_m = source_data['lat_m'].copy()

    lng_src_baidu = source_data['lng_Baidu'].copy()
    lat_src_baidu = source_data['lat_Baidu'].copy()

    stations = pd.read_csv(STATION_DATA_PATH).iloc[:-1]
    lng_sta = stations['lng'].copy()
    lat_sta = stations['lat'].copy()

    lng_sta_m = stations['lng_m'].copy()
    lat_sta_m = stations['lat_m'].copy()

    sta_name = stations['name'].copy()

    start_station = Station(sta_name.iloc[0], lng_sta.iloc[0], lat_sta.iloc[0],
                            lng_sta_m.iloc[0], lat_sta_m.iloc[0], arrive_terminal=False)
    end_station = Station(sta_name.iloc[-1], lng_sta.iloc[-1], lat_sta.iloc[-1],
                          lng_sta_m.iloc[-1], lat_sta_m.iloc[-1], arrive_terminal=True)

    line = linearizationLine(lng_std, lat_std, lng_std_m, lat_std_m, start_station, end_station)

    station_count = stations.shape[0]
    lng_arv, lat_arv = calcTerminalPos(lng_src_m, lat_src_m, lng_sta_m, lat_sta_m, station_count, 50)

    round_data = line.splitRound(lng_src_baidu, lat_src_baidu, lng_src_m, lat_src_m)

    corr_track = line.trackCorrection(lng_src_baidu, lat_src_baidu, lng_src_m, lat_src_m)

    for index in range(8):
        plt.axis('equal')
        plt.subplot(121)
        plt.plot(lng_std_m, lat_std_m, 'o-')
        plt.plot(round_data['lng_m'][index], round_data['lat_m'][index], 'gx')
        plt.grid()
        plt.plot(corr_track['lng_m'][index], corr_track['lat_m'][index], 'rv-')

        plt.subplot(122)
        plt.plot(lng_std, lat_std, 'o-')
        # plt.plot(lng_src_baidu, lat_src_baidu)
        plt.plot(lng_sta, lat_sta, '*')

        plt.show()
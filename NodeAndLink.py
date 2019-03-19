import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

class Node():
    def __init__(self, lng, lat, lng_m, lat_m, arrive_terminal=False):
        self.lng = lng
        self.lat = lat
        self.lng_m = lng_m
        self.lat_m = lat_m
        self.arrive_terminal = arrive_terminal

    def __str__(self):
        return 'Node:\n\tcoord: (%.8f, %.8f)\n\tcoord_m: (%d, %d)' % (self.lng, self.lat, self.lng_m. self.lat_m)

    def __repr__(self):
        return f'Node(coord:({self.lng},{self.lat}), coord_m({self.lng_m},{self.lat_m}))'

    def __eq__(self, n):
        if (self.lng == n.lng) and (self.lat == n.lat) and (self.lng_m == n.lng_m) and (self.lat_m == n.lat_m):
            return True
        else:
            return False

class Station(Node):
    def __init__(self, name, lng, lat, lng_m, lat_m, arrive_terminal=False):
        super().__init__(lng, lat, lng_m, lat_m, arrive_terminal)
        self.name = name


class Link():
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.length = self.getLength()
        # self.direction = self.getDirection()
        # self._theta = self.polyfit()

    def __str__(self):
        string = 'Link[(%f, %f), (%f, %f)]' % (self.upstream_node.lng_m, self.upstream_node.lat_m,
                                               self.downstream_node.lng_m, self.downstream_node.lat_m)
        return string




    def getLength(self):
        lng_def = self.upstream_node.lng_m - self.downstream_node.lng_m
        lat_def = self.upstream_node.lat_m - self.downstream_node.lat_m
        length = np.sqrt(lng_def**2 + lat_def**2)
        return length

    # def getDirection(self):
    #     lng_def = self.upstream_node.lng_m - self.downstream_node.lng_m
    #     lat_def = self.upstream_node.lat_m - self.downstream_node.lat_m
    #     vector = np.array([lng_def, lat_def]) / self.length
    #     return vector
    #
    # def polyfit(self):
    #     x = np.array([self.upstream_node.lng_m, self.downstream_node.lng_m])
    #     y = np.array([self.upstream_node.lat_m, self.downstream_node.lat_m])
    #     theta = np.polyfit(x, y, deg=1)
    #     theta = np.array([theta[0], -1, theta[1]])
    #     return theta

class Line():
    def __init__(self, start_station, end_station, node_list, link_list):
        self.start_station = start_station
        self.end_station = end_station
        self.node_list = node_list
        self.link_list = link_list

    def __evalDistanceToStation(self, x_m, y_m, station):
        distance = np.sqrt((station.lng_m - x_m)**2 + (station.lat_m - y_m)**2)
        return distance

    def __evalDistanceToLink(self, x_m, y_m, link):
        point = np.array([x_m, y_m, 1], dtype=np.float64)
        distance = np.abs(np.dot(point, link._theta.T)) / np.sqrt(link._theta[0]**2 + link._theta[1]**2)
        return distance

    def __evalProjPoint(self, x_m, y_m, link):
        x1 = link.upstream_node.lng_m
        y1 = link.upstream_node.lat_m

        x2 = link.downstream_node.lng_m
        y2 = link.downstream_node.lat_m

        A = np.array([[x2 - x1, -(y1 - y2)],
                      [y2 - y1, -(x2 - x1)]])

        b = np.array([x_m - x1, y_m - y1])

        r = np.linalg.solve(A, b)
        t = r[0]

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        p0 = np.array([x, y])

        return p0



    def __pointInRange(self, p1, p2, p0):
        v1 = p2 - p1
        v2 = p0 - p1
        if np.cross(v1, v2) < 1e-14 and v1.dot(v2) >= 0:
            return True
        else:
            return False


    def splitRound(self, lng_src, lat_src, lng_src_m, lat_src_m, threshold=50):
        fsm_state = 0
        first_rec = False
        round_data = []
        round_point = {'lng': [], 'lat': [], 'lng_m': [], 'lat_m': []}
        for x, y, x_m, y_m in zip(lng_src, lat_src, lng_src_m, lat_src_m):
            if fsm_state == 0:
                if self.__evalDistanceToStation(x_m, y_m, self.start_station) > threshold:
                    fsm_state = 1
                    first_rec = True
                else:
                    round_point = {'lng': [], 'lat': [], 'lng_m': [], 'lat_m': []}
            if fsm_state == 1:
                if self.__evalDistanceToStation(x_m, y_m, self.start_station) <= threshold:
                    fsm_state = 0
                elif self.__evalDistanceToStation(x_m, y_m, self.end_station) > threshold:
                    round_point['lng'].append(x)
                    round_point['lat'].append(y)
                    round_point['lng_m'].append(x_m)
                    round_point['lat_m'].append(y_m)
                elif self.__evalDistanceToStation(x_m, y_m, self.end_station) <= threshold:
                    fsm_state = 2
                elif self.__evalDistanceToStation(x_m, y_m, self.start_station) <= threshold:
                    fsm_state = 0
            if fsm_state == 2:
                if first_rec == True:
                    round_data.append(round_point)
                    # round_point = {'lng': [], 'lat': [], 'lng_m': [], 'lat_m': []}
                    first_rec = False
                if self.__evalDistanceToStation(x_m, y_m, self.end_station) > threshold:
                    fsm_state = 3
            if fsm_state == 3:
                if self.__evalDistanceToStation(x_m, y_m, self.start_station) <= threshold:
                    fsm_state = 0
        return pd.DataFrame(round_data)

    def trackCorrection(self, lng_src, lat_src, lng_src_m, lat_src_m):
        round_data = self.splitRound(lng_src, lat_src, lng_src_m, lat_src_m)
        round_count = round_data.shape[0]
        corrected_track = []

        for i in range(round_count):
            track = round_data.iloc[i]
            new_track = {'lng_m': [], 'lat_m': []}

            for x, y, x_m, y_m in zip(track['lng'], track['lat'], track['lng_m'], track['lat_m']):
                close_link_1st = {'link': None, 'dist': np.inf}

                p0 = None
                distance = 0


                for link in self.link_list:

                    p1 = np.array([link.upstream_node.lng_m, link.upstream_node.lat_m])
                    p2 = np.array([link.downstream_node.lng_m, link.downstream_node.lat_m])

                    proj = self.__evalProjPoint(x_m, y_m, link)

                    if self.__pointInRange(p1, p2, proj):

                        distance = np.sqrt((x_m - proj[0])**2 + (y_m - proj[1])**2)

                        if distance >= 15:
                            continue
                        elif distance <= close_link_1st['dist']:
                            close_link_1st['dist'] = distance
                            close_link_1st['link'] = link
                            p0 = proj


                if p0 is not None:
                    new_track['lng_m'].append(p0[0])
                    new_track['lat_m'].append(p0[1])

            corrected_track.append(new_track)
        return pd.DataFrame(corrected_track)
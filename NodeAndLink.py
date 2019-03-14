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
        self.direction = self.getDirection()
        self._theta = self.polyfit()

    def __str__(self):
        string = 'Link[(%f, %f), (%f, %f)]' % (self.upstream_node.lng_m, self.upstream_node.lat_m,
                                               self.downstream_node.lng_m, self.downstream_node.lat_m)
        return string




    def getLength(self):
        lng_def = self.upstream_node.lng_m - self.downstream_node.lng_m
        lat_def = self.upstream_node.lat_m - self.downstream_node.lat_m
        length = np.sqrt(lng_def**2 + lat_def**2)
        return length

    def getDirection(self):
        lng_def = self.upstream_node.lng_m - self.downstream_node.lng_m
        lat_def = self.upstream_node.lat_m - self.downstream_node.lat_m
        vector = np.array([lng_def, lat_def]) / self.length
        return vector

    def polyfit(self):
        x = np.array([self.upstream_node.lng_m, self.downstream_node.lng_m])
        y = np.array([self.upstream_node.lat_m, self.downstream_node.lat_m])
        theta = np.polyfit(x, y, deg=1)
        theta = np.array([theta[0], -1, theta[1]])
        return theta

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


    # def __evalProjPoint(self, x_m, y_m, link):
    #     ''' Calculate the projection point on the line'''
    #
    #     # coordinate of p1
    #     x1 = link.upstream_node.lng_m
    #     y1 = link.upstream_node.lat_m
    #
    #     # cooridnate of p2
    #     x2 = link.downstream_node.lng_m
    #     y2 = link.downstream_node.lat_m
    #
    #     t = ((x1 - x_m) * (x1 - x2) - (y1 - y_m) * (y2 - y1)) / ((y2 - y1)**2 + (x1 - x2)**2)
    #
    #     # result coordinate
    #     x = x1 + t * (x2 - x1)
    #     y = y1 + t * (y2 - y1)
    #
    #     plt.plot([x1, x2], [y1, y2], 'bo-')
    #     plt.plot([x_m, x], [y_m, y], 'ro-')
    #     plt.grid()
    #     plt.show()
    #
    #     print(np.array([x2-x1, y2-y1]).dot(np.array([x-x_m, y-y_m])))
    #
    #     return np.array([x, y])

    def __evalProjPoint(self, x_m, y_m, link):
        p1 = np.array([link.upstream_node.lng_m, link.upstream_node.lat_m], dtype=np.float64)
        p2 = np.array([link.downstream_node.lng_m, link.downstream_node.lat_m], dtype=np.float64)
        p3 = np.array([x_m, y_m], dtype=np.float64)
        v1 = p3 - p1
        v2 = p2 - p1

        k = (v1.dot(v2))/(np.linalg.norm(v2)**2)
        # print(k)
        p0 = k * v2+ p1

        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'bo-')
        plt.plot([p3[0], p0[0]], [p3[1], p0[1]], 'ro-')
        plt.grid()
        plt.show()

        return p0



    def __pointInRange(self, point, range_x, range_y):
        if (range_x[0] <= point[0] <= range_x[1]) and (range_y[0] <= point[1] <= range_y[1]):
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

                p1 = None
                distance = 0


                for link in self.link_list:
                    # distance = self.__evalDistanceToLink(x_m, y_m, link)

                    proj = self.__evalProjPoint(x_m, y_m, link)

                    distance = np.sqrt((x_m - proj[0])**2 + (y_m - proj[1])**2)

                    if distance >= 10:
                        continue
                    elif distance <= close_link_1st['dist']:
                        close_link_1st['dist'] = distance
                        close_link_1st['link'] = link
                        p1 = proj


                link_1 = close_link_1st['link']




                # range_1x = (min(link_1.upstream_node.lng_m, link_1.downstream_node.lng_m),
                #             max(link_1.upstream_node.lng_m, link_1.downstream_node.lng_m))
                #
                # range_1y = (min(link_1.upstream_node.lat_m, link_1.downstream_node.lat_m),
                #             max(link_1.upstream_node.lat_m, link_1.downstream_node.lat_m))

                # if p1 is not None and self.__pointInRange(p1, range_1x, range_1y):
                if p1 is not None:
                    new_track['lng_m'].append(p1[0])
                    new_track['lat_m'].append(p1[1])

            corrected_track.append(new_track)
        return pd.DataFrame(corrected_track)
import NodeAndLink
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    p1 = np.array([1.5, 5])
    p2 = np.array([3, 10])
    p = np.array([3, 3.0])

    # coordinate of p1
    x1 = p1[0]
    y1 = p1[1]

    # cooridnate of p2
    x2 = p2[0]
    y2 = p2[1]

    x_m = p[0]
    y_m = p[1]

    t = ((x1 - x_m) * (x1 - x2) - (y1 - y_m) * (y2 - y1)) / ((y2 - y1) ** 2 + (x1 - x2) ** 2)
    print(x1 - x_m, x1 - x2, y1 - y_m, y2 - y1)

    # result coordinate
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    v = np.array([x_m - x, y_m - y])
    print(t, v.dot(p2 - p1))


    plt.plot([x1, x2], [y1, y2], 'bo-')
    plt.scatter(x_m, y_m)
    plt.scatter(x, y)
    plt.grid()
    plt.show()
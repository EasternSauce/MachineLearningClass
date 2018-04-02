import random
import math
import itertools
import numpy
import matplotlib.pyplot as plt


def belongsToSphere(radius, point):
    sum = 0
    for x in range(0, len(point)):
        sum += point[x] ** 2
    return sum <= radius ** 2


def generatePoint(radius, dims):
    point = []
    for x in range(0, dims):
        point.append(random.uniform(-radius, radius))
    return point


def distance(dimensions, point1, point2):
    sum = 0
    for x in range(0, dimensions):
        sum += (point1[x] - point2[x]) ** 2
    return math.sqrt(sum)


def round_up_results(mean_result, std_result):
    if std_result == 0:
        precision = 0
    else:
        precision = -int(math.log(abs(float(std_result)), 10)) + 2
    mean_result = round(float(mean_result), precision)
    std_result = round(float(std_result), precision)
    return mean_result, std_result


# input
x = 1.0
dims = 30

ex1_results = []
ex1_errors = []

for dim in range(1, dims + 1):
    attempts = []
    for attempt in range(0, 10):
        points_list = []
        iterations = 500
        insideCount = 0
        for t in range(0, iterations):
            my_point = generatePoint(x, dim)
            points_list.append(my_point)
            if belongsToSphere(x, my_point):
                insideCount += 1

        percent = 100 * insideCount / iterations
        # print('dimension:', dim)
        # print('attempt:', attempt)
        # print('percentage:', percent, '%')
        attempts.append(percent)

    mean_result = numpy.mean(attempts)
    std_result = numpy.std(attempts)

    (mean_result, std_result) = round_up_results(mean_result, std_result)

    print('\ndimension', dim)

    print('mean result:', mean_result)
    print('result stddev:', std_result)
    ex1_results.append(mean_result)
    ex1_errors.append(std_result)

'''
ex2_results = []


for dim in range(1, dims + 1):
    points_list = []
    distances = []
    iterations = 10
    for t in range(0, iterations):
        my_point = generatePoint(x, dim)
        points_list.append(my_point)
    for t in itertools.combinations(points_list, 2):
        distances.append(distance(dim, t[0], t[1]))
    print('dimensions:', dim)
    stdbymean = numpy.std(distances)/numpy.mean(distances)
    print('st deviation / mean', stdbymean)
    ex2_results.append(stdbymean)

'''

plt.plot(range(1, dims + 1), ex1_results, 'ro', markersize=1)
plt.errorbar(range(1, dims + 1), ex1_results, yerr=ex1_errors, ls='none', markersize=2)
plt.xlabel('dimension')
plt.ylabel('% sphere filled')
plt.show()

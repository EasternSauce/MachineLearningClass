import random
import math
import numpy
import matplotlib.pyplot as plt
import itertools


def belongs_to_sphere(radius, point):
    sum = 0
    for x in range(0, len(point)):
        sum += point[x] ** 2
    return sum <= radius ** 2


def generate_point(radius, dimensions):
    point = []
    for x in range(0, dimensions):
        point.append(random.uniform(-radius, radius))
    return point


def distance(dimensions, point1, point2):
    dist_sum = 0
    for x in range(0, dimensions):
        dist_sum += (point1[x] - point2[x]) ** 2
    return math.sqrt(dist_sum)


def round_up_results(mean, std):
    if std == 0:
        precision = 0
    else:
        precision = -int(math.log(abs(float(std)), 10)) + 2
    mean = round(float(mean), precision)
    std = round(float(std), precision)
    return mean, std


# input
X = 1.0
dims = 15
iterations = 70
repeats = 120

ex2_results = []
ex2_errors = []

for dim in range(1, dims + 1):
    attempts = []
    for attempt in range(0, repeats):
        points_list = []
        distances = []
        for t in range(0, iterations):
            my_point = generate_point(X, dim)
            points_list.append(my_point)
        for t in itertools.combinations(points_list, 2):
            distances.append(distance(dim, t[0], t[1]))
        std_by_mean = numpy.std(distances) / numpy.mean(distances)
        attempts.append(std_by_mean)

    mean_result = numpy.mean(attempts)
    std_result = numpy.std(attempts)

    (mean_result, std_result) = round_up_results(mean_result, std_result)

    print('\ndimension', dim)

    print('mean result:', mean_result)
    print('result stddev:', std_result)

    ex2_results.append(mean_result)
    ex2_errors.append(std_result)

print("dimension stddev/mean error");
for k in range(0, len(ex2_results)):
    print(k+1, ex2_results[k], ex2_errors[k])

plt.plot(range(1, dims + 1), ex2_results, 'ro', markersize=1)
plt.errorbar(range(1, dims + 1), ex2_results, yerr=ex2_errors, ls='none')
plt.xlabel('dimension')
plt.ylabel('standard deviation/mean')
plt.show()

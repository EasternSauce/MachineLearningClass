import random
import math
import numpy
import matplotlib.pyplot as plt


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
repeats = 300
iterations = 500

ex1_results = []
ex1_errors = []

for dim in range(1, dims + 1):
    attempts = []
    for attempt in range(0, repeats):
        points_list = []
        insideCount = 0
        for t in range(0, iterations):
            my_point = generate_point(X, dim)
            points_list.append(my_point)
            if belongs_to_sphere(X, my_point):
                insideCount += 1

        percent = 100 * insideCount / iterations
        attempts.append(percent)

    mean_result = numpy.mean(attempts)
    std_result = numpy.std(attempts)

    (mean_result, std_result) = round_up_results(mean_result, std_result)

    print('\ndimension', dim)

    print('mean result:', mean_result)
    print('result stddev:', std_result)
    ex1_results.append(mean_result)
    ex1_errors.append(std_result)

print("dimension sphere_filled error");
for k in range(0, len(ex1_results)):
    print(k+1, ex1_results[k], ex1_errors[k])

plt.plot(range(1, dims + 1), ex1_results, 'ro', markersize=1)
plt.errorbar(range(1, dims + 1), ex1_results, yerr=ex1_errors, ls='none')
plt.xlabel('dimension')
plt.ylabel('% sphere filled')
plt.show()

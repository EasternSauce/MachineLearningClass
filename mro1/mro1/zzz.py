import math
mean = 1234.56785465
stddev = 0.1
precision = -int(math.log(abs(stddev), 10))+2
print(precision)
print(round(stddev,precision))
print(round(mean,precision))
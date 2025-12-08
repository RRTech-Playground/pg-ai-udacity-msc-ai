import numpy as np
import pandas as pd

groceries = pd.Series(data=[30, 60, 'Yes', 'No'], index=['eggs', 'apples', 'milk', 'bread'])

print(groceries)

print(groceries.shape)
print(groceries.ndim)
print(groceries.size)

print(groceries.values)
print(groceries.index)

x = 'bananas' in groceries
y = 'bread' in groceries

print(x)
print(y)

print(groceries['eggs'])
print(groceries[['eggs', 'milk']])

print(groceries[0])

print(groceries.drop('apples'))
print(groceries)

print('\n')
print(groceries['eggs'])
print(groceries[['milk', 'bread']])
print(groceries.loc[['eggs', 'apples']])
print(groceries[[0, 1]]) # also returns a deprication warning that .iloc should be used
print(groceries[[-1]])  # also returns a deprication warning that .iloc should be used
print(groceries[0])  # also returns a deprication warning that .iloc should be used
print(groceries.iloc[[2, 3]])

print('\n')
fruits= pd.Series(data = [10, 6, 3,], index = ['apples', 'oranges', 'bananas'])
print(fruits)

print(fruits + 2)
print(fruits - 2)
print(fruits * 2)
print(fruits / 2)

print('\n')
print(np.exp(fruits))
print(np.sqrt(fruits))
print(np.power(fruits,2))

print(groceries * 2)
#print(groceries / 2)

print('\n')
planets = ['Earth', 'Saturn', 'Venus', 'Mars', 'Jupiter']
distance_from_sun = [149.6, 1433.5, 108.2, 227.9, 778.6]

dist_planets = pd.Series(data = distance_from_sun, index = planets)

time_of_light = dist_planets / 18

# # TO DO: Use Boolean indexing to select only those planets for which sunlight takes less than 40 minutes to reach them.

close_planets = time_of_light[time_of_light < 40]

print(close_planets)

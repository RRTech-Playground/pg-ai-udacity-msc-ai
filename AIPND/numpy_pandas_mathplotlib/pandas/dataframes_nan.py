import pandas as pd

items2 = [{'bikes': 20, 'pants': 30, 'watches': 35, 'shirts': 15, 'shoes':8,
           'suits':45}, {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5,
                         'shirts': 2, 'shoes':5, 'suits':7}, {'bikes': 20, 'pants': 30,
                                                              'watches': 35, 'glasses': 4, 'shoes':10}]

store_items = pd.DataFrame(items2, index = ['store 1', 'store 2', 'store 3'])
print(store_items)

print('\n')
# Visualize the number of NaN values in store_items
x = store_items.isnull() #.sum().sum()
print(x)

x = store_items.isnull().sum() #.sum()
print(x)

x = store_items.count()
print(x)

x = store_items.isnull().sum().sum()
print(x)

print('\n')
print(store_items.dropna(axis = 0))
print(store_items.dropna(axis = 1))

# print(store_items.dropna(axis = 1, inplace = True))

print('\n')
print(store_items.fillna(0))  # Replace all NaN values with 0
print(store_items.ffill(axis=0))  # Forward Fill - replace NaN values with the previous value in the column
print(store_items.ffill(axis=1))  # Forward Fill - replace NaN values with the previous value in the row
print(store_items.bfill(axis=0))  # Backward Fill - replace NaN values with the next value in the column
print(store_items.bfill(axis=1))  # Backward Fill - replace NaN values with the next value in the row

print('\n')
x = store_items.interpolate(method='linear', axis=0)
print(x)

x = store_items.interpolate(method='linear', axis=1)
print(x)

import pandas as pd

items = {'Alice': pd.Series(data = [40, 110, 500, 45], index = ['book', 'glasses', 'bike', 'pants']),
         'Bob': pd.Series(data = [245, 25, 55], index = ['bike', 'pants', 'watch'])}

print(type(items))  # We print the type of items to see that it is a dictionary

shopping_carts = pd.DataFrame(items)
print(shopping_carts)

data = {'Alice': pd.Series([40, 110, 500, 45]),
        'Bob': pd.Series([245, 25, 55])}

df = pd.DataFrame(data)

print()
print(df)

print()
print(shopping_carts.shape)
print(shopping_carts.ndim)
print(shopping_carts.size)
print(shopping_carts.values)
print(shopping_carts.index)
print(shopping_carts.columns)

print()
bob_shopping_cart = pd.DataFrame(items, columns=['Bob'])
print(bob_shopping_cart)

sel_shopping_cart = pd.DataFrame(items, index = ['pants', 'book'])
print(sel_shopping_cart)

alice_sel_shopping_cart = pd.DataFrame(items, index = ['glasses', 'bike'], columns = ['Alice'])
print(alice_sel_shopping_cart)

print()
data = {'Floats': [4.5, 8.2, 9.6], 'Integers': [1, 2, 3]}
df = pd.DataFrame(data)
print(df)

df = pd.DataFrame(data, index = ['label 1', 'label 2', 'label 3'])
print(df)

print()
items2 = [{'bikes': 20, 'pants': 30, 'watches': 35}, {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5}]

store_items = pd.DataFrame(items2)
print(store_items)

store_items = pd.DataFrame(items2, index = ['store 1', 'store 2'])
print(store_items)

print()
print(store_items[['bikes']])
print(store_items[['bikes', 'pants']])
print(store_items.loc[['store 1']])
print(store_items['bikes']['store 2'])

store_items['shirts'] = [15, 2]
print(store_items)

store_items['suits'] = store_items['pants'] + store_items['shirts']
print(store_items)

new_items = [{'bikes': 20, 'pants': 30, 'watches': 35, 'glasses': 4}]
new_store = pd.DataFrame(new_items, index = ['store 3'])  # new DataFrame
store_items = pd.concat([store_items, new_store])
print(store_items)

print('\n')
store_items['new watches'] = store_items['watches'][1:]
print(store_items)

store_items.insert(4, 'shoes', [8, 5, 0])
print(store_items)

store_items.pop('new watches')
print(store_items)

store_items = store_items.drop(['watches', 'shoes'], axis=1)
print(store_items)

store_items = store_items.drop(['store 2', 'store 1'], axis=0)
print(store_items)

print('\n')
store_items = store_items.rename(columns = {'bikes': 'hats'})
print(store_items)

store_items = store_items.rename(index = {'store 3': 'last store'})
print(store_items)

store_items = store_items.set_index('pants')
print(store_items)

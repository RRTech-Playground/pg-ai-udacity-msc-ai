import numpy as np
import pandas as pd

# Create a random array
array = np.random.rand(25, 5).round(decimals=2)
print(array)

#######################
# Explode a list of values into rows

# create dataframe with single column
df = pd.DataFrame({"list_values": array.tolist()})
print(df.head())

print(df.info())

# each value is a list
print(type(df.iloc[0][0]))

# separate list into row with replicating index values
print(df.explode(column="list_values"))

# separate list into column by column
print(pd.DataFrame(df.list_values.values.tolist()).head())


#######################
# Explode a list of values into columns

# separate list into column by column
pd.DataFrame(df.list_values.values.tolist()).head()


#######################
# Explode a string representation of a list into columns

# now each list is a actually a list of values for each row
df_str = pd.DataFrame({"list_values": [str(a) for a in array.tolist()]})
print(df_str.head())

# each value is a str
print(type(df_str.iloc[0][0]))

# This does not work as expected
print(pd.DataFrame(df_str.list_values.values.tolist()).head())

# One option is to evaluate each string as a list, then proceed as usual
# 'eval' parses and evaluates the string of a list as a python expression
# turning it back into a list
print(pd.DataFrame(df_str.list_values.apply(lambda u: eval(u)).values.tolist()).head())


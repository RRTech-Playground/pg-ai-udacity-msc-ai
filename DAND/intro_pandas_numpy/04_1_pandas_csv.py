import pandas as pd

# load cancer data into data frame
df = pd.read_csv('data/cancer_data.csv')
# display first five rows of data
print(df.head())

# print the column labels in the dataframe
print(df.columns)

for i, v in enumerate(df.columns):
    print(i, v)


df = pd.read_csv('data/student_scores.csv')
print(df.head())

# If the file is separated by different characters, use read_csv() with the sep parameter.
df = pd.read_csv('data/student_scores.csv', sep=':')
print(df.head())

# This obviously didn't work because there our CSV file is separated by commas. Because there are no colons, nothing was separated and everything was read into one column!

## Headers

# Use another line as headers
df = pd.read_csv('data/student_scores.csv', header=2)
print(df.head())

# If no columms, use header=None
df = pd.read_csv('data/student_scores.csv', header=None)
print(df.head())

# Own column labels
labels = ['id', 'name', 'attendance', 'hw', 'test1', 'project1', 'test2', 'project2', 'final']
df = pd.read_csv('data/student_scores.csv', names=labels)
print(df.head())

# Replace a particular row
labels = ['id', 'name', 'attendance', 'hw', 'test1', 'project1', 'test2', 'project2', 'final']
df = pd.read_csv('data/student_scores.csv', header=0, names=labels)
print(df.head())


## Indexing

# Using another column as index
df = pd.read_csv('data/student_scores.csv', index_col='Name')
print(df.head())

df = pd.read_csv('data/student_scores.csv', index_col=['Name', 'ID'])
print(df.head())

# Exercises

df_cancer = pd.read_csv('data/cancer_data.csv', index_col='id')
print(df_cancer.head())

new_column_labels = ['temperature', 'exhaust_vacuum', 'pressure', 'humidity', 'energy_output']
df_powerplant = pd.read_csv('data/powerplant_data.csv', names=new_column_labels, header=0)
print(df_powerplant.head())

# Docs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html


## Writing to CSV

df_powerplant.to_csv('data/powerplant_data_edited.csv')

df = pd.read_csv('data/powerplant_data_edited.csv')
print(df.head())

# remove the index column
df_powerplant.to_csv('data/powerplant_data_edited.csv', index=False)

df = pd.read_csv('data/powerplant_data_edited.csv')
print(df.head())


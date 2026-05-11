import pandas as pd

df = pd.read_csv('data/cancer_data_means.csv')
print(df.head())
print(df.info())

###################
# Missing data

# use means to fill in missing values
# some columns are not numeric, only calculate mean on numeric columns by using .mean(numeric_only=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# confirm your correction with info()
df.info()


###################
# Duplicates

# check for duplicates in the data
print(sum(df.duplicated()))

# drop duplicates
df.drop_duplicates(inplace=True)

# confirm correction by rechecking for duplicates in the data
print(sum(df.duplicated()))


###################
# Rename columns

# remove "_mean" from column names
new_labels = []
for col in df.columns:
    if '_mean' in col:
        new_labels.append(col[:-5])  # exclude last 5 characters
    else:
        new_labels.append(col)

# new labels for our columns
print(new_labels)

# assign new labels to columns in dataframe
df.columns = new_labels

# display first few rows of dataframe to confirm changes
print(df.head())


###################
# Save new data

# save this for later
df.to_csv('data/cancer_data_edited.csv', index=False)


import pandas as pd


#######################
# Drop Columns

# load datasets
df_08 = pd.read_csv('data/all_alpha_08.csv')
df_18 = pd.read_csv('data/all_alpha_18.csv')

# view dataset
print(df_08.head(1))
print(df_18.head(1))

# drop columns from 2008 dataset: 'Stnd', 'Underhood ID', 'FE Calc Appr', 'Unadj Cmb MPG'
df_08.drop(['Stnd', 'Underhood ID', 'FE Calc Appr', 'Unadj Cmb MPG'], axis=1, inplace=True)

# drop columns from 2018 dataset: 'Stnd', 'Stnd Description', 'Underhood ID', 'Comb CO2'
df_18.drop(['Stnd', 'Stnd Description', 'Underhood ID', 'Comb CO2'], axis=1, inplace=True)

# confirm changes
df_08.head(1)
df_18.head(1)

# save data
df_08.to_csv('data/data_08_v1.csv', index=False)
df_18.to_csv('data/data_18_v1.csv', index=False)


#######################
# Rename Columns

# load datasets
df_08 = pd.read_csv('data/data_08_v1.csv')
df_18 = pd.read_csv('data/data_18_v1.csv')

# view dataset
print(df_08.head(1))
print(df_18.head(1))

# rename data
df_08.rename(columns={'Sales Area': 'Cert Region'}, inplace=True)

# confirm changes
print(df_08.head(1))

# with a function

# replace spaces with underscores and lowercase labels for the datasets
df_08.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
df_18.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)

# confirm changes
print(df_08.head(1))
print(df_18.head(1))

# confirm column labels for 2008 and 2018 datasets are identical
print(df_08.columns == df_18.columns)
print((df_08.columns == df_18.columns).all())

# save new datasets for next section
df_08.to_csv('data/data_08_v2.csv', index=False)
df_18.to_csv('data/data_18_v2.csv', index=False)


#######################
# Pandas Query

# Query looks better
#df_2wd = df[df["drive"] == "2WD"]
#df_2wd = df.query('drive == "2WD"')

#df_low_displ = df[df.displ < 3]
#df_low_displ = df.query('displ < 3')


#######################
# Filter, Drop Nulls, Dedupe

df_08 = pd.read_csv('data/data_08_v2.csv')
df_18 = pd.read_csv('data/data_18_v2.csv')

print(df_08.shape)
print(df_18.shape)

# 1. Filter with .query()
df_08 = df_08.query('cert_region == "CA"')
df_18 = df_18.query('cert_region == "CA"')

# confirm only certification region is California
print(df_08['cert_region'].unique())
print(df_18['cert_region'].unique())

# drop certification region columns form both df_08 and df_18 datasets you no longer this column since all data
# should now be filtered by California
df_08.drop('cert_region', axis=1, inplace=True)
df_18.drop('cert_region', axis=1, inplace=True)

print(df_08.shape)
print(df_18.shape)

# 2. Drop Nulls

# view missing value count for each feature
print(df_08.isnull().sum())
print(df_18.isnull().sum())

# drop rows with any null values in both datasets
df_08.dropna(inplace=True)
df_18.dropna(inplace=True)

# checks if any of columns have null values - should print False
print(df_08.isnull().sum().any())
print(df_08.isnull().sum().any())

# 3. Dedupe

# print number of duplicates
print(df_08.duplicated().sum())
print(df_18.duplicated().sum())

# drop duplicates in both datasets
df_08.drop_duplicates(inplace=True)
df_18.drop_duplicates(inplace=True)

# print number of duplicates again to confirm dedupe - should both be 0
print(df_08.duplicated().sum())
print(df_18.duplicated().sum())

# save progress for the next section
df_08.to_csv('data/data_08_v3.csv', index=False)
df_18.to_csv('data/data_18_v3.csv', index=False)

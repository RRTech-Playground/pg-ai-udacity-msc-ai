import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/cancer_data_edited.csv')
print(df.head())

# Example of using a mask to filter our data
mask = df['diagnosis'] == 'M'
print(mask)

# Creating a sub dataset for malignant diagnosis
df_m = df[df['diagnosis'] == 'M']
# Summary statistics, take a look at the mean
print(df_m['area'].describe())

# Creating a sub dataset for benign diagnosis
df_b = df[df['diagnosis'] == 'B']
# Create the same summary stats
print(df_b['area'].describe())

# Create a histogram plot
# .hist() returns a matplotlib subplot
# alpha changes it's transparency
# figsize changes the figure size
ax = df_b['area'].hist(alpha=0.5, figsize=(8, 6), label='benign');
# Layer a new histogram using the same subplot that was returned as 'ax'
df_m['area'].hist(alpha=0.5, figsize=(8, 6), label='malignant', ax=ax);
# Label the subplot with titles and a legend
ax.set_title('Distributions of Benign and Malignant Tumor Areas')
ax.set_xlabel('Area')
ax.set_ylabel('Count')
ax.legend(loc='upper right')

plt.show()



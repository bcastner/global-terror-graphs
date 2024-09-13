import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Load the dataset
data = pd.read_csv('globalterrorismdb.csv', encoding='ISO-8859-1', low_memory=False)

# View the first few rows of the dataset
# print(data.head())

# Check the column names
# print(data.columns)

# The dataset has many columns, so we will focus on the key features:
# iyear, imonth, iday, country_txt, region_txt, attacktype1_txt, targtype1_txt
# weaptype1_txt, nkill, and nwound
relevant_columns = [
    'iyear', 'imonth', 'iday', 'country_txt', 'region_txt', 'attacktype1_txt',
    'targtype1_txt', 'weaptype1_txt', 'nkill', 'nwound'
]
new_data = data[relevant_columns]

# Create dataframe
new_df = pd.DataFrame(new_data)

# We will handle missing values in the nkill and nwound columns by filling them with 0
df = new_df.fillna(0)

# View dataframe
print(df.head())

# Let's explore the dataset by performing analysis and visualizations to understand trends
# Number of attacks per year
attacks_per_year = df['iyear'].value_counts().sort_index()

# Plot the number of attacks per year
plt.figure(figsize=(10, 6))
attacks_per_year.plot(kind='bar', color='darkred')
plt.title('Number of Terrorist Attacks per Year')
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.show()

# Top 10 countries with the most terrorist attacks
top_countries = df['country_txt'].value_counts().head(10)

# Plot the top 10 countries with the most attacks
plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, hue=top_countries.index, legend=False)
plt.title('Top 10 Countries with the Most Terrorist Attacks')
plt.xlabel('Number of Attacks')
plt.ylabel('Country')
plt.show()

# Count of different attack types
attack_types = df['attacktype1_txt'].value_counts()

# Plot the distribution of attack types
plt.figure(figsize=(10, 6))
sns.barplot(x=attack_types.values, y=attack_types.index, hue=attack_types.index, legend=False)
plt.title('Most Common Types of Terrorist Attacks')
plt.xlabel('Number of Attacks')
plt.ylabel('Attack Type')
plt.show()

# Group data by year and sum the number of people killed and wounded
df['total_casualties'] = df.apply(lambda row: row.nkill + row.nwound, axis=1)
total_casualties_df = df[['iyear', 'total_casualties']]
df3 = total_casualties_df.groupby(['iyear']).sum()

# Plot casualties per year (killed and wounded)
plt.figure(figsize=(12, 6))
df3.plot(kind='line', marker='o')
plt.title('Casualties Due to Terrorist Attacks Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Casualties')
plt.show()

# Top 10 target types
target_types = df['targtype1_txt'].value_counts().head(10)

# Plot the top 10 target types
plt.figure(figsize=(10, 6))
sns.barplot(x=target_types.values, y=target_types.index, hue=target_types.index, legend=False)
plt.title('Top 10 Target Types in Terrorist Attacks')
plt.xlabel('Number of Attacks')
plt.ylabel('Target Type')
plt.show()

# Count the number of attacks by weapon type
weapon_types = df['weaptype1_txt'].value_counts()

# Plot the weapon types used in attacks
plt.figure(figsize=(10, 6))
sns.barplot(x=weapon_types.values, y=weapon_types.index, hue=weapon_types.index, legend=False)
plt.title('Types of Weapons Used in Terrorist Attacks')
plt.xlabel('Number of Attacks')
plt.ylabel('Weapon Type')
plt.show()

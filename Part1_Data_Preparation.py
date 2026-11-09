#!/usr/bin/env python
# coding: utf-8

# # COM7022 Machine Learning – Task 1
# ## Part 1: Data Understanding and Preparation
# **Dataset:** Sales-5444.csv (1,000 fragrance product listings)
# 
# **Goal of this notebook:** turn the raw listings into a clean, analysis-ready table that the clustering models in Part 2 can use.

# In[1]:


# Core libraries
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 200)
sns.set_theme(style='whitegrid')


# ### 1.1 Load the data
# The file starts with a hidden byte-order mark (BOM), which would make the first column read as `\ufeffItem_Name`.
# Using `encoding='utf-8-sig'` removes it. Column names are also standardised (fixing the typo `Catatory`).

# In[2]:


import glob
# Find the dataset whatever Colab named it on upload (e.g. 'Sales-[5444].csv' or 'Sales-5444 (1).csv')
files = [f for f in glob.glob('*.csv') if '5444' in f]
print('Loading:', files[0])
df_raw = pd.read_csv(files[0], encoding='utf-8-sig')

df = df_raw.rename(columns={
    'Item_Name': 'brand',
    'Catatory': 'category',
    'priceWithCurrency': 'price_text',
    'text': 'stock_text',
    'lastUpdated': 'last_updated',
    'itemLocation': 'location'
}).copy()

print(df.shape)
df.head()


# ### 1.2 Initial inspection: types, missing values, summary statistics

# In[3]:


df.info()


# In[4]:


missing = pd.DataFrame({'missing': df.isna().sum(),
                        'percent': (df.isna().mean() * 100).round(1)})
missing[missing['missing'] > 0]


# In[5]:


df.describe().T


# In[6]:


print('Duplicate rows:', df.duplicated().sum())
print('Unique categories:', df['category'].nunique())
print('Unique brands:', df['brand'].nunique())


# ### 1.3 Recover missing `price` from `price_text`
# `price_text` (e.g. `US $84.99/ea`) holds the same value as `price`. Where both exist they match exactly,
# so it is a reliable source for filling the gaps instead of dropping rows or using an average.
# The `/ea` suffix is kept as a flag showing a per-unit (multi-quantity) listing.

# In[7]:


price_from_text = (df['price_text']
                   .str.extract(r'([\d,]+\.?\d*)')[0]
                   .str.replace(',', '', regex=False)
                   .astype(float))

# Check consistency where both are present
both = df['price'].notna() & price_from_text.notna()
print('Rows where price and price_text disagree:',
      ((df.loc[both, 'price'] - price_from_text[both]).abs() > 0.01).sum())

print('Missing price before:', df['price'].isna().sum())
df['price'] = df['price'].fillna(price_from_text)
print('Missing price after :', df['price'].isna().sum())

df['per_unit_listing'] = df['price_text'].str.contains('/ea', na=False).astype(int)


# ### 1.4 Parse `stock_text` to recover `available` and validate `sold`
# `stock_text` looks like `"8 available / 48 sold"`, `"More than 10 available / 27 sold"`,
# `"Limited quantity available / 156 sold"` or `"Last One / 6 sold"`. It acts as a second source for both stock and sales.

# In[8]:


stock_part = df['stock_text'].fillna('').str.split('/').str[0].str.strip()
sold_from_text = (df['stock_text'].str.extract(r'/\s*([\d,]+)\s*sold')[0]
                  .str.replace(',', '', regex=False).astype(float))

stock_part.str.replace(r'^[\d,]+ available', '<n> available', regex=True).value_counts()


# In[9]:


# Check: does the sold column agree with the text?
conflict = df['sold'].notna() & sold_from_text.notna() & (df['sold'] != sold_from_text)
df.loc[conflict, ['details', 'stock_text', 'sold']]


# One listing says **"6 sold"** in the text but **7,654** in the `sold` column. This looks like a data-entry error,
# so the text value is taken as correct. Missing `sold` values are also filled from the text where possible.

# In[10]:


df.loc[conflict, 'sold'] = sold_from_text[conflict]
print('Missing sold before:', df['sold'].isna().sum())
df['sold'] = df['sold'].fillna(sold_from_text)
print('Missing sold after :', df['sold'].isna().sum())


# In[11]:


def stock_status(s):
    """Turn the free-text stock description into an ordered category."""
    s = s.lower()
    if 'last one' in s:
        return 'Last one'
    if 'limited' in s:
        return 'Limited'
    if 'more than' in s:
        return 'More than 10'
    m = re.match(r'([\d,]+) available', s)
    if m:
        n = int(m.group(1).replace(',', ''))
        return 'More than 10' if n > 10 else '2-10'
    return np.nan

df['stock_status'] = stock_part.apply(stock_status)

# Fill numeric 'available' from the text where it gives a usable number
avail_from_text = (stock_part.str.extract(r'^([\d,]+) available')[0]
                   .str.replace(',', '', regex=False).astype(float))
avail_from_text[stock_part.str.lower().str.startswith('last one')] = 1
avail_from_text[stock_part.str.lower().str.startswith('more than 10')] = 10

print('Missing available before:', df['available'].isna().sum())
df['available'] = df['available'].fillna(avail_from_text)
print('Missing available after :', df['available'].isna().sum())

pd.crosstab(df['stock_status'], df['available'].isna(), margins=True)


# The remaining missing `available` values are mostly **"Limited quantity"** listings, where the seller hides the exact number.
# These are filled with the median of the low-stock (2–10) group, because "limited" means low stock rather than an average level.
# Note that the platform reports anything above 10 as "More than 10", so `available` is effectively **capped at 10** for most rows.
# This is a limitation to mention in the report.

# In[12]:


low_stock_median = df.loc[df['stock_status'] == '2-10', 'available'].median()
df['available'] = df['available'].fillna(low_stock_median)
df['stock_status'] = df['stock_status'].fillna('Limited')
print('Low-stock median used:', low_stock_median)
print('Missing available now:', df['available'].isna().sum())


# ### 1.5 Standardise `category`
# There are 63 raw category labels, many of them duplicates in different spelling or case
# (e.g. *Eau de Parfum*, *Eau De Parfum*, *Eau de Perfume*). They are grouped into a small set of meaningful product types.

# In[13]:


df['category'].str.strip().str.lower().value_counts()


# In[14]:


def clean_category(c):
    if pd.isna(c):
        return 'Other'
    c = c.lower()
    if 'set' in c or 'gift' in c or 'sampler' in c or 'sample' in c:
        return 'Gift set / Sampler'
    if 'toilette' in c or 'toillette' in c or c.strip() == 'edt':
        return 'Eau de Toilette'
    if 'cologne' in c or c.strip() == 'edc':
        return 'Eau de Cologne'
    if 'parfum' in c or 'perfume' in c or 'edp' in c:
        # 'Parfum' / 'Perfume' alone (extrait) is grouped with EDP as high-concentration
        return 'Eau de Parfum / Parfum'
    if 'fragrance' in c:
        return 'Fragrance (unspecified)'
    # Remaining labels: body sprays, oils, aftershave, deodorant, placeholders like '/' or 'Does not apply'
    return 'Other'

df['category_group'] = df['category'].apply(clean_category)
df['category_group'].value_counts()


# ### 1.6 Simplify `location`
# 91% of listings are in the United States, and there are hundreds of different cities.
# City-level detail would create too many sparse dummy variables, so location is reduced to **country** and a binary **domestic (US) flag**.

# In[15]:


df['country'] = df['location'].str.split(',').str[-1].str.strip()
df['is_us_seller'] = (df['country'] == 'United States').astype(int)
df['country'].value_counts()


# ### 1.7 Brand flag
# There are 246 brand labels, too many to one-hot encode, and some listings are labelled *Unbranded*, *AS SHOW* or *Multiple Brands*.
# A simple flag separates named-brand products from unbranded / generic ones.

# In[16]:


generic = {'unbranded', 'multiple brands', 'as show', 'as shown', 'unknown', 'none', 'generic'}
df['brand'] = df['brand'].fillna('Unbranded').str.strip()
df['is_named_brand'] = (~df['brand'].str.lower().isin(generic)).astype(int)
df['is_named_brand'].value_counts()


# ### 1.8 `last_updated`
# Most timestamps fall in **May 2024** (median: 1 day before the latest update), with a handful of stale listings going back to 2022, plus 53 missing values.
# A `days_since_update` feature is created for inspection. Because it reflects seller activity rather than the product itself, and is missing for 5% of rows, it is a weak candidate for clustering.

# In[17]:


dates = pd.to_datetime(df['last_updated'].str.replace(r'\s+P[DS]T$', '', regex=True),
                       errors='coerce', format='mixed')
df['days_since_update'] = (dates.max() - dates).dt.days
print('Parsed dates:', dates.notna().sum(), 'of', len(df))
print(dates.min(), '->', dates.max())
df['days_since_update'].describe()


# ### 1.9 Distributions and outliers
# `sold` is extremely right-skewed: a small number of bestsellers sell tens of thousands of units.
# Distance-based algorithms like K-Means would be dominated by these few items, so a **log transform** (`log1p`) is applied.

# In[18]:


fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df['price'], bins=40, ax=axes[0, 0]).set_title('Price (USD)')
sns.histplot(df['sold'], bins=40, ax=axes[0, 1]).set_title('Units sold (raw)')
sns.histplot(np.log1p(df['sold']), bins=40, ax=axes[1, 0]).set_title('Units sold (log1p)')
sns.histplot(df['available'], bins=40, ax=axes[1, 1]).set_title('Available stock')
plt.tight_layout()
plt.show()

print('Skewness:')
print(pd.Series({'price': df['price'].skew(),
                 'sold': df['sold'].skew(),
                 'log_sold': np.log1p(df['sold']).skew(),
                 'available': df['available'].skew()}).round(2))


# In[19]:


df['log_sold'] = np.log1p(df['sold'])
df['log_price'] = np.log1p(df['price'])
df['revenue_est'] = df['price'] * df['sold']   # useful for profiling clusters later, not as an input


# In[20]:


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.boxplot(x=df['price'], ax=axes[0]).set_title('Price')
sns.boxplot(x=df['log_sold'], ax=axes[1]).set_title('log(Units sold)')
sns.boxplot(x=df['available'], ax=axes[2]).set_title('Available')
plt.tight_layout()
plt.show()


# ### 1.10 Remaining missing values and final checks
# `sold` still has a few rows where neither column nor text contains a value. With so few cases, dropping them has negligible impact.

# In[21]:


df = df.dropna(subset=['sold', 'price']).reset_index(drop=True)
print('Final shape:', df.shape)
df[['price', 'sold', 'available', 'category_group', 'stock_status',
    'is_us_seller', 'is_named_brand', 'per_unit_listing']].isna().sum()


# In[22]:


fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df[['price', 'log_price', 'log_sold', 'available', 'days_since_update',
                'is_us_seller', 'is_named_brand', 'per_unit_listing']].corr(),
            annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation between candidate numeric features')
plt.tight_layout()
plt.show()


# In[23]:


df.to_csv('sales_clean.csv', index=False)
print('Saved sales_clean.csv with', df.shape[0], 'rows and', df.shape[1], 'columns')


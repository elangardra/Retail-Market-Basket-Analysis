
# Retail Market Basket Analysis using Apriori Algorithm

This project aims to perform market basket analysis on a retail dataset using the Apriori algorithm. Market basket analysis is a technique used to uncover association rules and identify frequent itemsets in customer transactions, providing valuable insights for retailers to understand customer purchasing behavior and develop effective marketing strategies.

The Apriori algorithm is a popular approach for mining association rules and discovering frequent patterns within large datasets. It works by iteratively identifying frequent itemsets and generating association rules that satisfy predefined support and confidence thresholds.
## Dataset

[Online Retail Dataset](https://github.com/elangardra/Retail-Market-Basket-Analysis/blob/master/Online%20Retail%20Data.csv)
![Dataset](https://github.com/elangardra/Retail-Market-Basket-Analysis/blob/master/img/dataset.jpg)
his dataset contains order information from an e-commerce platform. The dataset consists of the following columns:

- `order_id`: A unique identifier assigned to each order placed by a customer.
- `product_code`: A unique code representing the product ordered.
- `product_name`: The name or description of the product ordered.
- `quantity`: The number of units of the product ordered.
- `order_date`: The date and time when the order was placed.
- `price`: The price per unit of the product.
- `customer_id`: A unique identifier for the customer who placed the order.
## Data cleansing
The data cleansing process aims to detect and handle any issues present in the dataset, so that we can have clean and ready-to-use data for analysis or modeling. Some of the steps we will take in this process include:
 #### Creating the 'date' column
```​python
df_clean = df.copy() # membuat kolom date
df_clean['date'] = pd.to_datetime(df_clean['order_date']).dt.date.astype('datetime64')
```
#### Removing rows without 'customer_id'
```​python
df_clean = df_clean[~df_clean['customer_id'].isna()]
```
#### Converting 'customer_id' to string
```​python
df_clean['customer_id'] = df_clean['customer_id'].astype(str)
```
#### Removing rows without 'product_name'
```​python
df_clean = df_clean[~df_clean['product_name'].isna()]
```
#### Converting all 'product_name' to lowercase
```​python
df_clean['product_name'] = df_clean['product_name'].str.lower()
```
#### Removing rows with 'test' in 'product_code' or 'product_name'
```​python
df_clean = df_clean[(~df_clean['product_code'].str.lower().str.contains('test')) | (~df_clean['product_name'].str.contains('test '))]
```
#### Removing rows with canceled status (order_id starting with 'C')
```​python
df_clean = df_clean[df_clean['order_id'].str[:1]!='C']
```
#### Converting negative 'quantity' values to positive
```​python
df_clean['quantity'] = df_clean['quantity'].abs()
```
#### Removing rows with negative 'price' values
```​python
df_clean = df_clean[df_clean['price']>0]
```
#### Creating the 'amount' value by multiplying 'quantity' and 'price'
```​python
df_clean['amount'] = df_clean['quantity'] * df_clean['price']
```
#### Replacing 'product_name' with the most frequent one for each 'product_code'
```​python
most_freq_product_name = df_clean.groupby(['product_code','product_name'], as_index=False).agg(order_cnt=('order_id','nunique')).sort_values(['product_code','order_cnt'], ascending=[True,False])
most_freq_product_name['rank'] = most_freq_product_name.groupby('product_code')['order_cnt'].rank(method='first', ascending=False)
most_freq_product_name = most_freq_product_name[most_freq_product_name['rank']==1].drop(columns=['order_cnt','rank'])
df_clean = df_clean.merge(most_freq_product_name.rename(columns={'product_name':'most_freq_product_name'}), how='left', on='product_code')
df_clean['product_name'] = df_clean['most_freq_product_name']
df_clean = df_clean.drop(columns='most_freq_product_name')
```
#### Removing outliers using z-score
```​python
from scipy import stats
df_clean = df_clean[(np.abs(stats.zscore(df_clean[['quantity','amount']]))<3).all(axis=1)]
df_clean = df_clean.reset_index(drop=True)
df_clean

```
#### Data Cleansing Result
![Dataset](https://github.com/elangardra/Retail-Market-Basket-Analysis/blob/master/img/data%20cleaning.jpg)
 Column | Non-Null Count | Dtype |
| --- | --- | --- |
| order_id | 350092 non-null | object |
| product_code | 350092 non-null | object |
| product_name | 350092 non-null | object |
| quantity | 350092 non-null | int64 |
| order_date | 350092 non-null | object |
| price | 350092 non-null | float64 |
| customer_id | 350092 non-null | object |
| date | 350092 non-null | datetime64[ns] |
| amount | 350092 non-null | float64 |

## Creating the Basket DataFrame

After performing the data cleaning process, the next step is to create a basket DataFrame that will be used for market basket analysis. This basket DataFrame is created using the `pivot_table` function from the Pandas library.

```python
import pandas as pd

basket = pd.pivot_table(df_clean, index='order_id', columns='product_name', values='product_code', aggfunc='nunique', fill_value=0)
```
This basket DataFrame enables analyses such as identifying frequently purchased product combinations or performing market basket analysis.
- `pd.pivot_table(df_clean, ...)` creates a pivot table from the `df_clean` DataFrame, which has been cleaned in the previous step.
 - `df_clean` is the DataFrame resulting from the data cleaning process.

- `index='order_id'` sets the 'order_id' column as the row index in the pivot table.
 - Each row in the pivot table will represent a single order_id.

- `columns='product_name'` sets the 'product_name' column as the column names in the pivot table.
 - Each column in the pivot table will represent a unique product.

- `values='product_code'` uses the values from the 'product_code' column to fill the cells in the pivot table.
 - The value in each cell will indicate the number of unique product codes for that combination of order_id and product_name.

- `aggfunc='nunique'` uses the `nunique` function to count the number of unique values in each cell of the pivot table.
 - This means each cell will contain the count of unique product codes for that combination of order_id and product_name.

- `fill_value=0` fills empty cells with the value 0.
 - If there is a combination of order_id and product_name with no data, that cell will be filled with 0.
## Encoding the Basket DataFrame

After creating the `basket` DataFrame, the next step is to encode its values with `True` for all values above 0 and `False` for all values equal to 0. This is done to simplify the representation of transaction data, where `True` indicates that a particular product is present in the shopping basket, and `False` indicates that the product is not present.

```python
def encode(x):
   if x == 0:
       return False
   if x > 0:
       return True

basket_encode = basket.applymap(encode)
```
- `def encode(x):` defines the `encode` function that will encode each value in the DataFrame.
 - This function takes a value `x` and returns `True` if `x` is greater than 0, and `False` if `x` is equal to 0.

- `basket_encode = basket.applymap(encode)` applies the `encode` function to each value in the `basket` DataFrame using the `applymap` method.
 - The result is a new DataFrame `basket_encode` containing `True` for all values above 0 and `False` for all values equal to 0.
![Dataset](https://github.com/elangardra/Retail-Market-Basket-Analysis/blob/master/img/pivot%20market%20basket.jpg)
## Retrieving Transactions with More Than One Unique Product

After encoding the `basket` DataFrame, the next step is to retrieve only transactions that have more than one unique product. This is done to focus the analysis on shopping baskets containing more than one product, allowing us to identify patterns of frequently purchased product combinations.

In market basket analysis, we are more interested in transactions consisting of multiple different products. Transactions with only one product do not provide useful information about frequently purchased product combinations. Therefore, by retrieving transactions with more than one unique product, we can focus the analysis on more relevant and interesting patterns of product combinations.

```python
basket_filter = basket_encode[(basket_encode>0).sum(axis=1)>1]
basket_filter
```
The code creates a new DataFrame `basket_filter` that contains only rows from `basket_encode` where the sum of `True` values (products present) across each row is greater than 1. This filters out transactions with only one unique product, as they do not provide useful information for identifying product combinations.
![Dataset](https://github.com/elangardra/Retail-Market-Basket-Analysis/blob/master/img/encoding.jpg)

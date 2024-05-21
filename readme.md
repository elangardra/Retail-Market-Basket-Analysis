
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
```
df_clean = df_clean[~df_clean['customer_id'].isna()]
```
#### Converting 'customer_id' to string
```
df_clean['customer_id'] = df_clean['customer_id'].astype(str)
```
#### Removing rows without 'product_name'
```
df_clean = df_clean[~df_clean['product_name'].isna()]
```
#### Converting all 'product_name' to lowercase
```
df_clean['product_name'] = df_clean['product_name'].str.lower()
```
#### Removing rows with 'test' in 'product_code' or 'product_name'
```
df_clean = df_clean[(~df_clean['product_code'].str.lower().str.contains('test')) | (~df_clean['product_name'].str.contains('test '))]
```
#### Removing rows with canceled status (order_id starting with 'C')
```
df_clean = df_clean[df_clean['order_id'].str[:1]!='C']
```
#### Converting negative 'quantity' values to positive
```
df_clean['quantity'] = df_clean['quantity'].abs()
```
#### Removing rows with negative 'price' values
```
df_clean = df_clean[df_clean['price']>0]
```
#### Creating the 'amount' value by multiplying 'quantity' and 'price'
```
df_clean['amount'] = df_clean['quantity'] * df_clean['price']
```
#### Replacing 'product_name' with the most frequent one for each 'product_code'
```
most_freq_product_name = df_clean.groupby(['product_code','product_name'], as_index=False).agg(order_cnt=('order_id','nunique')).sort_values(['product_code','order_cnt'], ascending=[True,False])
most_freq_product_name['rank'] = most_freq_product_name.groupby('product_code')['order_cnt'].rank(method='first', ascending=False)
most_freq_product_name = most_freq_product_name[most_freq_product_name['rank']==1].drop(columns=['order_cnt','rank'])
df_clean = df_clean.merge(most_freq_product_name.rename(columns={'product_name':'most_freq_product_name'}), how='left', on='product_code')
df_clean['product_name'] = df_clean['most_freq_product_name']
df_clean = df_clean.drop(columns='most_freq_product_name')
```
#### Removing outliers using z-score
```
from scipy import stats
df_clean = df_clean[(np.abs(stats.zscore(df_clean[['quantity','amount']]))<3).all(axis=1)]
df_clean = df_clean.reset_index(drop=True)
df_clean

```
#### Data Cleansing Result
![Dataset](https://github.com/elangardra/Retail-Market-Basket-Analysis/blob/master/img/data%20cleaning.jpg)

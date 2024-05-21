
# Retail Market Basket Analysis using Apriori Algorithm

This project aims to perform market basket analysis on a retail dataset using the Apriori algorithm. Market basket analysis is a technique used to uncover association rules and identify frequent itemsets in customer transactions, providing valuable insights for retailers to understand customer purchasing behavior and develop effective marketing strategies.

The Apriori algorithm is a popular approach for mining association rules and discovering frequent patterns within large datasets. It works by iteratively identifying frequent itemsets and generating association rules that satisfy predefined support and confidence thresholds.
## Dataset

[Online Retail Dataset](https://github.com/elangardra/Retail-Market-Basket-Analysis/blob/master/Online%20Retail%20Data.csv)

![Dataset](URL/lokasi/gambar.jpg)
## Data cleansing
 #### membuat kolom date
```​python
df_clean = df.copy() # membuat kolom date
df_clean['date'] = pd.to_datetime(df_clean['order_date']).dt.date.astype('datetime64')
```
#### menghapus semua baris tanpa customer_id
```
df_clean = df_clean[~df_clean['customer_id'].isna()]
```
#### mengkonversi customer_id menjadi string
```
df_clean['customer_id'] = df_clean['customer_id'].astype(str)
```
#### menghapus semua baris tanpa product_name
```
df_clean = df_clean[~df_clean['product_name'].isna()]
```
#### membuat semua product_name berhuruf kecil
```
df_clean['product_name'] = df_clean['product_name'].str.lower()
```
#### menghapus semua baris dengan product_code atau product_name test
```
df_clean = df_clean[(~df_clean['product_code'].str.lower().str.contains('test')) | (~df_clean['product_name'].str.contains('test '))]
```
#### menghapus baris dengan status cancelled, yaitu yang order_id-nya diawali 'C'
```
df_clean = df_clean[df_clean['order_id'].str[:1]!='C']
```
#### mengubah nilai quantity yang negatif menjadi positif karena nilai negatif tersebut hanya menandakan order tersebut cancelled
```
df_clean['quantity'] = df_clean['quantity'].abs()
```
#### menghapus baris dengan price bernilai negatif
```
df_clean = df_clean[df_clean['price']>0]
```
#### membuat nilai amount, yaitu perkalian antara quantity dan price
```
df_clean['amount'] = df_clean['quantity'] * df_clean['price']
```
#### mengganti product_name dari product_code yang memiliki beberapa product_name dengan salah satu product_name-nya yang paling sering muncul
```
most_freq_product_name = df_clean.groupby(['product_code','product_name'], as_index=False).agg(order_cnt=('order_id','nunique')).sort_values(['product_code','order_cnt'], ascending=[True,False])
most_freq_product_name['rank'] = most_freq_product_name.groupby('product_code')['order_cnt'].rank(method='first', ascending=False)
most_freq_product_name = most_freq_product_name[most_freq_product_name['rank']==1].drop(columns=['order_cnt','rank'])
df_clean = df_clean.merge(most_freq_product_name.rename(columns={'product_name':'most_freq_product_name'}), how='left', on='product_code')
df_clean['product_name'] = df_clean['most_freq_product_name']
df_clean = df_clean.drop(columns='most_freq_product_name')
```
### menghapus outlier
```
from scipy import stats
df_clean = df_clean[(np.abs(stats.zscore(df_clean[['quantity','amount']]))<3).all(axis=1)]
df_clean = df_clean.reset_index(drop=True)
df_clean

```
![Dataset](https://github.com/elangardra/Retail-Market-Basket-Analysis/blob/master/img/data%20cleaning.jpeg)

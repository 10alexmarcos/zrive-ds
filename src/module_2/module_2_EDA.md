```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

```


```python
pd.set_option('display.width', 220)
pd.set_option('display.max_colwidth', 30)
```

# ***Homework part 1***

# ***Orders dataset***

In this section I will check:

* Distribution of hours at which clients create orders

* The evolution of orders in the date available

* The distribution of the number of products per order

* Unique clients vs recurrent clients

* Find the most bought items (and relate the variant_id with the product_name, price, etc( inventory data))

* Make a market basket analysis to discover products bought together


```python
orders_path = '/home/alex/zriveAM/zrive-ds/aws/data/orders.parquet'
orders_df = pd.read_parquet(orders_path)
orders_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd739411...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618849693828, 336188601...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a8...</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618835243140, 336188359...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2204838822020</td>
      <td>329f08c66abb51f8c0b8a95266...</td>
      <td>2020-04-30 18:12:30</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618891145348, 336188935...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2208967852164</td>
      <td>f6451fce7b1c58d0effbe37fcb...</td>
      <td>2020-05-01 19:44:11</td>
      <td>2020-05-01</td>
      <td>1</td>
      <td>[33618830196868, 336188465...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>[33667166699652, 336671666...</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Orders info:")
print(orders_df.info())
```

    Orders info:
    <class 'pandas.core.frame.DataFrame'>
    Index: 8773 entries, 10 to 64538
    Data columns (total 6 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   id              8773 non-null   int64         
     1   user_id         8773 non-null   object        
     2   created_at      8773 non-null   datetime64[us]
     3   order_date      8773 non-null   datetime64[us]
     4   user_order_seq  8773 non-null   int64         
     5   ordered_items   8773 non-null   object        
    dtypes: datetime64[us](2), int64(2), object(2)
    memory usage: 479.8+ KB
    None



```python
#column with the hour of creation of the order
orders_df['created_at_hour']=orders_df['created_at'].dt.hour
print(orders_df.head())

orders_per_hour=orders_df['created_at_hour'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(12,6))
orders_per_hour.plot(kind='bar')
plt.title('Creation of orders per hour distribution')
plt.xlabel('Time of the day')
plt.ylabel('Number of orders')
plt.show()


```

                   id                        user_id          created_at order_date  user_order_seq                  ordered_items  created_at_hour
    10  2204073066628  62e271062eb827e411bd739411... 2020-04-30 14:32:19 2020-04-30               1  [33618849693828, 336188601...               14
    20  2204707520644  bf591c887c46d5d3513142b6a8... 2020-04-30 17:39:00 2020-04-30               1  [33618835243140, 336188359...               17
    21  2204838822020  329f08c66abb51f8c0b8a95266... 2020-04-30 18:12:30 2020-04-30               1  [33618891145348, 336188935...               18
    34  2208967852164  f6451fce7b1c58d0effbe37fcb... 2020-05-01 19:44:11 2020-05-01               1  [33618830196868, 336188465...               19
    49  2215889436804  68e872ff888303bff58ec56a3a... 2020-05-03 21:56:14 2020-05-03               1  [33667166699652, 336671666...               21



    
![png](module_2_EDA_files/module_2_EDA_6_1.png)
    


We can push notifications at these hours to maximize creating orders


```python
#evolution of orders in the last years

print("Orders evolution")

orders_by_month = orders_df.groupby(orders_df['order_date'].dt.to_period('M')).size()
orders_by_month.index = orders_by_month.index.to_timestamp()

plt.figure(figsize=(12,6))
plt.plot(orders_by_month.index, orders_by_month.values, marker='o', linestyle='-', color='b')
plt.title('Orders evolution over time')
plt.xlabel('Date')
plt.ylabel('Number of orders')
plt.grid(True, alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

    Orders evolution



    
![png](module_2_EDA_files/module_2_EDA_8_1.png)
    



```python
#Let's create a new column with the nbumber of items per order
orders_df['items_per_order'] = orders_df['ordered_items'].apply(len)
orders_df=orders_df[['id','user_id','created_at', 'created_at_hour','order_date','user_order_seq','ordered_items','items_per_order']]
orders_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>created_at_hour</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
      <th>items_per_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd739411...</td>
      <td>2020-04-30 14:32:19</td>
      <td>14</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618849693828, 336188601...</td>
      <td>14</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a8...</td>
      <td>2020-04-30 17:39:00</td>
      <td>17</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618835243140, 336188359...</td>
      <td>25</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2204838822020</td>
      <td>329f08c66abb51f8c0b8a95266...</td>
      <td>2020-04-30 18:12:30</td>
      <td>18</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618891145348, 336188935...</td>
      <td>15</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2208967852164</td>
      <td>f6451fce7b1c58d0effbe37fcb...</td>
      <td>2020-05-01 19:44:11</td>
      <td>19</td>
      <td>2020-05-01</td>
      <td>1</td>
      <td>[33618830196868, 336188465...</td>
      <td>10</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a...</td>
      <td>2020-05-03 21:56:14</td>
      <td>21</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>[33667166699652, 336671666...</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
#items_per_order distribution
print(orders_df['items_per_order'].describe())

item_counts = orders_df['items_per_order'].value_counts().sort_index()

item_counts_normalized = item_counts / item_counts.sum()

plt.figure(figsize=(14, 7))
sns.barplot(x= item_counts_normalized.index, y= item_counts_normalized.values, color='skyblue')
#item_counts_normalized.plot(kind='bar', color='skyblue')
plt.title('Items per order distribution')
plt.xlabel('Items per order')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(False)
plt.tight_layout()
plt.show()
```

    count    8773.000000
    mean       12.305711
    std         6.839507
    min         1.000000
    25%         8.000000
    50%        11.000000
    75%        15.000000
    max       114.000000
    Name: items_per_order, dtype: float64



    
![png](module_2_EDA_files/module_2_EDA_10_1.png)
    



```python
#Unique clients
unique_clients=orders_df['user_id'].value_counts()
unique_clients
```




    user_id
    ba7176c5f870cd86e51ecc4375e0becc8cc305845e70b9384ba0d4d156c6099ec96602b15420cb1ff69b6b9adcf9249d09489d511565531c4e928a92157b16d6    25
    114e78a8909ad3f9d481e66563998301eff9e7cd1b2d002b77ffc26619f0ef7a6e877d5b3460e0e0bde2d7c67787c66d7384ccf34b4aa4fa1409e978cf47e670    22
    04e9d7967f4dfd7d40175f130f1c80f62204ff697df92dfd83407ace7997b6744b6a7cab0382e60f7264b13ba3f03c64f016ae9dca664885ace6020aba3b5131    22
    1296e1e72f7f43ff28d7d285f880ad4d213fa8139233c78ad2673af4cc0c1c297180254da494cc3b934858347822be437eeb072acb1403e9476dc0b2f6a83cbe    21
    a655b992a3ee5f6aa9f5ab5662c4befac1fdc45c99885b173618b661ca7381c383b1364e977c9f51366cdc62e4d92ff383e8e5aac7ea9a552059be2671de24e2    21
                                                                                                                                        ..
    720193c88605b61f880b340ca9a02c673241a37ff74f7d20639fc933ed5438a3b96076b85f168231d34779994c26810adbb24e0d842639837780649a4baf16a8     1
    76f2a192182b2d88816355ac40307d4323c6d62da2a27d9730ada0ede741b392ff4eba195d0870fc0eee8e3cb05c9eb150eee8a34bd9fb6e95bbb6ed7ee4809b     1
    684faa133509c3941767934993b87c95993e24380e2fe140b774425938b1935d27b01b3b8c099fa3f471b4bba70b20318c2e90c0fb2bf4ac8af5fea3feb1cc00     1
    da143411647c56faccca7f404462dcd9212a59d75e8cc124b607d36e3a410201f78f520fd534858b3cbc60e2a3572494e3dfaf590769fc6ac108af5c402be473     1
    bd5dd90b15fc4c382330a17915927322d928a091b13f91aabe1b58afb721b3d4f47c56ff306eb34e6d94cabeed4dcea7d6a15ed81d605a0e9f4fd05a4b787c55     1
    Name: count, Length: 4983, dtype: int64




```python
plt.figure(figsize=(10, 6))
value_counts_unique_clients=unique_clients.value_counts().sort_index()

value_counts_unique_clients_normalized = value_counts_unique_clients / value_counts_unique_clients.sum()
sns.barplot(x=value_counts_unique_clients_normalized.index, y=value_counts_unique_clients_normalized.values, color='skyblue')

plt.title('Orders per user distribution')
plt.xlabel('Number of orders per user')
plt.ylabel('Users frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_12_0.png)
    



```python
num_users_with_1_order = (unique_clients == 1).sum()
print(f'Users with 1 order: {num_users_with_1_order}')
num_users_with_more_than_1_order = (unique_clients > 1).sum()
print(f"Users with more than 1 order: {num_users_with_more_than_1_order}")
categories= ['1 order', 'More than 1 order']
counts=[num_users_with_1_order, num_users_with_more_than_1_order]
total=sum(counts)
normalized= [count / total for count in counts]

plt.figure(figsize=(8,6))
plt.bar(categories, normalized, color=['blue', 'green'])
plt.tight_layout
plt.show()

percentage_clients_repeat= num_users_with_more_than_1_order / (num_users_with_more_than_1_order + num_users_with_1_order)
print(f"Clients that repeated buying in our platform: {percentage_clients_repeat*100} %")
```

    Users with 1 order: 3572
    Users with more than 1 order: 1411



    
![png](module_2_EDA_files/module_2_EDA_13_1.png)
    


    Clients that repeated buying in our platform: 28.316275336142887 %



```python
#Lets' examine the items in the orders
all_products = [product for sublist in orders_df['ordered_items'] for product in sublist]
print(type(all_products))

product_series = pd.Series(all_products)
print(product_series.dtype)
product_counts = product_series.value_counts().reset_index()
product_counts.columns = ['variant_id', 'Frequency']
product_counts = product_counts.sort_values(by='Frequency', ascending=False)
print(product_counts.head())
```

    <class 'list'>
    int64
           variant_id  Frequency
    0  34081589887108       4487
    1  39284117930116       2658
    2  34137590366340       1459
    3  34081331970180       1170
    4  34284951863428       1133



```python
top_products = product_counts.head(150)

plt.figure(figsize=(12, 8))
sns.barplot(x='variant_id', y='Frequency',data=top_products,legend=False,color='blue',order=top_products['variant_id'])


plt.title('Top sold products')
plt.xlabel('Product (variant_id)')
plt.ylabel('Times sold')
plt.xticks([])
plt.tight_layout()
plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_15_0.png)
    



```python
#join products counts with inventory data
inventory_path = '/home/alex/zriveAM/zrive-ds/aws/data/inventory.parquet'
inventory_df = pd.read_parquet(inventory_path)
print(inventory_df.shape)
inventory_df.head()
```

    (1733, 6)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34370361229444</td>
      <td>4.99</td>
      <td>5.50</td>
      <td>whogivesacrap</td>
      <td>toilet-roll-kitchen-roll-t...</td>
      <td>[b-corp, eco, toilet-rolls]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34284951863428</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-t...</td>
      <td>[kitchen-roll]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33667283583108</td>
      <td>1.79</td>
      <td>1.99</td>
      <td>thecheekypanda</td>
      <td>toilet-roll-kitchen-roll-t...</td>
      <td>[b-corp, cruelty-free, eco...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33803537973380</td>
      <td>1.99</td>
      <td>2.09</td>
      <td>colgate</td>
      <td>dental</td>
      <td>[dental-accessories]</td>
    </tr>
  </tbody>
</table>
</div>



In the inventory df we have 1733 variant_ids (products), while in product counts we have 2117 variant_ids. It suggests that some products have been removed from stock and are no longer available


```python
#Complete product_counts dataframe with information from inventory
combined_df = pd.merge(product_counts,inventory_df,on='variant_id',how='left')

print(combined_df.head())
print(combined_df.shape)
print('------------------------------------')
print('Delete rows with null values')
print('------------------------------------')
combined_df_cleaned=combined_df.dropna()
print(combined_df_cleaned.head())
print(combined_df_cleaned.shape)


```

           variant_id  Frequency  price  compare_at_price  vendor                   product_type               tags
    0  34081589887108       4487  10.79             11.94   oatly     long-life-milk-substitutes  [oat-milk, vegan]
    1  39284117930116       2658    NaN               NaN     NaN                            NaN                NaN
    2  34137590366340       1459    NaN               NaN     NaN                            NaN                NaN
    3  34081331970180       1170    NaN               NaN     NaN                            NaN                NaN
    4  34284951863428       1133   3.69              3.99  plenty  toilet-roll-kitchen-roll-t...     [kitchen-roll]
    (2117, 7)
    ------------------------------------
    Delete rows with null values
    ------------------------------------
           variant_id  Frequency  price  compare_at_price         vendor                   product_type                           tags
    0  34081589887108       4487  10.79             11.94          oatly     long-life-milk-substitutes              [oat-milk, vegan]
    4  34284951863428       1133   3.69              3.99         plenty  toilet-roll-kitchen-roll-t...                 [kitchen-roll]
    5  34284950356100        954   1.99              3.00          fairy                    dishwashing  [discontinue, swapped, was...
    6  34370361229444        939   4.99              5.50  whogivesacrap  toilet-roll-kitchen-roll-t...    [b-corp, eco, toilet-rolls]
    7  33826465153156        884   1.89              1.99    clearspring            tins-packaged-foods  [gluten-free, meat-alterna...
    (1477, 7)


Althought we have 1733 variant_ids in the inventory, some have null values, that's the reason of having 1477 rows after the cleaning instead of 1733 (supposedly we would go from 2117 --> 1733, but we went from 2117 --> 1477).

At this point we should guess why there are missing variant_ids, it's probably that it's because this items are no longer available, but we should confirm this hypothesis


```python
null_rows_df = combined_df[combined_df.isnull().any(axis=1)]
null_rows_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>Frequency</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>39284117930116</td>
      <td>2658</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34137590366340</td>
      <td>1459</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34081331970180</td>
      <td>1170</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>42</th>
      <td>39711187894404</td>
      <td>315</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>48</th>
      <td>39478260662404</td>
      <td>293</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#use orders_df and null_rows_df to discover the date of the null items
null_variants_id=null_rows_df['variant_id'].tolist()
print(null_variants_id)
print('-----------------------')


def count_null_items(ordered_items):
    return sum(item in null_variants_id for item in ordered_items)


orders_df['null_items_count'] = orders_df['ordered_items'].apply(count_null_items)
orders_df['ratio_null_items'] = orders_df['null_items_count'] / orders_df['items_per_order']


print(orders_df[['id', 'order_date', 'ordered_items', 'items_per_order', 'null_items_count', 'ratio_null_items']].head())
```

    [39284117930116, 34137590366340, 34081331970180, 39711187894404, 39478260662404, 33977921208452, 39459281404036, 39462593233028, 39511041605764, 34370917204100, 33667283648644, 34436055269508, 39459282059396, 33977922650244, 33826477834372, 34519123820676, 33667263627396, 34081589624964, 34502822396036, 39459279437956, 34436055171204, 33667216441476, 34304124420228, 34457368199300, 39459280912516, 34276571316356, 33667174695044, 39368665825412, 34284953763972, 39459280224388, 33667232891012, 34304124387460, 39459280027780, 33667214180484, 33826457059460, 39711187861636, 34276569219204, 33667305373828, 39459280126084, 33826427240580, 34276570562692, 33826423406724, 33667232661636, 33826478555268, 33826467446916, 39711187927172, 34317850673284, 33826459844740, 34317851984004, 33667226173572, 34535159398532, 34221708312708, 34370914320516, 33973242429572, 33667241083012, 34081589690500, 34368925237380, 39459279732868, 33826454667396, 34535159595140, 33826414329988, 34317851525252, 33826410791044, 33826465480836, 34137389400196, 33826431008900, 34529808777348, 33667232694404, 39345365680260, 39459277275268, 39403032543364, 34081590313092, 33951140053124, 39459277308036, 33719430185092, 34276570300548, 34370914222212, 33977921405060, 34368927957124, 34276569317508, 33667293249668, 33863279214724, 39563904024708, 33826424750212, 34284955598980, 39459281797252, 33826457157764, 34221708247172, 33973247017092, 34276569186436, 33973247082628, 39573620555908, 33977922715780, 33826462761092, 39459280060548, 34037940256900, 39459279503492, 33973242331268, 34488548098180, 39478260859012, 34246817087620, 33667172466820, 39459278684292, 34488547967108, 34137389301892, 33826455158916, 34488548130948, 33973242658948, 33803538006148, 34304124354692, 33973242626180, 33667235971204, 34460976251012, 33973242691716, 34490663338116, 39463987970180, 33826459779204, 34284954779780, 33667249930372, 34465294221444, 34173018407044, 34081590149252, 33826448244868, 39272601813124, 33826412920964, 33826459385988, 34246817644676, 39650084356228, 33826409906308, 33803539775620, 33981947904132, 34037939798148, 39587297853572, 34198503096452, 33826466857092, 33973247049860, 33719430545540, 33973243347076, 39368665792644, 34137389465732, 33826456436868, 34246817316996, 34276569448580, 34173018275972, 34276569284740, 34284955631748, 33719437983876, 33826420621444, 34436055138436, 33667307798660, 33803541020804, 34284953960580, 33667184492676, 34370916614276, 33667192193156, 34415986671748, 34246817480836, 33826414461060, 33719434641540, 34086451609732, 33981947314308, 34173021126788, 33667173449860, 34537169092740, 33826460008580, 33814259761284, 33667262972036, 34284954026116, 33826459287684, 34284952420484, 39483175698564, 34086451576964, 33719431266436, 34081590083716, 33826447229060, 34276571152516, 33667184787588, 34246817185924, 33977921437828, 34284952780932, 34284954386564, 34086451019908, 34198503030916, 33667226140804, 33667290267780, 33826453094532, 34460976283780, 34460976382084, 33826433106052, 33973248065668, 33667214246020, 39542986965124, 34246817284228, 33719431200900, 34368927760516, 33719437918340, 33719435427972, 33973243445380, 33826414264452, 34086451773572, 33667196321924, 34079207653508, 34370917007492, 33826424193156, 34086451413124, 33826431729796, 33826427666564, 39589498978436, 34198503194756, 34173018243204, 33826464956548, 33667241410692, 33826471739524, 34284952682628, 33667215032452, 39482337525892, 33826415411332, 39459278225540, 34457368100996, 33826412593284, 34276570988676, 39459281830020, 33667166699652, 33719430447236, 33667308716164, 34081590378628, 33667254026372, 34284954681476, 39857724850308, 34488548688004, 39587298574468, 34535159562372, 39459281764484, 33977921470596, 33826427535492, 34284954615940, 34519123853444, 33719436312708, 34368925270148, 33981947379844, 33667235938436, 33667236364420, 39690868260996, 33826430976132, 34284953993348, 33826457616516, 34037939568772, 33977921011844, 34137389105284, 34284950552708, 39459279569028, 33667182657668, 33826454503556, 34173018439812, 33719429562500, 33667312025732, 34037939536004, 33719430250628, 33667216277636, 34368926843012, 33667213754500, 33826432123012, 33667312124036, 33719437885572, 33826422489220, 39830149988484, 34415989096580, 34460976119940, 39478260924548, 33667236331652, 34086451150980, 33667216343172, 34368924483716, 33977921765508, 34113606320260, 39540483817604, 33973246820484, 34284952387716, 33826446508164, 34488547639428, 34284952518788, 34246817251460, 34370914943108, 39272600928388, 39830154805380, 34368927727748, 39478260891780, 34086451183748, 33973242364036, 33719433363588, 33667239772292, 33973242298500, 33667278405764, 33667247014020, 33719433494660, 33719433953412, 33826424127620, 33667253960836, 34081589067908, 33826431139972, 34284950126724, 34368924254340, 34276571545732, 33667203498116, 34173018112132, 34441183068292, 33826428485764, 33826415542404, 39614170857604, 39482337198212, 39309575389316, 34415985262724, 34370916745348, 39367712800900, 34086451118212, 33826445918340, 34284954517636, 39349425471620, 34415989162116, 39367712637060, 34368924385412, 33814259728516, 33719433527428, 34460976644228, 34534187663492, 34276570890372, 34086450921604, 34221746028676, 34537143763076, 33667302424708, 33667241312388, 39250947506308, 39326057857156, 39318967976068, 33667185016964, 33981948067972, 33826423865476, 34221711491204, 34415989227652, 34370914746500, 34368926449796, 39830150414468, 33803539415172, 33667171221636, 34221708083332, 33667293839492, 33719431233668, 33667260645508, 33826471936132, 34460975923332, 34368926777476, 34284954124420, 33719434117252, 33667265790084, 33667312255108, 33826446082180, 33667222896772, 33667235250308, 33667184951428, 33667239739524, 39349425275012, 34221709590660, 33667307896964, 34037939601540, 33719430578308, 34368926941316, 33826413707396, 39729614848132, 33719429464196, 33803540562052, 33803538956420, 39729614782596, 34086451806340, 34276569481348, 33803540463748, 33826427142276, 33949879402628, 39367712768132, 33667184722052, 33977922158724, 39250947965060, 33667279192196, 33951137923204, 39336624062596, 33667185639556, 33667260514436, 34198503227524, 33826422423684, 33719437492356, 34284954583172, 33826416033924, 33826446475396, 34368927694980, 33667281256580, 33667275227268, 33667171287172, 33992554610820, 34081590247556, 33977922257028, 34537168928900, 33667178856580, 33719434150020, 39376326099076, 39474126979204, 34137389006980, 33667299803268, 34537144320132, 33814259794052, 33826412822660, 33667214966916, 34488548589700, 33667192782980, 33803539316868, 33667241476228, 39614170923140, 39729614585988, 33667258450052, 33977921798276, 33667296100484, 34103693344900, 33826413805700, 33667177382020, 34185891283076, 33826440544388, 33826422325380, 34304124059780, 39349425373316, 33667308617860, 33951139659908, 34370916253828, 33826428813444, 33863279378564, 33667241181316, 33826415345796, 33667300491396, 33719436574852, 33803539808388, 33826416656516, 33826450178180, 33826455650436, 39483175731332, 39326057988228, 33977920979076, 33667206250628, 34284955762820, 33667182035076, 33826422194308, 33803539939460, 33667256811652, 33803542757508, 34037939404932, 34335774343300, 33826461417604, 33667308748932, 33826457649284, 33826412331140, 33826434252932, 33667214049412, 34335774376068, 34304124190852, 33618967036036, 34284952879236, 33826414100612, 33719435591812, 33667293511812, 33667293085828, 33826413641860, 33667270934660, 39326057922692, 33667179020420, 34173018505348, 33949926883460, 33973246328964, 39349425307780, 33719434018948, 33667281813636, 33667185672324, 34370916089988, 33618905202820, 33667271032964, 33667302686852, 33667242688644, 33667166404740, 33667175153796, 33618956746884, 39310694547588, 33719428710532, 33826426290308, 33719429431428, 33719435395204, 33667175284868, 33863279280260, 34534188220548, 34037940027524, 33826415804548, 33618907005060, 34335774441604, 33618897666180, 33667302588548, 33618874040452, 33667300360324, 39518365843588, 39336623964292, 33618999902340, 33719434182788, 33618995413124, 34529810251908, 33826460827780, 34284954714244, 34537144418436, 34284954812548, 34304342818948, 33667192717444, 34246817382532, 34083934339204, 33667178594436, 34284951601284, 39318968008836, 33667281289348, 33826429108356, 34304342884484, 33826439757956, 34037939503236, 39518365810820, 39517282730116, 34284954878084, 33667298590852, 33826433859716, 34304124125316, 33667236069508, 34304124158084, 33826460336260, 34037939732612, 33618860179588, 34441183035524, 34370915893380, 34370916909188, 34284953796740, 33826428747908, 39686337364100, 33667229253764, 33667256877188, 33826464661636, 33719433822340, 34037940125828, 33667192324228, 33826467250308, 33826460532868, 34137389039748, 33826474786948, 33667299704964, 34037939667076, 33667192553604, 33803540627588, 33977921241220, 33863279247492, 33826412494980, 33667312287876, 33667310256260, 33863279181956, 33890088812676, 33814259695748, 33667192127620, 33667311992964, 34519124279428, 33667215097988, 33667274506372, 33667178496132, 33667308093572, 33667286827140, 33667286728836, 33667304685700, 39830154772612, 33667287285892, 33667271229572, 34502822199428, 33667290202244, 33667305078916, 34304124092548, 33618990039172, 33667168796804, 33667229352068, 33619010060420, 33619002785924, 33618990530692, 33618950357124, 33667270738052, 33618963857540, 33618905923716, 33618967560324, 33618835964036, 33618835243140, 33618997379204, 33618981421188, 33667268706436, 33667294855300, 33667295084676, 33618968641668, 33619002556548, 33618998919300, 33618830196868, 33618998853764, 33618977095812, 33618976342148, 33618846580868, 33618996592772, 33618912346244, 33618924339332, 33667178659972, 33667178561668, 33667271295108, 33667218604164, 33667195437188, 33667168829572, 33618979389572, 33618997805188, 33618995445892, 33618972704900, 33618962284676, 33618957041796, 33618948522116, 33618909495428, 33618906579076, 33618905825412, 33618902450308, 33618895732868, 33618882822276, 33667166437508, 33667222438020, 34529809563780, 33618996723844, 33619008749700, 33619009470596, 33618937544836, 33618923061380, 33618862440580, 33618893570180, 33618891145348, 33619012288644, 33618849693828]
    -----------------------
                   id order_date                  ordered_items  items_per_order  null_items_count  ratio_null_items
    10  2204073066628 2020-04-30  [33618849693828, 336188601...               14                14               1.0
    20  2204707520644 2020-04-30  [33618835243140, 336188359...               25                25               1.0
    21  2204838822020 2020-04-30  [33618891145348, 336188935...               15                15               1.0
    34  2208967852164 2020-05-01  [33618830196868, 336188465...               10                10               1.0
    49  2215889436804 2020-05-03  [33667166699652, 336671666...               10                 8               0.8



```python
#how count_null_items function works
my_list=[1,2,3,4,5]

values=[1,4]

sum(item in my_list for item in values )
```




    2




```python
orders_df['order_date_month']= orders_df['order_date'].dt.to_period('M')

monthly_summary = orders_df.groupby('order_date_month').agg(
    items_per_order_month=('items_per_order', 'sum'),
    null_items_count_month=('null_items_count', 'sum')
).reset_index()

monthly_summary['missing_ratio_month'] = (
    monthly_summary['null_items_count_month'] / monthly_summary['items_per_order_month']
)

monthly_summary['order_date_month'] = monthly_summary['order_date_month'].dt.to_timestamp()
monthly_summary.head()

plt.figure(figsize=(12,6))
plt.plot(monthly_summary['order_date_month'], monthly_summary['missing_ratio_month'], marker='o', label='Null Items Count')
plt.title('Null Items Ratio Over Time')
plt.tight_layout()
plt.show()

```


    
![png](module_2_EDA_files/module_2_EDA_23_0.png)
    


It's clear that the null items (non existing variant_ids in our inventory) are products that are no longer available


```python
#let's try a market basket analysis to discover products bought together
from mlxtend.preprocessing import TransactionEncoder

transactions= orders_df['ordered_items'].tolist()
te=TransactionEncoder()
te_ary=te.fit(transactions).transform(transactions)

orders_df_encoded=pd.DataFrame(te_ary, columns=te.columns_)
#print(orders_df_encoded)

from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(orders_df_encoded, min_support=0.02, use_colnames=True)
print(frequent_itemsets.head())
print('--------------------------------------')

from mlxtend.frequent_patterns import association_rules
num_itemsets = len(frequent_itemsets)
rules= association_rules(frequent_itemsets, metric='lift', min_threshold=1.0, num_itemsets=num_itemsets)
print(rules)
print('--------------------------------------')
filtered_rules = rules[['antecedents','consequents','support','confidence','lift']].copy()
variant_id_to_product_type= inventory_df.set_index('variant_id')['product_type'].to_dict()
def replace_variant_with_product_type(itemset, mapping):
    return frozenset({mapping.get(int(item), item) for item in itemset})

filtered_rules['antecedents'] = filtered_rules['antecedents'].apply(lambda x: replace_variant_with_product_type(x, variant_id_to_product_type))
filtered_rules['consequents'] = filtered_rules['consequents'].apply(lambda x: replace_variant_with_product_type(x, variant_id_to_product_type))
print(filtered_rules)
```

        support          itemsets
    0  0.023823  (33667185279108)
    1  0.025647  (33667206283396)
    2  0.047874  (33667207266436)
    3  0.035564  (33667222798468)
    4  0.020062  (33667247276164)
    --------------------------------------
            antecedents       consequents  antecedent support  consequent support   support  confidence      lift  representativity  leverage  conviction  zhangs_metric   jaccard  certainty  kulczynski
    0  (34284949766276)  (39284117930116)            0.070329            0.302975  0.024507    0.348460  1.150129               1.0  0.003199    1.069812       0.140407  0.070261   0.065256    0.214674
    1  (39284117930116)  (34284949766276)            0.302975            0.070329  0.024507    0.080888  1.150129               1.0  0.003199    1.011488       0.187270  0.070261   0.011357    0.214674
    2  (39284117930116)  (34284950356100)            0.302975            0.063946  0.029180    0.096313  1.506157               1.0  0.009806    1.035816       0.482133  0.086399   0.034578    0.276321
    3  (34284950356100)  (39284117930116)            0.063946            0.302975  0.029180    0.456328  1.506157               1.0  0.009806    1.282069       0.359016  0.086399   0.220011    0.276321
    4  (39284117930116)  (39459279929476)            0.302975            0.056993  0.023709    0.078254  1.373050               1.0  0.006442    1.023066       0.389792  0.070508   0.022546    0.247127
    5  (39459279929476)  (39284117930116)            0.056993            0.302975  0.023709    0.416000  1.373050               1.0  0.006442    1.193536       0.288115  0.070508   0.162153    0.247127
    6  (39284117930116)  (39711187894404)            0.302975            0.035906  0.023367    0.077126  2.148011               1.0  0.012489    1.044665       0.766763  0.074061   0.042755    0.363960
    7  (39711187894404)  (39284117930116)            0.035906            0.302975  0.023367    0.650794  2.148011               1.0  0.012489    1.996026       0.554358  0.074061   0.499005    0.363960
    --------------------------------------
                         antecedents                    consequents   support  confidence      lift
    0  (toilet-roll-kitchen-roll-...               (39284117930116)  0.024507    0.348460  1.150129
    1               (39284117930116)  (toilet-roll-kitchen-roll-...  0.024507    0.080888  1.150129
    2               (39284117930116)                  (dishwashing)  0.029180    0.096313  1.506157
    3                  (dishwashing)               (39284117930116)  0.029180    0.456328  1.506157
    4               (39284117930116)                       (dental)  0.023709    0.078254  1.373050
    5                       (dental)               (39284117930116)  0.023709    0.416000  1.373050
    6               (39284117930116)               (39711187894404)  0.023367    0.077126  2.148011
    7               (39711187894404)               (39284117930116)  0.023367    0.650794  2.148011


In this case, the Rule 0 says that if someone buy product 34284949766276, there is a 34.85% probability to also buy 39284117930116.

This analysis is not very relevant; all the pairs of related products have a missing product and it doesn't bring information at the present time

### Insights

* Most orders have between 8 and 15 items (average of 12), but there are outliers of up to 114 items, suggesting opportunities to serve wholesale customers or drive larger purchases with strategic promotions.

* 28.3 % of our clients repeated buying in our platform.

* Some products (variant_id) appear in the orders but not in the inventory, these are products that were available at some point, but not currently.

### Problems & comments

* Variant_ids shown in orders dataset not registered in the inventory.

* We could also extract some informationfrom the order_date column combined with users that repeated buying to discover how often they buy.

# ***Regulars dataset***

* Include info about each variant_id

* Find popular regular items

* Find users that added more regular items (we can identify the most interested and active clients)


```python
regulars_path = '/home/alex/zriveAM/zrive-ds/aws/data/regulars.parquet'
regulars_df = pd.read_parquet(regulars_path)
regulars_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>variant_id</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>68e872ff888303bff58ec56a3a...</td>
      <td>33618848088196</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>11</th>
      <td>aed88fc0b004270a62ff1fe4b9...</td>
      <td>33667178659972</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>18</th>
      <td>68e872ff888303bff58ec56a3a...</td>
      <td>33619009208452</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>46</th>
      <td>aed88fc0b004270a62ff1fe4b9...</td>
      <td>33667305373828</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>47</th>
      <td>4594e99557113d5a1c5b59bf31...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Dataframe info:")
print(regulars_df.info())
```

    Dataframe info:
    <class 'pandas.core.frame.DataFrame'>
    Index: 18105 entries, 3 to 37720
    Data columns (total 3 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   user_id     18105 non-null  object        
     1   variant_id  18105 non-null  int64         
     2   created_at  18105 non-null  datetime64[us]
    dtypes: datetime64[us](1), int64(1), object(1)
    memory usage: 565.8+ KB
    None



```python
#use dictionary to include product_name of variant_id
regulars_combined_df = pd.merge(regulars_df,inventory_df,on='variant_id',how='left')
regulars_combined_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>variant_id</th>
      <th>created_at</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>68e872ff888303bff58ec56a3a...</td>
      <td>33618848088196</td>
      <td>2020-04-30 15:07:03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aed88fc0b004270a62ff1fe4b9...</td>
      <td>33667178659972</td>
      <td>2020-05-05 23:34:35</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68e872ff888303bff58ec56a3a...</td>
      <td>33619009208452</td>
      <td>2020-04-30 15:07:03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aed88fc0b004270a62ff1fe4b9...</td>
      <td>33667305373828</td>
      <td>2020-05-05 23:34:35</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4594e99557113d5a1c5b59bf31...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
      <td>3.49</td>
      <td>3.5</td>
      <td>method</td>
      <td>cleaning-products</td>
      <td>[cruelty-free, eco, vegan,...</td>
    </tr>
  </tbody>
</table>
</div>



As in the previous case, there are products that are no longer available, the best option is get rid of them


```python
regulars_combined_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 18105 entries, 0 to 18104
    Data columns (total 8 columns):
     #   Column            Non-Null Count  Dtype         
    ---  ------            --------------  -----         
     0   user_id           18105 non-null  object        
     1   variant_id        18105 non-null  int64         
     2   created_at        18105 non-null  datetime64[us]
     3   price             15034 non-null  float64       
     4   compare_at_price  15034 non-null  float64       
     5   vendor            15034 non-null  object        
     6   product_type      15034 non-null  object        
     7   tags              15034 non-null  object        
    dtypes: datetime64[us](1), float64(2), int64(1), object(4)
    memory usage: 1.1+ MB



```python
cleaned_regulars_combined_df=regulars_combined_df.dropna()
cleaned_regulars_combined_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>variant_id</th>
      <th>created_at</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>4594e99557113d5a1c5b59bf31...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
      <td>3.49</td>
      <td>3.50</td>
      <td>method</td>
      <td>cleaning-products</td>
      <td>[cruelty-free, eco, vegan,...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4594e99557113d5a1c5b59bf31...</td>
      <td>33667182493828</td>
      <td>2020-05-06 14:42:11</td>
      <td>4.29</td>
      <td>5.40</td>
      <td>bulldog</td>
      <td>skincare</td>
      <td>[cruelty-free, eco, facial...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>d883991facbc3b07b62da342d0...</td>
      <td>33667198910596</td>
      <td>2020-07-06 10:12:08</td>
      <td>14.99</td>
      <td>16.55</td>
      <td>ecover</td>
      <td>dishwashing</td>
      <td>[cruelty-free, dishwasher-...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>66a195720d6988ff4d32155cc0...</td>
      <td>33826459320452</td>
      <td>2020-07-06 17:17:52</td>
      <td>5.09</td>
      <td>5.65</td>
      <td>treeoflife</td>
      <td>snacks-confectionery</td>
      <td>[christmas, nuts-dried-fru...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0b7e02fee4b9e215da3bdae700...</td>
      <td>33667247276164</td>
      <td>2020-07-18 16:56:55</td>
      <td>2.49</td>
      <td>3.00</td>
      <td>method</td>
      <td>hand-soap-sanitisers</td>
      <td>[cruelty-free, eco, hand-s...</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(regulars_combined_df['variant_id'].value_counts())
```




    1843




```python
len(cleaned_regulars_combined_df['variant_id'].value_counts())
```




    1285



From the initial 1843 products in the regulars section, 558 are no longer available


```python
#find the product (variant_id) more requested
requested_products = cleaned_regulars_combined_df['variant_id'].value_counts().reset_index()
requested_products.columns=['variant_id', 'request_count']
requested_products= pd.merge(requested_products, inventory_df, on='variant_id', how='left')
requested_products.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>request_count</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34081589887108</td>
      <td>253</td>
      <td>10.79</td>
      <td>11.94</td>
      <td>oatly</td>
      <td>long-life-milk-substitutes</td>
      <td>[oat-milk, vegan]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34370361229444</td>
      <td>127</td>
      <td>4.99</td>
      <td>5.50</td>
      <td>whogivesacrap</td>
      <td>toilet-roll-kitchen-roll-t...</td>
      <td>[b-corp, eco, toilet-rolls]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34370915041412</td>
      <td>112</td>
      <td>4.99</td>
      <td>6.60</td>
      <td>mutti</td>
      <td>tins-packaged-foods</td>
      <td>[pasta-pizza-sauce, tinned...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34284951863428</td>
      <td>105</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-t...</td>
      <td>[kitchen-roll]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33667282436228</td>
      <td>101</td>
      <td>3.99</td>
      <td>4.00</td>
      <td>thecheekypanda</td>
      <td>toilet-roll-kitchen-roll-t...</td>
      <td>[b-corp, cruelty-free, eco...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#users that specified more items
users_activity= cleaned_regulars_combined_df['user_id'].value_counts().reset_index()
users_activity.columns=['user_id', 'number_of_regular_products']
users_activity.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>number_of_regular_products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a124c8bb0453ea0957405b7a08...</td>
      <td>701</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ba068a3398230c10a98868ced1...</td>
      <td>455</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9b5b3679033da9e1f3a4def186...</td>
      <td>393</td>
    </tr>
    <tr>
      <th>3</th>
      <td>257be7ae940425880bbb20bf16...</td>
      <td>364</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9a4b53affbe91ca5fd0c97f6d8...</td>
      <td>359</td>
    </tr>
  </tbody>
</table>
</div>




```python
#find repeated users-product combination
recurrent_purchases = cleaned_regulars_combined_df.groupby(['user_id', 'variant_id']).size().reset_index(name='added_to_regulars')
pd.set_option('display.max_colwidth', 150)
print(len(recurrent_purchases))
print(recurrent_purchases)
print(len(recurrent_purchases[recurrent_purchases['added_to_regulars'] == 1]))
print(len(recurrent_purchases[recurrent_purchases['added_to_regulars'] > 1]))

```

    9837
                                                                                                                                   user_id      variant_id  added_to_regulars
    0     004b3e3cb9a9f5b0974ce4179db394057c72e7a82077bfe6a28af9e6306ebc51b0d8c5c8bd4c9b59ebb3237827df723745c1374d12ad2053e0131edc184df17d  33667274997892                  1
    1     005743eefffa4ce840608c4f47b8c548b134d89be5c39020ea20c4e708544b2dbd94e4b662b34c7c8b2ba1557a54454e45a7349bd7c024d9f5def354d3f38c53  34081589887108                  1
    2     005743eefffa4ce840608c4f47b8c548b134d89be5c39020ea20c4e708544b2dbd94e4b662b34c7c8b2ba1557a54454e45a7349bd7c024d9f5def354d3f38c53  34519123951748                  1
    3     0074992079c1836c6509eec748a973dc97388b4877e770f57a3dc05917641897fd65ed9ab35168ce32f7c428c3e048206e12a653f40d48d7eb7db3570b4521b4  33667247243396                  1
    4     0074992079c1836c6509eec748a973dc97388b4877e770f57a3dc05917641897fd65ed9ab35168ce32f7c428c3e048206e12a653f40d48d7eb7db3570b4521b4  33667289514116                  1
    ...                                                                                                                                ...             ...                ...
    9832  fffd9f989509e36d1fc3e3e53627d6341482f385052a034a897249a5455c66475dfc78fc8eec13b742ada69537b20dc5b24cc59864ee21c50816f1131cf10527  39320912003204                  1
    9833  fffd9f989509e36d1fc3e3e53627d6341482f385052a034a897249a5455c66475dfc78fc8eec13b742ada69537b20dc5b24cc59864ee21c50816f1131cf10527  39459282354308                  1
    9834  fffd9f989509e36d1fc3e3e53627d6341482f385052a034a897249a5455c66475dfc78fc8eec13b742ada69537b20dc5b24cc59864ee21c50816f1131cf10527  39496806072452                  1
    9835  fffd9f989509e36d1fc3e3e53627d6341482f385052a034a897249a5455c66475dfc78fc8eec13b742ada69537b20dc5b24cc59864ee21c50816f1131cf10527  39506484461700                  4
    9836  fffd9f989509e36d1fc3e3e53627d6341482f385052a034a897249a5455c66475dfc78fc8eec13b742ada69537b20dc5b24cc59864ee21c50816f1131cf10527  39590266536068                  4
    
    [9837 rows x 3 columns]
    7739
    2098


According to the information above, there are some users (2098) that added the same product multiple times, this could give rise to confusion, because at the end it's only 1 product


```python

user_id_filter = "fffd9f989509e36d1fc3e3e53627d6341482f385052a034a897249a5455c66475dfc78fc8eec13b742ada69537b20dc5b24cc59864ee21c50816f1131cf10527"
variant_id_filter = 39590266536068

filtered_rows = cleaned_regulars_combined_df[
    (cleaned_regulars_combined_df['user_id'] == user_id_filter) & 
    (cleaned_regulars_combined_df['variant_id'] == variant_id_filter)
]


print(filtered_rows)
```

                                                                                                                                    user_id      variant_id          created_at  price  compare_at_price    vendor  \
    1805   fffd9f989509e36d1fc3e3e53627d6341482f385052a034a897249a5455c66475dfc78fc8eec13b742ada69537b20dc5b24cc59864ee21c50816f1131cf10527  39590266536068 2022-01-20 09:27:43   2.49              2.59  bacofoil   
    5786   fffd9f989509e36d1fc3e3e53627d6341482f385052a034a897249a5455c66475dfc78fc8eec13b742ada69537b20dc5b24cc59864ee21c50816f1131cf10527  39590266536068 2021-10-28 13:23:35   2.49              2.59  bacofoil   
    11085  fffd9f989509e36d1fc3e3e53627d6341482f385052a034a897249a5455c66475dfc78fc8eec13b742ada69537b20dc5b24cc59864ee21c50816f1131cf10527  39590266536068 2022-02-04 10:56:52   2.49              2.59  bacofoil   
    13165  fffd9f989509e36d1fc3e3e53627d6341482f385052a034a897249a5455c66475dfc78fc8eec13b742ada69537b20dc5b24cc59864ee21c50816f1131cf10527  39590266536068 2022-01-20 09:39:11   2.49              2.59  bacofoil   
    
                        product_type          tags  
    1805   food-bags-cling-film-foil  [cling-film]  
    5786   food-bags-cling-film-foil  [cling-film]  
    11085  food-bags-cling-film-foil  [cling-film]  
    13165  food-bags-cling-film-foil  [cling-film]  


## Insights

* long-life-milk-substitutes is the main product added to regulars, also the main product in orders, it's by far the most important product for us

# ***Abandoned_carts data***

* Find most frequent items in abandoned carts (and distribution)

* Find users with most abandoned_carts

* Find days with most abandoned_carts


```python
abdandoned_carts_path = '/home/alex/zriveAM/zrive-ds/aws/data/abandoned_carts.parquet'
abandoned_carts_df = pd.read_parquet(abdandoned_carts_path)
abandoned_carts_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>variant_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12858560217220</td>
      <td>5c4e5953f13ddc3bc9659a3453356155e5efe4739d7a2b4130557c2e49c1193921975a2116d57fefbf911523ce44b6b6e0f8acbf598b36d0e4fc2727ec89378a</td>
      <td>2020-05-20 13:53:24</td>
      <td>[33826459287684, 33826457616516, 33667192127620, 33826412331140, 33826472558724, 33826427240580, 33826474590340, 33826457157764, 33667198976132, 3...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20352449839236</td>
      <td>9d6187545c005d39e44d0456d87790db18611d7c7379bd2fb2c4c9ab335534542fecf99d201921dece1889ed054e5b5e8cd87a564815c04b07f6f53ee96861fc</td>
      <td>2021-06-27 05:24:13</td>
      <td>[34415988179076, 34037940158596, 34502822363268, 33719435722884, 33803537973380, 39459279929476, 34284955304068, 34284952813700, 39542989619332]</td>
    </tr>
    <tr>
      <th>45</th>
      <td>20478401413252</td>
      <td>e83fb0273d70c37a2968fee107113698fd4f389c442c0b432bca5c2bad4fb803f6c24f90edb0dcb6e64d5791b9bfa389a738b7bad0f2cb19a5f6e4c4ae931009</td>
      <td>2021-07-18 08:23:49</td>
      <td>[34543001337988, 34037939372164, 34113606090884, 34284952584324, 34221708673156, 39336624193668, 33667247145092, 39403031167108, 33951139135620, 3...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>20481783103620</td>
      <td>10c42e10e530284b7c7c50f3a23a98726d5747b812808423874f43384ffdbfb574e3ccc11706aece4f1329b4365fe5bce1a01827d382852daedaf18a2e22fcb8</td>
      <td>2021-07-18 21:29:36</td>
      <td>[33667268116612, 34037940224132, 34436055203972, 39403032117380, 33667207266436, 34284951273604, 34284951240836, 39403031691396, 39418337591428, 3...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>20485321687172</td>
      <td>d9989439524b3f6fc4f41686d043f315fb408b954d6153a40c83cfa13aa8faebe1d7a2835a2e9c5120b806bc789704622081a413af5c418297da0516c892d756</td>
      <td>2021-07-19 12:17:05</td>
      <td>[33667268083844, 34284950454404, 33973246886020]</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Dataframe info:")
print(abandoned_carts_df.info())
```

    Dataframe info:
    <class 'pandas.core.frame.DataFrame'>
    Index: 5457 entries, 0 to 70050
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   id          5457 non-null   int64         
     1   user_id     5457 non-null   object        
     2   created_at  5457 non-null   datetime64[us]
     3   variant_id  5457 non-null   object        
    dtypes: datetime64[us](1), int64(1), object(2)
    memory usage: 213.2+ KB
    None



```python
#most frequent items in abandoned_carts
all_abandoned_products = [product for sublist in abandoned_carts_df['variant_id'] for product in sublist]
print(type(all_abandoned_products))

abandoned_product_series = pd.Series(all_abandoned_products)
print(abandoned_product_series.dtype)
abandoned_product_counts = abandoned_product_series.value_counts().reset_index()
abandoned_product_counts.columns = ['variant_id', 'Frequency']
abandoned_product_counts = abandoned_product_counts.sort_values(by='Frequency', ascending=False)
print(abandoned_product_counts.head(15))
```

    <class 'list'>
    int64
            variant_id  Frequency
    0   34081589887108        608
    1   34284951863428        478
    2   34284950356100        409
    3   34137590366340        395
    4   34284949766276        382
    5   34284950519940        307
    6   34284950454404        306
    7   39459279929476        305
    8   39284117930116        298
    9   34037939372164        290
    10  39709997760644        277
    11  39405098369156        256
    12  34370361229444        253
    13  34543001370756        242
    14  34543001337988        241



```python
top_abandoned_products = abandoned_product_counts.head(250)

plt.figure(figsize=(12, 8))
sns.barplot(
    x='variant_id', 
    y='Frequency', 
    data=top_abandoned_products, 
    order=top_abandoned_products['variant_id']
)


plt.title('Top abandoned products')
plt.xlabel('Product (variant_id)')
plt.ylabel('Times abandoned')
plt.xticks([])
plt.tight_layout()
plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_49_0.png)
    



```python
#Complete abandoned_product_counts dataframe with information from inventory
combined_abandoned_df = pd.merge(abandoned_product_counts,inventory_df,on='variant_id',how='left')
combined_abandoned_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>Frequency</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34081589887108</td>
      <td>608</td>
      <td>10.79</td>
      <td>11.94</td>
      <td>oatly</td>
      <td>long-life-milk-substitutes</td>
      <td>[oat-milk, vegan]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34284951863428</td>
      <td>478</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[kitchen-roll]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34284950356100</td>
      <td>409</td>
      <td>1.99</td>
      <td>3.00</td>
      <td>fairy</td>
      <td>dishwashing</td>
      <td>[discontinue, swapped, washing-up-liquid]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34137590366340</td>
      <td>395</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34284949766276</td>
      <td>382</td>
      <td>8.49</td>
      <td>9.00</td>
      <td>andrex</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[toilet-rolls]</td>
    </tr>
  </tbody>
</table>
</div>




```python
#users with most abandoned_carts
users_abandoned_carts = abandoned_carts_df['user_id'].value_counts().reset_index()
users_abandoned_carts.columns=['user_id', 'abandoned_carts_count']

users_path = '/home/alex/zriveAM/zrive-ds/aws/data/users.parquet'
users_df = pd.read_parquet(users_path)

combined_users_abandoned_carts = pd.merge(users_abandoned_carts,users_df,on='user_id',how='left')
combined_users_abandoned_carts.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>abandoned_carts_count</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>257be7ae940425880bbb20bf162c2616b32881bf0a8bda4e4ce7f5ce0356c29b06e032dd593030a460d20c71353737305e51431a7df58ea0284c67c95084a42b</td>
      <td>10</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2021-06-10 15:00:55</td>
      <td>2021-06-01 00:00:00</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1dacfd2a360677052d8605f843ae410dd23b0ddb7f506cc447fa78ee26bb7182ea72a8107f16633ac611c1142e90dd47d4afec7f6374b8d7b1a29492d37dd51a</td>
      <td>9</td>
      <td>Proposition</td>
      <td>UKG</td>
      <td>2021-08-06 09:23:27</td>
      <td>2021-08-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fffd9f989509e36d1fc3e3e53627d6341482f385052a034a897249a5455c66475dfc78fc8eec13b742ada69537b20dc5b24cc59864ee21c50816f1131cf10527</td>
      <td>7</td>
      <td>Top Up</td>
      <td>UKF</td>
      <td>2021-10-21 18:17:53</td>
      <td>2021-10-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a8ea4d1ff9cfc5005b7354d1d17564347dd842bab2a6c39a52301046e9104a2819f65f03343f0e92a7a20cc8f1cc29d0fcf614a0f9601064b9fc684b0a908728</td>
      <td>7</td>
      <td>Top Up</td>
      <td>UKF</td>
      <td>2022-01-31 21:11:09</td>
      <td>2022-01-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>97e81469f5758878f4d7eaa3af6b4fc37b2b5c2255881189c9bc6b4e7a2f2b63ac0c376b6116046cae8f6abc9a25f5eb28204a58c92f7e16635781cb216aa755</td>
      <td>7</td>
      <td>Proposition</td>
      <td>UKK</td>
      <td>2021-11-19 21:02:15</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
abandoned_carts_df_per_nuts=combined_users_abandoned_carts.groupby('user_nuts1')['abandoned_carts_count'].sum().reset_index()
abandoned_carts_df_per_nuts.sort_values(by='user_nuts1', inplace=True)
print(abandoned_carts_df_per_nuts)
```

       user_nuts1  abandoned_carts_count
    0         UKC                    116
    1         UKD                    371
    2         UKE                    349
    3         UKF                    298
    4         UKG                    321
    5         UKH                    457
    6         UKI                   1451
    7         UKJ                    772
    8         UKK                    657
    9         UKL                    257
    10        UKM                    358



```python
plt.figure(figsize=(12,6))
sns.barplot(x=abandoned_carts_df_per_nuts['user_nuts1'], y=abandoned_carts_df_per_nuts['abandoned_carts_count'])
plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_53_0.png)
    


This analysis would make more sense by calculating a ratio, as these results may be conditioned by the number of orders in each region.


```python
orders_df = pd.read_parquet(orders_path)
orders_df.head(5)
users_df = pd.read_parquet(users_path)

orders_per_nuts= pd.merge(orders_df, users_df, on='user_id', how='left')
orders_per_nuts=orders_per_nuts[['id', 'user_nuts1']]
orders_per_nuts.head()

orders_per_nuts= orders_per_nuts.groupby('user_nuts1')['id'].count().reset_index()
orders_per_nuts.columns=['user_nuts1', 'orders_count']
print(orders_per_nuts.sort_values(by='user_nuts1', inplace=True))
print(orders_per_nuts)
```

    None
       user_nuts1  orders_count
    0         UKC           205
    1         UKD           513
    2         UKE           467
    3         UKF           455
    4         UKG           477
    5         UKH           860
    6         UKI          2177
    7         UKJ          1427
    8         UKK          1148
    9         UKL           382
    10        UKM           584
    11        UKN             4



```python
#calculate abandonments compared to num of orders for each region
ratio_abandonments_per_orders= abandoned_carts_df_per_nuts ['abandoned_carts_count'] / orders_per_nuts['orders_count']
ratio_abandonments_per_orders
```




    0     0.565854
    1     0.723197
    2     0.747323
    3     0.654945
    4     0.672956
    5     0.531395
    6     0.666514
    7     0.540995
    8     0.572300
    9     0.672775
    10    0.613014
    11         NaN
    dtype: float64




```python
plt.figure(figsize=(12,6))
sns.barplot(x=abandoned_carts_df_per_nuts['user_nuts1'], y=ratio_abandonments_per_orders)
plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_57_0.png)
    


Indeed, the ratio is similar in all regions

### Insights

* Abandoned carts are related to orders, the best seller product long-life-milk-substitutes	is also the most abandoned product 

# ***Inventory data***

* Price and compare_at_price distribution

* Find products without price and without discount

* Top vendors

* Top product type

* Top tags


```python
inventory_path = '/home/alex/zriveAM/zrive-ds/aws/data/inventory.parquet'
inventory_df = pd.read_parquet(inventory_path)
inventory_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34370361229444</td>
      <td>4.99</td>
      <td>5.50</td>
      <td>whogivesacrap</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, eco, toilet-rolls]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34284951863428</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[kitchen-roll]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33667283583108</td>
      <td>1.79</td>
      <td>1.99</td>
      <td>thecheekypanda</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, cruelty-free, eco, tissue, vegan]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33803537973380</td>
      <td>1.99</td>
      <td>2.09</td>
      <td>colgate</td>
      <td>dental</td>
      <td>[dental-accessories]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>34460976447620</td>
      <td>2.79</td>
      <td>2.85</td>
      <td>carex</td>
      <td>hand-soap-sanitisers</td>
      <td>[hand-soap, refills]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>33667202121860</td>
      <td>8.99</td>
      <td>12.55</td>
      <td>ecover</td>
      <td>washing-liquid-gel</td>
      <td>[cruelty-free, eco, vegan, washing-liquid-gel]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>39478260695172</td>
      <td>1.99</td>
      <td>2.00</td>
      <td>napolina</td>
      <td>cooking-sauces</td>
      <td>[pasta-pizza-sauce, vegan]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>39772627533956</td>
      <td>1.99</td>
      <td>2.30</td>
      <td>thepinkstuff</td>
      <td>cleaning-products</td>
      <td>[all-purpose-cleaner, vegan]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>39887964766340</td>
      <td>2.59</td>
      <td>2.69</td>
      <td>profusion</td>
      <td>tins-packaged-foods</td>
      <td>[gluten-free, meat-alternatives, organic, vegan]</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("Dataframe info:")
print(inventory_df.info())
print("----------------------------------")
print("Statistical info")
print(inventory_df.describe())

```

    Dataframe info:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1733 entries, 0 to 1732
    Data columns (total 6 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   variant_id        1733 non-null   int64  
     1   price             1733 non-null   float64
     2   compare_at_price  1733 non-null   float64
     3   vendor            1733 non-null   object 
     4   product_type      1733 non-null   object 
     5   tags              1733 non-null   object 
    dtypes: float64(2), int64(1), object(3)
    memory usage: 81.4+ KB
    None
    ----------------------------------
    Statistical info
             variant_id        price  compare_at_price
    count  1.733000e+03  1733.000000       1733.000000
    mean   3.694880e+13     6.307351          7.028881
    std    2.725674e+12     7.107218          7.660542
    min    3.361529e+13     0.000000          0.000000
    25%    3.427657e+13     2.490000          2.850000
    50%    3.927260e+13     3.990000          4.490000
    75%    3.948318e+13     7.490000          8.210000
    max    4.016793e+13    59.990000         60.000000



```python
#Discover, total items, items without price and items without discount
inventory_df['discount']=inventory_df['compare_at_price'] - inventory_df['price']
inventory_df=inventory_df[['variant_id', 'price', 'compare_at_price','discount', 'vendor', 'product_type', 'tags']]
print('Total items:', len(inventory_df))
print(inventory_df.head())
print("--------------------------------")

no_price_products_df=inventory_df[inventory_df['compare_at_price'] == 0]
print("Items without price:", len(no_price_products_df))
print(no_price_products_df.head())
print("--------------------------------")

inventory_with_price_df = inventory_df[inventory_df['compare_at_price'] != 0]
print('Total items with price:', len(inventory_with_price_df))
print(inventory_with_price_df.head())
print("--------------------------------")

zero_discount_df= inventory_with_price_df[inventory_with_price_df['discount'] == 0]
print('Items with zero discount:', len(zero_discount_df))
print(zero_discount_df.head())
```

    Total items: 1733
           variant_id  price  compare_at_price  discount          vendor                     product_type                                        tags
    0  39587297165444   3.09              3.15      0.06           heinz             condiments-dressings                       [table-sauces, vegan]
    1  34370361229444   4.99              5.50      0.51   whogivesacrap  toilet-roll-kitchen-roll-tissue                 [b-corp, eco, toilet-rolls]
    2  34284951863428   3.69              3.99      0.30          plenty  toilet-roll-kitchen-roll-tissue                              [kitchen-roll]
    3  33667283583108   1.79              1.99      0.20  thecheekypanda  toilet-roll-kitchen-roll-tissue  [b-corp, cruelty-free, eco, tissue, vegan]
    4  33803537973380   1.99              2.09      0.10         colgate                           dental                        [dental-accessories]
    --------------------------------
    Items without price: 72
             variant_id  price  compare_at_price  discount         vendor       product_type                          tags
    95   40070658490500    0.0               0.0       0.0        jordans             cereal                 [cereal-bars]
    96   40167931674756    0.0               0.0       0.0  whogivesacrap                                               []
    97   40167931707524    0.0               0.0       0.0  whogivesacrap                                               []
    99   40070656786564    0.0               0.0       0.0       mcvities  biscuits-crackers                [biscuits, pm]
    100  40070657933444    0.0               0.0       0.0       astonish  cleaning-products  [bathroom-limescale-cleaner]
    --------------------------------
    Total items with price: 1661
           variant_id  price  compare_at_price  discount          vendor                     product_type                                        tags
    0  39587297165444   3.09              3.15      0.06           heinz             condiments-dressings                       [table-sauces, vegan]
    1  34370361229444   4.99              5.50      0.51   whogivesacrap  toilet-roll-kitchen-roll-tissue                 [b-corp, eco, toilet-rolls]
    2  34284951863428   3.69              3.99      0.30          plenty  toilet-roll-kitchen-roll-tissue                              [kitchen-roll]
    3  33667283583108   1.79              1.99      0.20  thecheekypanda  toilet-roll-kitchen-roll-tissue  [b-corp, cruelty-free, eco, tissue, vegan]
    4  33803537973380   1.99              2.09      0.10         colgate                           dental                        [dental-accessories]
    --------------------------------
    Items with zero discount: 35
             variant_id  price  compare_at_price  discount          vendor                     product_type                                        tags
    287  34368926023812   1.99              1.99       0.0          crespo              tins-packaged-foods                      [pickled-foods-olives]
    310  33667282075780   1.99              1.99       0.0  thecheekypanda  toilet-roll-kitchen-roll-tissue  [b-corp, cruelty-free, eco, tissue, vegan]
    321  34529809825924   8.99              8.99       0.0         cowgate                baby-milk-formula                     [ready-to-feed-bottles]
    326  40070657704068   1.00              1.00       0.0            sure                        deodorant                                 [deodorant]
    351  34514676678788   8.75              8.75       0.0         cowgate                baby-milk-formula                            [formula-powder]



```python
fig, axes=plt.subplots(nrows=1, ncols=2, figsize=(14,6))
sns.histplot(inventory_df['price'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Price distribution')

sns.histplot(inventory_df['compare_at_price'].dropna(), bins=30, kde=True, ax=axes[1])
axes[1].set_title('Compare at price distribution')

plt.tight_layout()
plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_64_0.png)
    



```python
top_vendors=inventory_with_price_df['vendor'].value_counts().reset_index()
top_vendors.columns=['vendor', 'count']
top_vendors.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vendor</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>biona</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ecover</td>
      <td>34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>faithinnature</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>method</td>
      <td>26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hiderfoods</td>
      <td>24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>greencuisine</td>
      <td>24</td>
    </tr>
    <tr>
      <th>6</th>
      <td>various</td>
      <td>23</td>
    </tr>
    <tr>
      <th>7</th>
      <td>napolina</td>
      <td>19</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ellaskitchen</td>
      <td>18</td>
    </tr>
    <tr>
      <th>9</th>
      <td>febreze</td>
      <td>17</td>
    </tr>
    <tr>
      <th>10</th>
      <td>cooksco</td>
      <td>17</td>
    </tr>
    <tr>
      <th>11</th>
      <td>heinz</td>
      <td>17</td>
    </tr>
    <tr>
      <th>12</th>
      <td>scrumbles</td>
      <td>17</td>
    </tr>
    <tr>
      <th>13</th>
      <td>tommeetippee</td>
      <td>16</td>
    </tr>
    <tr>
      <th>14</th>
      <td>bulldog</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_products=inventory_with_price_df['product_type'].value_counts().reset_index()
top_products.columns=['product_type', 'count']
top_products.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_type</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cleaning-products</td>
      <td>154</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tins-packaged-foods</td>
      <td>112</td>
    </tr>
    <tr>
      <th>2</th>
      <td>snacks-confectionery</td>
      <td>110</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cooking-ingredients</td>
      <td>73</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pasta-rice-noodles</td>
      <td>64</td>
    </tr>
    <tr>
      <th>5</th>
      <td>baby-toddler-food</td>
      <td>62</td>
    </tr>
    <tr>
      <th>6</th>
      <td>condiments-dressings</td>
      <td>52</td>
    </tr>
    <tr>
      <th>7</th>
      <td>haircare</td>
      <td>50</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cereal</td>
      <td>49</td>
    </tr>
    <tr>
      <th>9</th>
      <td>soft-drinks-mixers</td>
      <td>48</td>
    </tr>
    <tr>
      <th>10</th>
      <td>baby-kids-toiletries</td>
      <td>43</td>
    </tr>
    <tr>
      <th>11</th>
      <td>skincare</td>
      <td>42</td>
    </tr>
    <tr>
      <th>12</th>
      <td>dog-food</td>
      <td>42</td>
    </tr>
    <tr>
      <th>13</th>
      <td>baby-accessories</td>
      <td>41</td>
    </tr>
    <tr>
      <th>14</th>
      <td>dental</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_tags = inventory_with_price_df['tags'].explode()
tag_frequency = all_tags.value_counts().reset_index()
tag_frequency.columns = ['tag', 'count']
tag_frequency.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tag</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>vegan</td>
      <td>673</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gluten-free</td>
      <td>299</td>
    </tr>
    <tr>
      <th>2</th>
      <td>eco</td>
      <td>285</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cruelty-free</td>
      <td>208</td>
    </tr>
    <tr>
      <th>4</th>
      <td>organic</td>
      <td>170</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b-corp</td>
      <td>144</td>
    </tr>
    <tr>
      <th>6</th>
      <td>discontinue</td>
      <td>114</td>
    </tr>
    <tr>
      <th>7</th>
      <td>christmas</td>
      <td>108</td>
    </tr>
    <tr>
      <th>8</th>
      <td>refills</td>
      <td>101</td>
    </tr>
    <tr>
      <th>9</th>
      <td>pm</td>
      <td>68</td>
    </tr>
  </tbody>
</table>
</div>



These are the core of our inventory (in quantity), maybe not in sells and revenue

### Insights

* Our main providers are biona, ecover and faithinnature

* The top3 categories with the largest number of products available are cleaning-products, tins-packaged-foods and snacks-confectionery

* Our stock is very focused in healthy and sustainable products (vegan, gluten-free, eco, cruelty-free, organic...)

## Problems & Rework

* It would be interesting to do an analysis of the orders to check if also these products are the core of our revenue (for example long-life-milk-substitutes product-type doesn't have as many products availbable as other  product types but the best seller is of this category, so it would have an enormous impact in out revenue)

# ***Users data***

* Top Up vs Proposition clients

* Region users (user_nuts1) distribution

* User registration evolution (customer_cohort_month)


```python
users_path = '/home/alex/zriveAM/zrive-ds/aws/data/users.parquet'
users_df = pd.read_parquet(users_path)
users_df.head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2160</th>
      <td>0e823a42e107461379e5b5613b7aa00537a72e1b0eaa7a962aa3d39097d41d37b01d5089f13306c248e66a110da986a44448c007ce8e1052db8d1802f00fbd85</td>
      <td>Top Up</td>
      <td>UKH</td>
      <td>2021-05-08 13:33:49</td>
      <td>2021-05-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1123</th>
      <td>15768ced9bed648f745a7aa566a8895f7a73b9a47c1d4f65f3d519b46bc97b938812e4cea840a67b82c9ff349f086af76b5ba8171c0866103b942061d74027b1</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-17 16:30:20</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>33e0cb6eacea0775e34adbaa2c1dec16b9d6484e6b93249db254a5f358ed7c17e47bebe76903b28447ece00be7e266aad036337cdf46ccdd9ab485136d40b85d</td>
      <td>Top Up</td>
      <td>UKD</td>
      <td>2022-03-09 23:12:25</td>
      <td>2022-03-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>675</th>
      <td>57ca7591dc79825df0cecc4836a58e6062454555c86c354dd736bd34eca7eb4e588c5afea0af0bf3ee41290630bcdc6042b6fae0c19428d887b586a719825bda</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2021-04-23 16:29:02</td>
      <td>2021-04-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4694</th>
      <td>085d8e598139ce6fc9f75d9de97960fa9e1457b409ec007b214d1039e959eca61c2b5bd6dc5cc09b8147f3581366413cf9ca901906969ba6c152e44ab4cd8fb1</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-02 13:50:06</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2203</th>
      <td>ce6ca649c21f4eec31ed7489f74743e4fc756c0805c3f84ad330d2b0b762b861f014decf00e8c6b1422dbcef9f1845d54eaacda965c5b7d07e97fab58e15c219</td>
      <td>Proposition</td>
      <td>UKH</td>
      <td>2020-12-21 10:33:30</td>
      <td>2020-12-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>49</th>
      <td>f4b1822b86f82de690b19c87bf125a6bb8ff6b45a894cb2b500b5e9c48bbfe3e9719197ad6ad1bfcd07fe94ed4e06913e0b96cd58da2d21051812e8dab4debab</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-04-13 11:05:11</td>
      <td>2021-04-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4007</th>
      <td>605864ba5462c30b9ce789b768980fc0cac7fcfea0a43fb886137c7cb951dff84a1c92c28339447a3b1c47893912b65acc303ce8c4a55bf339636a5a0b85b9d8</td>
      <td>Top Up</td>
      <td>UKH</td>
      <td>2021-08-31 16:07:45</td>
      <td>2021-08-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3563</th>
      <td>bd55f50eb43f2052212c6ed171d67c094f44644b79625e78ebf19295bb9df721b2e6e3510a1be655ed013693c3f59b4e892589a215de4b2f2a8900c84c607812</td>
      <td>Top Up</td>
      <td>UKH</td>
      <td>2020-12-30 15:52:36</td>
      <td>2020-12-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>358</th>
      <td>78e1666879833f5c2d815c19aa7c19cddbc89da4917473ab4a31530ff397fce0c6da6ecc23a1d89c0bfa1389ce779e4f8e10f2cad4e7c29ad2bdf853a1e6ec53</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2020-11-19 10:12:07</td>
      <td>2020-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1647</th>
      <td>9313cef2d0393418029728a660f4cfb8638ca85ad778f376bf9cd2e590025f7325608ab398a0a1e026bc0cd49230161e2fa73ddbb7e6ad65ed6432292c4a2b3e</td>
      <td>Top Up</td>
      <td>UKH</td>
      <td>2022-02-12 18:51:14</td>
      <td>2022-02-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>745</th>
      <td>ca8f74f79ea72d1772ad10b3857ea7111ff2800fea6f8f7220f8585b7d59119fcab247b950443344c5eddbc461ab5b13e2dccf197fdb5c739741d098887017ca</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2022-03-06 16:12:59</td>
      <td>2022-03-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>285</th>
      <td>a78fdea4c816754a6c555ee13e5b1f07d3467ac3cd66cde78c5c92cf9444a21df7004710a934753191bdd0225c065681377c1df9a5eaa396d574ab120967877d</td>
      <td>Top Up</td>
      <td>UKH</td>
      <td>2021-07-30 12:13:54</td>
      <td>2021-07-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4443</th>
      <td>528b7b27de13dad8cc0309d565d88520991981320a2da4a86acb11302d3b3097caf7af576cdfc765afe6a549363011028b16d76fd60799ac006409f14596497f</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2021-10-12 13:53:47</td>
      <td>2021-10-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2354</th>
      <td>cfc54e57ef36b4c47f6143964bf29cf1af58e3b8a3c17dfd23e48de0c1b6eff8fa27bf58a0a560c467cc94eb22bc3b46701de9880f376b21b5c1f9e68ab7632b</td>
      <td>Top Up</td>
      <td>UKE</td>
      <td>2021-12-02 23:24:53</td>
      <td>2021-12-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1791</th>
      <td>a63f5747706dfff363d697d2a2e2b2e31780c596e35a000e231e4af2b47d5ea3dfaefb0e563e16775ed4ebefc77a41286b0e0ae9c8abc9ba2030c5f8f93d20b3</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2021-12-16 22:37:19</td>
      <td>2021-12-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3414</th>
      <td>e8b2d5a84381304290415638c3162129ba54ae283439e362d88ceda365a43466c37b86b527fef7b46f853b70b5caa16998a904ba382749582f30397176381882</td>
      <td>Proposition</td>
      <td>UKK</td>
      <td>2021-10-08 12:38:23</td>
      <td>2021-10-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3138</th>
      <td>348e3a94e93dcf442d9e8f466e389cfb5c127a7a992df54cbece9a1cc5d49034e0e5e66659ad960e9d60ca38ce766df22b6179c723a7ae0f4cb15f85d0e475be</td>
      <td>Top Up</td>
      <td>UKF</td>
      <td>2021-11-22 17:13:59</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3448</th>
      <td>6c37a82f217ddcbbe22c0612e95f1120ff64d70e20bd9fe5ad9e6a3ec4f99cf7d90d423f6b9d42d7c5d1cea50be8252c7e5aa778a297ae78dba03d6c8bb8238c</td>
      <td>Proposition</td>
      <td>UKK</td>
      <td>2021-07-28 11:32:37</td>
      <td>2021-07-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4594</th>
      <td>385e522707e05ab26705a6733197ec74ea567b40e321688db724ac6ca1e9462aa09cf51aca0c2d0b9d692771a7b69d00d1f7c0720d31e56e7a028688886dd5ef</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2021-06-30 19:49:06</td>
      <td>2021-06-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4158</th>
      <td>fbd43e29669a72705f6d5a6ca23e09d6a1cdf82e12cf01a06699f458fbb96d84598ecf76427f8d9c4eb9d9a699f63bce8419382ec6967088c0162ce9e20d71e4</td>
      <td>Proposition</td>
      <td>UKL</td>
      <td>2022-01-25 12:47:17</td>
      <td>2022-01-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>74</th>
      <td>f2ae8b76f88c1ef1242a10449162a197ba8c984abeef78e94c98fc589676f63acbbdef82129c42b9c7b3fb9c0f9e638eba002e2b96abdba86c84e93cbf8561d5</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-02-07 00:55:07</td>
      <td>2021-02-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1282</th>
      <td>888753f48af7d8d06d4392f772e8843b4298181c86da6057cdf7f2836a79d721d588d60c2a6ac322914a63293cac8b55efd23d5daa933ca0f3abe3c88bf0a683</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-06-23 20:05:17</td>
      <td>2021-06-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4751</th>
      <td>09d70e0b0778117aec5550c08032d56f8e06f99274168031585877e8294b9a62d07a2a817a3b5cd38852dddd3d99ce483b312bb2e128c6686766d90792929702</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2021-06-28 12:07:04</td>
      <td>2021-06-01 00:00:00</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3493</th>
      <td>5cc02bcea51a26c4257536110eb667fae1ee985c80b25a455f5f9fc6e951abfcc026997028a0916fd6ad1c12aa83f2618dd0d3b2b4bec960cdc9bf6ce84f170b</td>
      <td>Proposition</td>
      <td>UKJ</td>
      <td>2021-06-17 15:25:20</td>
      <td>2021-06-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Este dataframe es el que más variables incluye :

* user_id: id del cliente

* user_segment: algún tipo de distintivo de clientes (top up y proposition)

* user_nuts1: region de uk (UKH - East of England, UKD - North West England, etc)

* first _ordered_at: Fecha exacta primer pedido (día, mes y año)

* customer_cohort_month: Mes y año de primer pedido (se puede utilizar para analizar la retención)

* count_people: numero de personas asociadas al id

* count_adults: numero de adultos asociados al id

* count_children: numero de niños asociados al id

* count_babies: numero de bebés asociados al id

* count_pets: numero de mascotas asociados al id


```python
print("Dataframe info:")
print(users_df.info())
print("----------------------------------")
print("Null values:")
print(users_df.isnull().sum())

```

    Dataframe info:
    <class 'pandas.core.frame.DataFrame'>
    Index: 4983 entries, 2160 to 3360
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   user_id                4983 non-null   object 
     1   user_segment           4983 non-null   object 
     2   user_nuts1             4932 non-null   object 
     3   first_ordered_at       4983 non-null   object 
     4   customer_cohort_month  4983 non-null   object 
     5   count_people           325 non-null    float64
     6   count_adults           325 non-null    float64
     7   count_children         325 non-null    float64
     8   count_babies           325 non-null    float64
     9   count_pets             325 non-null    float64
    dtypes: float64(5), object(5)
    memory usage: 428.2+ KB
    None
    ----------------------------------
    Null values:
    user_id                     0
    user_segment                0
    user_nuts1                 51
    first_ordered_at            0
    customer_cohort_month       0
    count_people             4658
    count_adults             4658
    count_children           4658
    count_babies             4658
    count_pets               4658
    dtype: int64



```python
#Unique clients
unique_clients=orders_df['user_id'].value_counts()
print(len(unique_clients))
```

    4983


According to orders info, we had 4893 user_ids, and according to this dataset we have also 4983, so in this case all users_id are registered


```python
user_segment=users_df['user_segment'].value_counts()
print(user_segment)

plt.figure(figsize=(7,5))
plt.bar(user_segment.index, user_segment.values, color=['green', 'blue'])
plt.xlabel('User Segment')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

```

    user_segment
    Top Up         2643
    Proposition    2340
    Name: count, dtype: int64



    
![png](module_2_EDA_files/module_2_EDA_77_1.png)
    


However, in the first analysis I separated betwwen users with 1 order and > than 1 order

![image.png](module_2_EDA_files/image.png)

Categories top up and proposition appear to be unrelated to categories 1 order and > 1 order 


```python
user_nuts1=users_df['user_nuts1'].value_counts()
print(user_nuts1)

plt.figure(figsize=(8,4))
sns.barplot(x=user_nuts1.index, y=user_nuts1.values, hue=user_nuts1.index, legend=False, palette='magma')
plt.title('Region users distribution')
plt.xlabel('Region(nuts1)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
```

    user_nuts1
    UKI    1318
    UKJ     745
    UKK     602
    UKH     414
    UKD     358
    UKM     315
    UKE     303
    UKG     295
    UKF     252
    UKL     224
    UKC     102
    UKN       4
    Name: count, dtype: int64



    
![png](module_2_EDA_files/module_2_EDA_79_1.png)
    



```python
users_df['cohort_month'] = pd.to_datetime(users_df['customer_cohort_month']).dt.to_period('M')


cohort_counts = users_df['cohort_month'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=cohort_counts.index, y=cohort_counts.values, hue=cohort_counts.index, legend=False, palette='dark:blue')
#cohort_counts.plot(kind='bar', color='skyblue')
plt.title('User Registration by Month (Cohorts)')
plt.xlabel('Registration date')
plt.ylabel('Users')
plt.xticks(rotation=90)
plt.show()
```


    
![png](module_2_EDA_files/module_2_EDA_80_0.png)
    


This is very similar to the sales evolution

![image.png](module_2_EDA_files/image.png)


```python
print(users_df.isnull().sum())
```

    user_id                     0
    user_segment                0
    user_nuts1                 51
    first_ordered_at            0
    customer_cohort_month       0
    count_people             4658
    count_adults             4658
    count_children           4658
    count_babies             4658
    count_pets               4658
    cohort_month                0
    dtype: int64


Here obviously we have a problem, because count_people = NaN is impossible and count_people=0 is neither an option because count_people=1 exists and means that it is an individual account.

The amount of NaN could suggest that it isn't mandatory to our clients to complete this information. It would be interesting to check the evolution of NaN of this category along the time.  maybe is a feature included recently and that explain why are so many users without whis information.

If there is no solution, I think these columns should be deleted and analyze the distributions of adults, children, babies and pets with the users that included this info.

Another minor problem is the 51 missing values in user_nuts1


```python
users_df = pd.read_parquet(users_path)  # Reemplaza con el nombre real del archivo
users_df['customer_cohort_month']=pd.to_datetime(users_df['customer_cohort_month'])
users_df.sort_values(by='customer_cohort_month', inplace=True)
users_df.head(7)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3510</th>
      <td>329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700573ca76f11b45151d67944f171a88fd4f860f06d662c7b29d7b91f0dbc8bf14d410a169a0ed531040b</td>
      <td>Top Up</td>
      <td>UKF</td>
      <td>2020-04-30 18:12:30</td>
      <td>2020-04-01</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1860</th>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f545e06fba241f377a0fbf04b5efe8607e8faf52f58ad39dc6a1b66ebca64a6747002ba543652cbc664d</td>
      <td>Proposition</td>
      <td>UKM</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-01</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2475</th>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f50190a1d6741e02be71414684ca6df992c186522a0433f10367d18d7102da301989cb7929559747eda</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-01</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3083</th>
      <td>0074992079c1836c6509eec748a973dc97388b4877e770f57a3dc05917641897fd65ed9ab35168ce32f7c428c3e048206e12a653f40d48d7eb7db3570b4521b4</td>
      <td>Proposition</td>
      <td>UKI</td>
      <td>2020-05-18 21:37:52</td>
      <td>2020-05-01</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2872</th>
      <td>1296e1e72f7f43ff28d7d285f880ad4d213fa8139233c78ad2673af4cc0c1c297180254da494cc3b934858347822be437eeb072acb1403e9476dc0b2f6a83cbe</td>
      <td>Proposition</td>
      <td>UKJ</td>
      <td>2020-05-20 16:04:59</td>
      <td>2020-05-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>245</th>
      <td>dae0ada010ff9ac559172ee1c784e77d6b12a72b1df604ea503a514194edd94d88dd52195b74ef36b445aff40d1173a025a8955cd3593d577b6ef80433264edc</td>
      <td>Proposition</td>
      <td>UKK</td>
      <td>2020-05-09 14:11:59</td>
      <td>2020-05-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1501</th>
      <td>2d20842e0b680c1143a4daaabffd6c7b018b2fe4062a70d75356d4a141faffffe5f90324ad9af3d975d4ff58baa46e95d132142203e7c67484f36965e3d9f52e</td>
      <td>Proposition</td>
      <td>UKK</td>
      <td>2020-05-11 19:04:39</td>
      <td>2020-05-01</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



This discards the hypothesis that the option to specify the people associated to the account was included recently, the first users had the option and it was optional, of the first 7 clients, 5 indicated this information and 2 didn't

## Insights

* There exist 2 user_segment categories (Top Up and Proposition), almost equal distributed

* information abount adults, children, babies and pets associated to each account is optional and only 6.5 % (325 of 4983 accounts) include this info

* Customer_cohort_month is strongly related to the number of orders along the months, this is something negative, our orders strongly depend of new users (better depend of recurrent clients)

### Problems & rework

Decide what to do with count_ values

# ***Homework part 2***

# ***Feature frame dataset***


```python
feature_frame_path = '/home/alex/zriveAM/zrive-ds/aws/data/feature_frame.csv'
feature_frame_df = pd.read_csv(feature_frame_path)
```


```python
feature_frame_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808429314180</td>
      <td>3537167515780</td>
      <td>2020-10-06 10:37:05</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
print('Dataframe info:')
print(feature_frame_df.info())
```

    Dataframe info:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2880549 entries, 0 to 2880548
    Data columns (total 27 columns):
     #   Column                            Dtype  
    ---  ------                            -----  
     0   variant_id                        int64  
     1   product_type                      object 
     2   order_id                          int64  
     3   user_id                           int64  
     4   created_at                        object 
     5   order_date                        object 
     6   user_order_seq                    int64  
     7   outcome                           float64
     8   ordered_before                    float64
     9   abandoned_before                  float64
     10  active_snoozed                    float64
     11  set_as_regular                    float64
     12  normalised_price                  float64
     13  discount_pct                      float64
     14  vendor                            object 
     15  global_popularity                 float64
     16  count_adults                      float64
     17  count_children                    float64
     18  count_babies                      float64
     19  count_pets                        float64
     20  people_ex_baby                    float64
     21  days_since_purchase_variant_id    float64
     22  avg_days_to_buy_variant_id        float64
     23  std_days_to_buy_variant_id        float64
     24  days_since_purchase_product_type  float64
     25  avg_days_to_buy_product_type      float64
     26  std_days_to_buy_product_type      float64
    dtypes: float64(19), int64(4), object(4)
    memory usage: 593.4+ MB
    None



```python
print("Null values:")
print(feature_frame_df.isnull().sum())
```

    Null values:
    variant_id                          0
    product_type                        0
    order_id                            0
    user_id                             0
    created_at                          0
    order_date                          0
    user_order_seq                      0
    outcome                             0
    ordered_before                      0
    abandoned_before                    0
    active_snoozed                      0
    set_as_regular                      0
    normalised_price                    0
    discount_pct                        0
    vendor                              0
    global_popularity                   0
    count_adults                        0
    count_children                      0
    count_babies                        0
    count_pets                          0
    people_ex_baby                      0
    days_since_purchase_variant_id      0
    avg_days_to_buy_variant_id          0
    std_days_to_buy_variant_id          0
    days_since_purchase_product_type    0
    avg_days_to_buy_product_type        0
    std_days_to_buy_product_type        0
    dtype: int64


Although there are no null values, there are values=0 that are missing values


```python
# Distribution of outcome
outcome_dist = feature_frame_df['outcome'].value_counts(normalize=True)
print(outcome_dist)

plt.figure(figsize=(8,6))
sns.barplot(x=outcome_dist.index, y= outcome_dist.values)
plt.show()
```

    outcome
    0.0    0.988463
    1.0    0.011537
    Name: proportion, dtype: float64



    
![png](module_2_EDA_files/module_2_EDA_95_1.png)
    


There are different types of columns, we can group them to do a better analysis


```python
feature_frame_df.columns
```




    Index(['variant_id', 'product_type', 'order_id', 'user_id', 'created_at', 'order_date', 'user_order_seq', 'outcome', 'ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular', 'normalised_price',
           'discount_pct', 'vendor', 'global_popularity', 'count_adults', 'count_children', 'count_babies', 'count_pets', 'people_ex_baby', 'days_since_purchase_variant_id', 'avg_days_to_buy_variant_id',
           'std_days_to_buy_variant_id', 'days_since_purchase_product_type', 'avg_days_to_buy_product_type', 'std_days_to_buy_product_type'],
          dtype='object')




```python
info_cols = ["variant_id", "order_id", "user_id", "created_at", "order_date"]
label_col = "outcome"
features_cols = [col for col in feature_frame_df.columns if col not in info_cols + [label_col]]

categorical_cols = ["product_type", "vendor"]
binary_cols = ["ordered_before", "abandoned_before", "active_snoozed", "set_as_regular"]
numerical_cols = [col for col in features_cols if col not in categorical_cols + binary_cols]

```


```python
for col in binary_cols:
    print(f"Value counts {col}: {feature_frame_df[col].value_counts().to_dict()}")
    print(f"Mean outcome by {col} value: {feature_frame_df.groupby(col)[label_col].mean().to_dict()}")
    print(" ------------ ")
```

    Value counts ordered_before: {0.0: 2819658, 1.0: 60891}
    Mean outcome by ordered_before value: {0.0: 0.008223337723936732, 1.0: 0.1649669080816541}
     ------------ 
    Value counts abandoned_before: {0.0: 2878794, 1.0: 1755}
    Mean outcome by abandoned_before value: {0.0: 0.011106039542947498, 1.0: 0.717948717948718}
     ------------ 
    Value counts active_snoozed: {0.0: 2873952, 1.0: 6597}
    Mean outcome by active_snoozed value: {0.0: 0.011302554809544488, 1.0: 0.1135364559648325}
     ------------ 
    Value counts set_as_regular: {0.0: 2870093, 1.0: 10456}
    Mean outcome by set_as_regular value: {0.0: 0.010668992259135854, 1.0: 0.24971308339709258}
     ------------ 


Las features que nos indican si el usuario había interactuado con el producto antes (ordered, abandoned, snoozed, set_as_regular) estan muy desbalanceadas. Algunas como snooze o abandoned son muy extremas, quizas prodrían eliminarse o crear una meta feaure que represente si el usuario ha interactuado con el producto antes ( si cualquiera de las features anteriores es 1, meta feature es 1, si todas son 0, meta feature es 0)

These binary variables are strongly related with `outcome`.

Specifically, abandoned_before. If an item had not been abandoned before, it will only be purchased 1.11% of the time.

However, if an item had been abandoned before, it will be purchased 71.8 % of the time in a future order.


```python
#Correlation matrix

corr = feature_frame_df[numerical_cols + [label_col]].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(10,8))

cmap = sns.diverging_palette(230,20, as_cmap=True)

sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=0.3,
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink":0.5},
)
```




    <Axes: >




    
![png](module_2_EDA_files/module_2_EDA_102_1.png)
    


- Check correlations between variables

- Check correlations with outcome

Algunas variables númericas están moderadamente correladas. Algo a tener en cuenta si utilizamos modelos donde la colinearidad pueda ser importante.


```python
cols = 3
rows = int(np.ceil(len(numerical_cols) / cols))
fig, ax = plt.subplots(rows, cols, figsize=(20, 5 * rows))
ax = ax.flatten()

for i, col in enumerate(numerical_cols):
    sns.kdeplot(feature_frame_df.loc[lambda x: x.outcome == 0, col], label="0", ax=ax[i])
    sns.kdeplot(feature_frame_df.loc[lambda x: x.outcome == 1, col], label="1", ax=ax[i])
    ax[i].set_title(col)

ax[0].legend()
plt.tight_layout()


ENTENDER BIEN EL CODIGO


```


      Cell In[128], line 15
        ENTENDER BIEN EL CODIGO
                 ^
    SyntaxError: invalid syntax



En este caso representamos las distribuciones para las dos clases (outcome 0 y outcome 1) de forma separada.
Saber si la distribución es distinta para cada una de las clases es muy informativo. 

* En global_popularity, cuando outcome=1, la popularidad de los productos es mas alta (cuando la popularidad es las alta, la probabilidad de que se compre es mas alta)

* Las variables de count_ siguen un patron extraño, ya que tienen unos picos extremadamente altos, esto se debe a que los missing values de estas variables se han imputado (ya sea con la media, mediana, freq...), esto lo que hace es que perdamos info, ya que quizas los usuarios que tenian esta info estaban muy correlados con outcome=1, se deberia haber añadido una columna que indique si el valor se ha imputado o no. Sin embargo, si aumenta el poder predictivo, es una buena imputacion, a pesar de la distribucion


```python
feature_frame_df[categorical_cols].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_type</th>
      <th>vendor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2880549</td>
      <td>2880549</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>62</td>
      <td>264</td>
    </tr>
    <tr>
      <th>top</th>
      <td>tinspackagedfoods</td>
      <td>biona</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>226474</td>
      <td>146828</td>
    </tr>
  </tbody>
</table>
</div>



Tienen muchas categorias, por lo que categorical enconding empezaria por frequency encoding o algo que no aumente el numero de columnas en 62 o 264.

Es cierto, que la mejor forma de hacer coding de las variables categoricas es aquella que mejore el poder predictivo de mi modelo, por eso en el siguiente modulo se ve como crear un pipeline de entrenamiento que optimice las decisiones.



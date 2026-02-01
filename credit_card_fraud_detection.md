### Import All Libraries


```python
from idlelib.macosx import hideTkConsole

from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
from matplotlib.pyplot import clf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from xgboost import XGBClassifier


%matplotlib inline
plt.style.use('ggplot')
```

### Brief Data Check

*Notice that PCA transformation has been performed on the data to protect PII. It has resulted in 28 principal components


```python
df = pd.read_csv('creditcard.csv')
df_copy = df.copy()
print(df.head(), df.info(), df.describe())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 284807 entries, 0 to 284806
    Data columns (total 31 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   Time    284807 non-null  float64
     1   V1      284807 non-null  float64
     2   V2      284807 non-null  float64
     3   V3      284807 non-null  float64
     4   V4      284807 non-null  float64
     5   V5      284807 non-null  float64
     6   V6      284807 non-null  float64
     7   V7      284807 non-null  float64
     8   V8      284807 non-null  float64
     9   V9      284807 non-null  float64
     10  V10     284807 non-null  float64
     11  V11     284807 non-null  float64
     12  V12     284807 non-null  float64
     13  V13     284807 non-null  float64
     14  V14     284807 non-null  float64
     15  V15     284807 non-null  float64
     16  V16     284807 non-null  float64
     17  V17     284807 non-null  float64
     18  V18     284807 non-null  float64
     19  V19     284807 non-null  float64
     20  V20     284807 non-null  float64
     21  V21     284807 non-null  float64
     22  V22     284807 non-null  float64
     23  V23     284807 non-null  float64
     24  V24     284807 non-null  float64
     25  V25     284807 non-null  float64
     26  V26     284807 non-null  float64
     27  V27     284807 non-null  float64
     28  V28     284807 non-null  float64
     29  Amount  284807 non-null  float64
     30  Class   284807 non-null  int64  
    dtypes: float64(30), int64(1)
    memory usage: 67.4 MB
       Time        V1        V2        V3        V4        V5        V6        V7  \
    0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
    1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
    2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
    3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
    4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   
    
             V8        V9  ...       V21       V22       V23       V24       V25  \
    0  0.098698  0.363787  ... -0.018307  0.277838 -0.110474  0.066928  0.128539   
    1  0.085102 -0.255425  ... -0.225775 -0.638672  0.101288 -0.339846  0.167170   
    2  0.247676 -1.514654  ...  0.247998  0.771679  0.909412 -0.689281 -0.327642   
    3  0.377436 -1.387024  ... -0.108300  0.005274 -0.190321 -1.175575  0.647376   
    4 -0.270533  0.817739  ... -0.009431  0.798278 -0.137458  0.141267 -0.206010   
    
            V26       V27       V28  Amount  Class  
    0 -0.189115  0.133558 -0.021053  149.62      0  
    1  0.125895 -0.008983  0.014724    2.69      0  
    2 -0.139097 -0.055353 -0.059752  378.66      0  
    3 -0.221929  0.062723  0.061458  123.50      0  
    4  0.502292  0.219422  0.215153   69.99      0  
    
    [5 rows x 31 columns] None                 Time            V1            V2            V3            V4  \
    count  284807.000000  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean    94813.859575  1.175161e-15  3.384974e-16 -1.379537e-15  2.094852e-15   
    std     47488.145955  1.958696e+00  1.651309e+00  1.516255e+00  1.415869e+00   
    min         0.000000 -5.640751e+01 -7.271573e+01 -4.832559e+01 -5.683171e+00   
    25%     54201.500000 -9.203734e-01 -5.985499e-01 -8.903648e-01 -8.486401e-01   
    50%     84692.000000  1.810880e-02  6.548556e-02  1.798463e-01 -1.984653e-02   
    75%    139320.500000  1.315642e+00  8.037239e-01  1.027196e+00  7.433413e-01   
    max    172792.000000  2.454930e+00  2.205773e+01  9.382558e+00  1.687534e+01   
    
                     V5            V6            V7            V8            V9  \
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean   1.021879e-15  1.494498e-15 -5.620335e-16  1.149614e-16 -2.414189e-15   
    std    1.380247e+00  1.332271e+00  1.237094e+00  1.194353e+00  1.098632e+00   
    min   -1.137433e+02 -2.616051e+01 -4.355724e+01 -7.321672e+01 -1.343407e+01   
    25%   -6.915971e-01 -7.682956e-01 -5.540759e-01 -2.086297e-01 -6.430976e-01   
    50%   -5.433583e-02 -2.741871e-01  4.010308e-02  2.235804e-02 -5.142873e-02   
    75%    6.119264e-01  3.985649e-01  5.704361e-01  3.273459e-01  5.971390e-01   
    max    3.480167e+01  7.330163e+01  1.205895e+02  2.000721e+01  1.559499e+01   
    
           ...           V21           V22           V23           V24  \
    count  ...  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean   ...  1.628620e-16 -3.576577e-16  2.618565e-16  4.473914e-15   
    std    ...  7.345240e-01  7.257016e-01  6.244603e-01  6.056471e-01   
    min    ... -3.483038e+01 -1.093314e+01 -4.480774e+01 -2.836627e+00   
    25%    ... -2.283949e-01 -5.423504e-01 -1.618463e-01 -3.545861e-01   
    50%    ... -2.945017e-02  6.781943e-03 -1.119293e-02  4.097606e-02   
    75%    ...  1.863772e-01  5.285536e-01  1.476421e-01  4.395266e-01   
    max    ...  2.720284e+01  1.050309e+01  2.252841e+01  4.584549e+00   
    
                    V25           V26           V27           V28         Amount  \
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  284807.000000   
    mean   5.109395e-16  1.686100e-15 -3.661401e-16 -1.227452e-16      88.349619   
    std    5.212781e-01  4.822270e-01  4.036325e-01  3.300833e-01     250.120109   
    min   -1.029540e+01 -2.604551e+00 -2.256568e+01 -1.543008e+01       0.000000   
    25%   -3.171451e-01 -3.269839e-01 -7.083953e-02 -5.295979e-02       5.600000   
    50%    1.659350e-02 -5.213911e-02  1.342146e-03  1.124383e-02      22.000000   
    75%    3.507156e-01  2.409522e-01  9.104512e-02  7.827995e-02      77.165000   
    max    7.519589e+00  3.517346e+00  3.161220e+01  3.384781e+01   25691.160000   
    
                   Class  
    count  284807.000000  
    mean        0.001727  
    std         0.041527  
    min         0.000000  
    25%         0.000000  
    50%         0.000000  
    75%         0.000000  
    max         1.000000  
    
    [8 rows x 31 columns]
    

### Check for Missing Values

There are no missing values, so we will not have to deal with null values.


```python
print(df.isnull().sum())
```

    Time      0
    V1        0
    V2        0
    V3        0
    V4        0
    V5        0
    V6        0
    V7        0
    V8        0
    V9        0
    V10       0
    V11       0
    V12       0
    V13       0
    V14       0
    V15       0
    V16       0
    V17       0
    V18       0
    V19       0
    V20       0
    V21       0
    V22       0
    V23       0
    V24       0
    V25       0
    V26       0
    V27       0
    V28       0
    Amount    0
    Class     0
    dtype: int64
    

### Conduction of Elementary Exploratory Data Analysis

We will look at:
- Countplot
    - We notice that the dataset is HIGHLY imbalanced, with a very small percentage of the transactions being fraudulent
- Time of Transaction (entire dataset, non-fraudulent, fraudulent)


```python
fig, ax = plt.subplots(figsize=(6, 5))

ax = sns.countplot(x = 'Class', data = df)
plt.tight_layout()
df.hist(figsize=(10, 10))
```




    array([[<Axes: title={'center': 'Time'}>, <Axes: title={'center': 'V1'}>,
            <Axes: title={'center': 'V2'}>, <Axes: title={'center': 'V3'}>,
            <Axes: title={'center': 'V4'}>, <Axes: title={'center': 'V5'}>],
           [<Axes: title={'center': 'V6'}>, <Axes: title={'center': 'V7'}>,
            <Axes: title={'center': 'V8'}>, <Axes: title={'center': 'V9'}>,
            <Axes: title={'center': 'V10'}>, <Axes: title={'center': 'V11'}>],
           [<Axes: title={'center': 'V12'}>, <Axes: title={'center': 'V13'}>,
            <Axes: title={'center': 'V14'}>, <Axes: title={'center': 'V15'}>,
            <Axes: title={'center': 'V16'}>, <Axes: title={'center': 'V17'}>],
           [<Axes: title={'center': 'V18'}>, <Axes: title={'center': 'V19'}>,
            <Axes: title={'center': 'V20'}>, <Axes: title={'center': 'V21'}>,
            <Axes: title={'center': 'V22'}>, <Axes: title={'center': 'V23'}>],
           [<Axes: title={'center': 'V24'}>, <Axes: title={'center': 'V25'}>,
            <Axes: title={'center': 'V26'}>, <Axes: title={'center': 'V27'}>,
            <Axes: title={'center': 'V28'}>,
            <Axes: title={'center': 'Amount'}>],
           [<Axes: title={'center': 'Class'}>, <Axes: >, <Axes: >, <Axes: >,
            <Axes: >, <Axes: >]], dtype=object)




    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_7_1.png)
    



    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_7_2.png)
    



```python
df['Time'].describe()

# Time after first Transaction
df['Time_After_First_Transaction'] = df['Time'] - df['Time'].min()

fig, ax = plt.subplots(figsize=(10, 5))

plt.figure(figsize=(12, 4), dpi=80)
sns.histplot(
    df["Time_After_First_Transaction"],
    bins=50,
    kde=True,                 # add KDE curve
    stat="count",
    linewidth=0,
    color="steelblue",
    ax=ax
)
ax.set_title("Distribution of Time After First Transaction")
ax.set_xlabel("Time After First Transaction (s)")
ax.set_ylabel("Count")

plt.show()
```


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_8_0.png)
    



    <Figure size 960x320 with 0 Axes>


Let's plot the time of transactions against the count. We are hoping to see a different trend with the fraudulent transactions. This is a good visual test.


```python
fig, ax = plt.subplots(figsize=(10, 5))

sns.histplot(
    data=df[df["Class"] == 0],
    x="Time_After_First_Transaction",
    bins=50,
    stat="count",
    kde=True,
    linewidth=0,
    color="green",
    #alpha=0.6,
    label="Normal (0)",
    ax=ax
)
ax.set_title("Distribution of Time After First Transaction (Non-Fraudulent)")
ax.set_xlabel("Time After First Transaction (s)")
ax.set_ylabel("Count")

plt.show()
```


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_10_0.png)
    



```python
fig, ax = plt.subplots(figsize=(10, 5))

# Fraud (Class = 1)
sns.histplot(
    data=df[df["Class"] == 1],
    x="Time_After_First_Transaction",
    bins=50,
    stat="count",
    kde=True,
    linewidth=0,
    color="red",
    #alpha=0.6,
    label="Fraud (1)",
    ax=ax
)
ax.set_title("Distribution of Time After First Transaction (Fraudulent)")
ax.set_xlabel("Time After First Transaction (s)")
ax.set_ylabel("Count")

plt.show()
```


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_11_0.png)
    


We will plot the fraudulent transactions on the same plot at the non-fraudulent transactions. We see that some of the fraudulent transactions are happening during the regular downtime of the non-fraudulent transactions.

\* We took the log of the count for the non-fraudulent transactions so that the values may be comparable.


```python
fig, ax = plt.subplots(figsize=(10, 5))

#1) Non‑fraud (Class = 0) as the background
sns.histplot(
    data=df[df["Class"] == 0],
    x="Time_After_First_Transaction",
    bins=50,
    stat="count",
    color="steelblue",
    alpha=0.6,
    label="Normal (0)",
    ax=ax
)

# 2) Fraud (Class = 1) layered on top
sns.histplot(
    data=df[df["Class"] == 1],
    x="Time_After_First_Transaction",
    bins=50,
    stat="count",
    color="red",
    alpha=0.8,
    label="Fraud (1)",
    ax=ax
)

ax.set_yscale('log')
#ax.set_ylim(0, 200)
ax.set_title("Time After First Transaction: Normal vs Fraud")
ax.set_xlabel("Time After First Transaction (seconds)")
ax.set_ylabel("Count")
ax.legend()

plt.show()
```


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_13_0.png)
    


We have some outliers. Removing these outliers may remove some fraudulent transactions. Due to the low count of fraudulent transactions, we should not remove the already rare fraudulent transactions. We can see from boxplots that the distribution is extremely right skewed. We have a high number of outliers that will negatively impact our model.


```python
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(20, 6))
sns.histplot(df_copy.Amount, ax=ax1)
sns.boxplot(df_copy.Amount, ax=ax2, orient = "h")
```




    <Axes: xlabel='Amount'>




    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_15_1.png)
    


Using the Interquartile Range (IQR), a measure of statistical dispersion, we can use the points that fall outside the range to help us decide what outliers are appropriate to remove.


```python
from scipy.stats import iqr
upper_limit_remove = df_copy.Amount.quantile(0.75) + (1.5 * iqr(df_copy.Amount))
print(upper_limit_remove)
print(df_copy[df_copy.Amount > upper_limit_remove]["Class"].value_counts())
print("-------")
df = df[df.Amount <=8000]
print(df['Class'].value_counts())
print('/nPercentage of fraudulent transactions: {:.2%}'.format((df[df['Class'] ==1].shape[0] / df.shape[0])))
```

    184.5125
    Class
    0    31813
    1       91
    Name: count, dtype: int64
    -------
    Class
    0    284303
    1       492
    Name: count, dtype: int64
    /nPercentage of fraudulent transactions: 0.17%
    


```python
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(20, 15))
sns.boxplot(df[df['Class'] == 1].Amount, ax=ax1, orient="h")
ax1.set_title('Fraudulent Transactions')
sns.boxplot(df[df['Class'] == 0].Amount, ax=ax2, orient="h")
ax2.set_title('Non-Fraudulent Transactions')
```




    Text(0.5, 1.0, 'Non-Fraudulent Transactions')




    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_18_1.png)
    


We can see that the majority of the outliers have been removed while keeping the fraudulent transactions. Now, we will look at a feature correlation heatmap. We can remove or combine redundant features to prevent overfitting, and or speed up training time. It can also help us decide whether to use regularization, trees, etc.

We will use a correlation heat map to visualize the strength of the relationships between the variables.


```python
correlation_heatmap = df_copy.corr()
fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(correlation_heatmap, xticklabels=correlation_heatmap.columns, linewidths=.1, cmap="RdBu", ax=ax)
plt.tight_layout()
plt.show()
```


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_21_0.png)
    


It is time to begin preprocessing the data. We are dealing with a highly imbalanced dataset. There are various ways to handle this class imbalance, but we will start with undersampling the majority class. Our approach will consider three options for the minority:majority class ratio. I am choosing: 1:10, 1:20, and 1:50. This will help with mitigating bias in the learning of the model.


```python
print(df["Class"].value_counts())

minority_cnt = df["Class"].value_counts()[1]

ratio_1_to_10 = minority_cnt*10
ratio_1_to_20 = minority_cnt*20
ratio_1_to_50 = minority_cnt*50

non_fraud_transactions_1_to_10 = df[df["Class"] == 0].sample(ratio_1_to_10)
non_fraud_transactions_1_to_20 = df[df["Class"] == 0].sample(ratio_1_to_20)
non_fraud_transactions_1_to_50 = df[df["Class"] == 0].sample(ratio_1_to_50)
fraud = df[df['Class'] == 1]
print(len(non_fraud_transactions_1_to_10), len(non_fraud_transactions_1_to_20), len(non_fraud_transactions_1_to_50), len(fraud))

df_1_to_10 = non_fraud_transactions_1_to_10._append(fraud).sample(frac=1).reset_index(drop=True)
df_1_to_20 = non_fraud_transactions_1_to_20._append(fraud).sample(frac=1).reset_index(drop=True)
df_1_to_50 = non_fraud_transactions_1_to_50._append(fraud).sample(frac=1).reset_index(drop=True)

x_1_to_10 = df_1_to_10.drop(['Class'], axis=1).values
x_1_to_20 = df_1_to_20.drop(['Class'], axis=1).values
x_1_to_50 = df_1_to_50.drop(['Class'], axis=1).values

y_1_to_10 = df_1_to_10['Class'].values
y_1_to_20 = df_1_to_20['Class'].values
y_1_to_50 = df_1_to_50['Class'].values
```

    Class
    0    284303
    1       492
    Name: count, dtype: int64
    4920 9840 24600 492
    

Now, we need to use our random sampled non-fraudulent datasets and combine it with our fraudulent dataset to create 3 training datasets.


```python
tsne_embedding_1_to_10 = TSNE(n_components=2, random_state=24).fit_transform(x_1_to_10)
tsne_embedding_1_to_20 = TSNE(n_components=2, random_state=24).fit_transform(x_1_to_20)
tsne_embedding_1_to_50 = TSNE(n_components=2, random_state=24).fit_transform(x_1_to_50)
```

We are working with high dimensional data. We want to visualize this data using t-distributed Stochastic Neighbor Embedding (t-SNE). This maps the data points in higher dimensions down to 2-dimensions so that we may see the similarity or differences between the points. Using dimensionality reduction is useful for visualizing clusters and relationships within the data that we have. Unlike Principal Component Analysis (PCA), another form of dimensionality reduction which was done to the data set before we started, it is not a linear method, and we may discover intricate, non-linear relationships in the data. Typically, PCA is performed first to reduce the noise and dimensionality (also speeding up t-SNE), allowing t-SNE to map that data into a 2D (or 3D) space, focussing on the structure of the important components.


```python
colour_map = {0:'green', 1:'red'}
name_map = {0: "Non-fraud", 1: "Fraud"}

#1:10
plt.figure()
for idx, cl in enumerate(np.unique(y_1_to_10)):
    plt.scatter(x = tsne_embedding_1_to_10[y_1_to_10 == cl,0],
                y = tsne_embedding_1_to_10[y_1_to_10 == cl,1],
                c = colour_map[idx],
                label = name_map[cl])
plt.xlabel('X in t_SNE')
plt.ylabel('Y in t_SNE')
plt.legend(loc = 'upper right')
plt.title('t-SNE plot (1:10)')
plt.show()

#1:20
plt.figure()
for idx, cl in enumerate(np.unique(y_1_to_20)):
    plt.scatter(x = tsne_embedding_1_to_20[y_1_to_20 == cl,0],
                y = tsne_embedding_1_to_20[y_1_to_20 == cl,1],
                c = colour_map[idx],
                label = name_map[cl])
plt.xlabel('X in t_SNE')
plt.ylabel('Y in t_SNE')
plt.legend(loc = 'upper right')
plt.title('t-SNE plot (1:20)')
plt.show()

#1:50
plt.figure()
for idx, cl in enumerate(np.unique(y_1_to_50)):
    plt.scatter(x = tsne_embedding_1_to_50[y_1_to_50 == cl,0],
                y = tsne_embedding_1_to_50[y_1_to_50 == cl,1],
                c = colour_map[idx],
                label = name_map[cl])
plt.xlabel('X in t_SNE')
plt.ylabel('Y in t_SNE')
plt.legend(loc = 'upper right')
plt.title('t-SNE plot (1:50)')
plt.show()
```


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_27_0.png)
    



    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_27_1.png)
    



    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_27_2.png)
    


Since there is no separation in the clusters/patterns that we have between the fraudulent and non-fraudulent transactions, we are going to use the hidden representation. This will make the data "human-readable".

We will use and autoencoder as there is a very high volume of "normal" (non-fraudulent) transactions. It will do a good job of "squishing" the data down into lower dimensions. It will eliminate noise, anomaly detection, etc.

The autoencoder has two uses for us. One, it will help us generate a human-readable t-SNE plot, and it will help with our models by balancing complex and imbalanced features.

We will now scale the data so that the model may treat the features more fairly, and will behave in a more stable predictable manner. Autoencoders are sensitive to the input scale, so unscaled data may lead to prolonged training times, and poor local minima, among other things.

### Constructing the Autoencoder.

We need to be careful as to not use the same model for all three datasets. There are two ways to resolve this issue. We can either clone the model and retrain the clones, or we can build three different models. We will choose the latter.


```python
x_scale_1_to_10 = preprocessing.MinMaxScaler().fit_transform(x_1_to_10)
x_scale_1_to_20 = preprocessing.MinMaxScaler().fit_transform(x_1_to_20)
x_scale_1_to_50 = preprocessing.MinMaxScaler().fit_transform(x_1_to_50)

x_norm_1_to_10, x_fraud_1_to_10 = x_scale_1_to_10[y_1_to_10 == 0], x_scale_1_to_10[y_1_to_10 == 1]

x_norm_1_to_20, x_fraud_1_to_20 = x_scale_1_to_20[y_1_to_20 == 0], x_scale_1_to_20[y_1_to_20 == 1]

x_norm_1_to_50, x_fraud_1_to_50 = x_scale_1_to_50[y_1_to_50 == 0], x_scale_1_to_50[y_1_to_50 == 1]
```

### Training the Autoencoder


```python
def build_autoencoder(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    model.add(Dense(100, activation='tanh'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(100, activation='tanh'))

    model.add(Dense(input_dim, activation='tanh'))
    model.compile(optimizer='adadelta', loss='mean_squared_error')
    return model

# 1) autoencoder for 1–10
autoEncoder_1_to_10 = build_autoencoder(x_norm_1_to_10.shape[1])

hist_1_to_10 = autoEncoder_1_to_10.fit(
    x_norm_1_to_10, x_norm_1_to_10,
    batch_size=10, epochs=50, shuffle=True, validation_split=0.2
)

# 2) autoencoder for 1–20
autoEncoder_1_to_20 = build_autoencoder(x_norm_1_to_20.shape[1])

hist_1_to_20 = autoEncoder_1_to_20.fit(
    x_norm_1_to_20, x_norm_1_to_20,
    batch_size=10, epochs=50, shuffle=True, validation_split=0.2
)

# 3) autoencoder for 1–50
autoEncoder_1_to_50 = build_autoencoder(x_norm_1_to_50.shape[1])

hist_1_to_50 = autoEncoder_1_to_50.fit(
    x_norm_1_to_50, x_norm_1_to_50,
    batch_size=10, epochs=50, shuffle=True, validation_split=0.2
)
```

    Epoch 1/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 2ms/step - loss: 0.4153 - val_loss: 0.3877
    Epoch 2/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.3575 - val_loss: 0.3272
    Epoch 3/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.2951 - val_loss: 0.2627
    Epoch 4/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.2301 - val_loss: 0.1979
    Epoch 5/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.1673 - val_loss: 0.1379
    Epoch 6/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.1129 - val_loss: 0.0900
    Epoch 7/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0724 - val_loss: 0.0566
    Epoch 8/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0454 - val_loss: 0.0355
    Epoch 9/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0291 - val_loss: 0.0233
    Epoch 10/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0198 - val_loss: 0.0165
    Epoch 11/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0147 - val_loss: 0.0128
    Epoch 12/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0120 - val_loss: 0.0108
    Epoch 13/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0104 - val_loss: 0.0097
    Epoch 14/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0095 - val_loss: 0.0091
    Epoch 15/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0090 - val_loss: 0.0086
    Epoch 16/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0086 - val_loss: 0.0084
    Epoch 17/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0084 - val_loss: 0.0082
    Epoch 18/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0082 - val_loss: 0.0081
    Epoch 19/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0081 - val_loss: 0.0080
    Epoch 20/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0080 - val_loss: 0.0079
    Epoch 21/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0080 - val_loss: 0.0078
    Epoch 22/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0079 - val_loss: 0.0078
    Epoch 23/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0079 - val_loss: 0.0078
    Epoch 24/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0078 - val_loss: 0.0077
    Epoch 25/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0078 - val_loss: 0.0077
    Epoch 26/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0077 - val_loss: 0.0077
    Epoch 27/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0077 - val_loss: 0.0076
    Epoch 28/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0077 - val_loss: 0.0076
    Epoch 29/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0077 - val_loss: 0.0076
    Epoch 30/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0076 - val_loss: 0.0076
    Epoch 31/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0076 - val_loss: 0.0075
    Epoch 32/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0076 - val_loss: 0.0075
    Epoch 33/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0076 - val_loss: 0.0075
    Epoch 34/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0076 - val_loss: 0.0075
    Epoch 35/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0075 - val_loss: 0.0075
    Epoch 36/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0075 - val_loss: 0.0074
    Epoch 37/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0075 - val_loss: 0.0074
    Epoch 38/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0075 - val_loss: 0.0074
    Epoch 39/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0075 - val_loss: 0.0074
    Epoch 40/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0074 - val_loss: 0.0074
    Epoch 41/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0074 - val_loss: 0.0073
    Epoch 42/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0074 - val_loss: 0.0073
    Epoch 43/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0074 - val_loss: 0.0073
    Epoch 44/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0074 - val_loss: 0.0073
    Epoch 45/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0074 - val_loss: 0.0073
    Epoch 46/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0073 - val_loss: 0.0073
    Epoch 47/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0073 - val_loss: 0.0073
    Epoch 48/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0073 - val_loss: 0.0072
    Epoch 49/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0073 - val_loss: 0.0072
    Epoch 50/50
    [1m394/394[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0073 - val_loss: 0.0072
    Epoch 1/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 2ms/step - loss: 0.4800 - val_loss: 0.4033
    Epoch 2/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.3285 - val_loss: 0.2559
    Epoch 3/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.1937 - val_loss: 0.1362
    Epoch 4/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0953 - val_loss: 0.0611
    Epoch 5/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0416 - val_loss: 0.0270
    Epoch 6/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0203 - val_loss: 0.0152
    Epoch 7/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0131 - val_loss: 0.0112
    Epoch 8/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0105 - val_loss: 0.0096
    Epoch 9/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0094 - val_loss: 0.0088
    Epoch 10/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0088 - val_loss: 0.0084
    Epoch 11/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0085 - val_loss: 0.0081
    Epoch 12/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0083 - val_loss: 0.0080
    Epoch 13/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0081 - val_loss: 0.0078
    Epoch 14/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0081 - val_loss: 0.0078
    Epoch 15/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0080 - val_loss: 0.0077
    Epoch 16/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0079 - val_loss: 0.0076
    Epoch 17/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0079 - val_loss: 0.0076
    Epoch 18/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0078 - val_loss: 0.0075
    Epoch 19/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0078 - val_loss: 0.0075
    Epoch 20/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0078 - val_loss: 0.0075
    Epoch 21/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0077 - val_loss: 0.0074
    Epoch 22/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0077 - val_loss: 0.0074
    Epoch 23/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0077 - val_loss: 0.0074
    Epoch 24/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0076 - val_loss: 0.0074
    Epoch 25/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0076 - val_loss: 0.0073
    Epoch 26/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0076 - val_loss: 0.0073
    Epoch 27/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0075 - val_loss: 0.0073
    Epoch 28/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0075 - val_loss: 0.0072
    Epoch 29/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0075 - val_loss: 0.0072
    Epoch 30/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0075 - val_loss: 0.0072
    Epoch 31/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0075 - val_loss: 0.0072
    Epoch 32/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0074 - val_loss: 0.0072
    Epoch 33/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0074 - val_loss: 0.0071
    Epoch 34/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0074 - val_loss: 0.0071
    Epoch 35/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - loss: 0.0074 - val_loss: 0.0071
    Epoch 36/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0073 - val_loss: 0.0071
    Epoch 37/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0073 - val_loss: 0.0070
    Epoch 38/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0073 - val_loss: 0.0070
    Epoch 39/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0073 - val_loss: 0.0070
    Epoch 40/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0072 - val_loss: 0.0070
    Epoch 41/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0072 - val_loss: 0.0069
    Epoch 42/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0072 - val_loss: 0.0069
    Epoch 43/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0072 - val_loss: 0.0069
    Epoch 44/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0071 - val_loss: 0.0069
    Epoch 45/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0071 - val_loss: 0.0068
    Epoch 46/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0071 - val_loss: 0.0068
    Epoch 47/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0071 - val_loss: 0.0068
    Epoch 48/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0070 - val_loss: 0.0068
    Epoch 49/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0070 - val_loss: 0.0067
    Epoch 50/50
    [1m788/788[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - loss: 0.0070 - val_loss: 0.0067
    Epoch 1/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.2948 - val_loss: 0.1587
    Epoch 2/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0733 - val_loss: 0.0229
    Epoch 3/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0132 - val_loss: 0.0089
    Epoch 4/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0080 - val_loss: 0.0075
    Epoch 5/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0072 - val_loss: 0.0070
    Epoch 6/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 2ms/step - loss: 0.0069 - val_loss: 0.0068
    Epoch 7/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0067 - val_loss: 0.0066
    Epoch 8/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0066 - val_loss: 0.0065
    Epoch 9/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0064 - val_loss: 0.0063
    Epoch 10/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0063 - val_loss: 0.0062
    Epoch 11/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0062 - val_loss: 0.0061
    Epoch 12/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0061 - val_loss: 0.0060
    Epoch 13/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0059 - val_loss: 0.0059
    Epoch 14/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0058 - val_loss: 0.0058
    Epoch 15/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0057 - val_loss: 0.0057
    Epoch 16/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0056 - val_loss: 0.0056
    Epoch 17/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0055 - val_loss: 0.0055
    Epoch 18/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0054 - val_loss: 0.0054
    Epoch 19/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0053 - val_loss: 0.0053
    Epoch 20/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0052 - val_loss: 0.0052
    Epoch 21/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0051 - val_loss: 0.0051
    Epoch 22/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0050 - val_loss: 0.0050
    Epoch 23/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0049 - val_loss: 0.0049
    Epoch 24/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0048 - val_loss: 0.0048
    Epoch 25/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0047 - val_loss: 0.0047
    Epoch 26/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0047 - val_loss: 0.0046
    Epoch 27/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0046 - val_loss: 0.0045
    Epoch 28/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0045 - val_loss: 0.0044
    Epoch 29/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0044 - val_loss: 0.0043
    Epoch 30/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0043 - val_loss: 0.0043
    Epoch 31/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0042 - val_loss: 0.0042
    Epoch 32/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0041 - val_loss: 0.0041
    Epoch 33/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0040 - val_loss: 0.0040
    Epoch 34/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0040 - val_loss: 0.0039
    Epoch 35/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0039 - val_loss: 0.0039
    Epoch 36/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0038 - val_loss: 0.0038
    Epoch 37/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0037 - val_loss: 0.0037
    Epoch 38/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0037 - val_loss: 0.0037
    Epoch 39/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0036 - val_loss: 0.0036
    Epoch 40/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0036 - val_loss: 0.0035
    Epoch 41/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0035 - val_loss: 0.0035
    Epoch 42/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0035 - val_loss: 0.0034
    Epoch 43/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0034 - val_loss: 0.0034
    Epoch 44/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0034 - val_loss: 0.0033
    Epoch 45/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0033 - val_loss: 0.0033
    Epoch 46/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0033 - val_loss: 0.0033
    Epoch 47/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0032 - val_loss: 0.0032
    Epoch 48/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0032 - val_loss: 0.0032
    Epoch 49/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0032 - val_loss: 0.0032
    Epoch 50/50
    [1m1968/1968[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 1ms/step - loss: 0.0031 - val_loss: 0.0031
    

Approximately 30-35 epochs until the validation loss stops decreasing meaningfully and overfitting starts.


```python
encoder_1_to_10 = Sequential(autoEncoder_1_to_10.layers[:2])
encoder_1_to_20 = Sequential(autoEncoder_1_to_20.layers[:2])
encoder_1_to_50 = Sequential(autoEncoder_1_to_50.layers[:2])
```



After the autoencoder that we built has learnt the representation. The latent representation is the mapping of the transactions in some abstract space. Now, we need to uncover the vectors of that latent space that the autoencoder has learnt.


```python
#hidden representations for non_fraud(norm) and fraud
norm_hidden_representation_1_to_10 = encoder_1_to_10.predict(x_norm_1_to_10)
fraud_hidden_representation_1_to_10 = encoder_1_to_10.predict(x_fraud_1_to_10)

norm_hidden_representation_1_to_20 = encoder_1_to_20.predict(x_norm_1_to_20)
fraud_hidden_representation_1_to_20 = encoder_1_to_20.predict(x_fraud_1_to_20)

norm_hidden_representation_1_to_50 = encoder_1_to_50.predict(x_norm_1_to_50)
fraud_hidden_representation_1_to_50 = encoder_1_to_50.predict(x_fraud_1_to_50)

# 1–10
representation_x_1_to_10 = np.append(
    norm_hidden_representation_1_to_10,
    fraud_hidden_representation_1_to_10,
    axis=0
)
y_norm_1_to_10   = np.zeros(norm_hidden_representation_1_to_10.shape[0])
y_fraud_1_to_10  = np.ones(fraud_hidden_representation_1_to_10.shape[0])
representation_y_1_to_10 = np.append(y_norm_1_to_10, y_fraud_1_to_10)

# 1–20
representation_x_1_to_20 = np.append(
    norm_hidden_representation_1_to_20,
    fraud_hidden_representation_1_to_20,
    axis=0
)
y_norm_1_to_20   = np.zeros(norm_hidden_representation_1_to_20.shape[0])
y_fraud_1_to_20  = np.ones(fraud_hidden_representation_1_to_20.shape[0])
representation_y_1_to_20 = np.append(y_norm_1_to_20, y_fraud_1_to_20)

# 1–50
representation_x_1_to_50 = np.append(
    norm_hidden_representation_1_to_50,
    fraud_hidden_representation_1_to_50,
    axis=0
)
y_norm_1_to_50   = np.zeros(norm_hidden_representation_1_to_50.shape[0])
y_fraud_1_to_50  = np.ones(fraud_hidden_representation_1_to_50.shape[0])
representation_y_1_to_50 = np.append(y_norm_1_to_50, y_fraud_1_to_50)
```

    [1m154/154[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 784us/step
    [1m16/16[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step 
    [1m308/308[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 665us/step
    [1m16/16[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    [1m769/769[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 583us/step
    [1m16/16[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step 
    

Let's graph the uncovered vectors with a new t-SNE graph. This is a more human-readable t-SNE mapping. This can show the separation between the two classe.


```python
p10 = TSNE(n_components=2, random_state = 24).fit_transform(representation_x_1_to_10)
p20 = TSNE(n_components=2, random_state = 24).fit_transform(representation_x_1_to_20)
p50 = TSNE(n_components=2, random_state = 24).fit_transform(representation_x_1_to_50)

colour_map = {0: 'green', 1: 'red'}
name_map = {0: "Non-fraud", 1: "Fraud"}

plt.figure()
for i, cl in enumerate(np.unique(representation_y_1_to_10)):
    plt.scatter(x = p10[representation_y_1_to_10 == cl, 0],
                y = p10[representation_y_1_to_10 == cl, 1],
                c = colour_map[i],
                label = name_map[cl])
plt.xlabel('X in t-SNE space')
plt.ylabel('Y in t-SNE space')
plt.legend(loc = 'upper right')
plt.title('t-SNE visualization of test data (1:10)')
plt.show()

plt.figure()
for i, cl in enumerate(np.unique(representation_y_1_to_20)):
    plt.scatter(x = p20[representation_y_1_to_20 == cl, 0],
                y = p20[representation_y_1_to_20 == cl, 1],
                c = colour_map[i],
                label = name_map[cl])
plt.xlabel('X in t-SNE space')
plt.ylabel('Y in t-SNE space')
plt.legend(loc = 'upper right')
plt.title('t-SNE visualization of test data (1:20)')
plt.show()

plt.figure()
for i, cl in enumerate(np.unique(representation_y_1_to_50)):
    plt.scatter(x = p50[representation_y_1_to_50 == cl, 0],
                y = p50[representation_y_1_to_50 == cl, 1],
                c = colour_map[i],
                label = name_map[cl])
plt.xlabel('X in t-SNE space')
plt.ylabel('Y in t-SNE space')
plt.legend(loc = 'upper right')
plt.title('t-SNE visualization of test data (1:50)')
plt.show()

```


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_39_0.png)
    



    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_39_1.png)
    



    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_39_2.png)
    


To be a thorough as possible, we added many pre-processing steps to make the learning easier and to increase the model's performance. More specifically, our pre-processing steps helped in dealing with: Class Imbalances, Reducing Noise (w/ Representation), Better Separation.

The algorithms that we are going to use are:
- Logistic Regression
    - A great baseline performance. It is a simple model
- Decision Tree
     - Good baseline model
- XGBoost
    - Effective at handling complex data and great at minimizing both false positives and negatives. Precisely what is needed for credit card fraud
- Random Forest
     - Ensemble method with high robustness when dealing with imbalanced datasets.


```python
def eval_scale(X, y, title_suffix):
    # train/val split
    train_x, val_x, train_y, val_y = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=24
    )
    # fit logistic regression
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(train_x, train_y)
    pred_y = clf.predict(val_x)

    print(f"=== {title_suffix} ===")
    print(classification_report(val_y, pred_y))

    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(val_y, pred_y, normalize='true'),
        annot=True, ax=ax
    )
    ax.set_title(f'Confusion Matrix ({title_suffix})')
    ax.set_ylabel('Real Value')
    ax.set_xlabel('Predicted Value')
    plt.show()

# 1–10
eval_scale(representation_x_1_to_10, representation_y_1_to_10, "1:10")

# 1–20
eval_scale(representation_x_1_to_20, representation_y_1_to_20, "1:20")

# 1–50
eval_scale(representation_x_1_to_50, representation_y_1_to_50, "1:50")
```

    === 1:10 ===
                  precision    recall  f1-score   support
    
             0.0       0.97      1.00      0.99       985
             1.0       1.00      0.71      0.83        98
    
        accuracy                           0.97      1083
       macro avg       0.99      0.86      0.91      1083
    weighted avg       0.97      0.97      0.97      1083
    
    


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_42_1.png)
    


    === 1:20 ===
                  precision    recall  f1-score   support
    
             0.0       0.98      1.00      0.99      1969
             1.0       0.99      0.67      0.80        98
    
        accuracy                           0.98      2067
       macro avg       0.98      0.84      0.90      2067
    weighted avg       0.98      0.98      0.98      2067
    
    


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_42_3.png)
    


    === 1:50 ===
                  precision    recall  f1-score   support
    
             0.0       0.99      1.00      1.00      4921
             1.0       0.95      0.62      0.75        98
    
        accuracy                           0.99      5019
       macro avg       0.97      0.81      0.87      5019
    weighted avg       0.99      0.99      0.99      5019
    
    


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_42_5.png)
    



```python
def eval_tree_scale(X, y, title_suffix):
    train_x, val_x, train_y, val_y = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=24
    )

    model = DecisionTreeClassifier(max_depth=6, criterion='entropy', random_state=24)
    model.fit(train_x, train_y)
    y_pred = model.predict(val_x)

    print(f"=== Decision Tree ({title_suffix}) ===")
    print(classification_report(val_y, y_pred))

    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(val_y, y_pred, normalize='true'),
        annot=True,
        ax=ax
    )
    ax.set_title(f'Confusion Matrix ({title_suffix})')
    ax.set_ylabel('True Value')
    ax.set_xlabel('Predicted Value')
    plt.show()

# 1–10
eval_tree_scale(representation_x_1_to_10, representation_y_1_to_10, "1:10")

# 1–20
eval_tree_scale(representation_x_1_to_20, representation_y_1_to_20, "1:20")

# 1–50
eval_tree_scale(representation_x_1_to_50, representation_y_1_to_50, "1:50")
```

    === Decision Tree (1:10) ===
                  precision    recall  f1-score   support
    
             0.0       0.97      0.99      0.98       985
             1.0       0.91      0.71      0.80        98
    
        accuracy                           0.97      1083
       macro avg       0.94      0.85      0.89      1083
    weighted avg       0.97      0.97      0.97      1083
    
    


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_43_1.png)
    


    === Decision Tree (1:20) ===
                  precision    recall  f1-score   support
    
             0.0       0.99      1.00      0.99      1969
             1.0       0.90      0.81      0.85        98
    
        accuracy                           0.99      2067
       macro avg       0.94      0.90      0.92      2067
    weighted avg       0.99      0.99      0.99      2067
    
    


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_43_3.png)
    


    === Decision Tree (1:50) ===
                  precision    recall  f1-score   support
    
             0.0       0.99      1.00      1.00      4921
             1.0       0.91      0.69      0.79        98
    
        accuracy                           0.99      5019
       macro avg       0.95      0.85      0.89      5019
    weighted avg       0.99      0.99      0.99      5019
    
    


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_43_5.png)
    



```python
def eval_xgb_scale(X, y, title_suffix):
    train_x, val_x, train_y, val_y = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=24
    )

    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=24,
        n_jobs=-1
    )
    model.fit(train_x, train_y)

    y_pred = model.predict(val_x)

    print(f"=== XGBoost ({title_suffix}) ===")
    print(classification_report(val_y, y_pred))

    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(val_y, y_pred, normalize='true'),
        annot=True,
        ax=ax
    )
    ax.set_title(f'Confusion Matrix (XGBoost, {title_suffix})')
    ax.set_ylabel('True Value')
    ax.set_xlabel('Predicted Value')
    plt.show()

# 1–10
eval_xgb_scale(representation_x_1_to_10, representation_y_1_to_10, "1:10")

# 1–20
eval_xgb_scale(representation_x_1_to_20, representation_y_1_to_20, "1:20")

# 1–50
eval_xgb_scale(representation_x_1_to_50, representation_y_1_to_50, "1:50")

```

    === XGBoost (1:10) ===
                  precision    recall  f1-score   support
    
             0.0       0.98      1.00      0.99       985
             1.0       0.96      0.80      0.87        98
    
        accuracy                           0.98      1083
       macro avg       0.97      0.90      0.93      1083
    weighted avg       0.98      0.98      0.98      1083
    
    


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_44_1.png)
    


    === XGBoost (1:20) ===
                  precision    recall  f1-score   support
    
             0.0       0.99      1.00      1.00      1969
             1.0       0.98      0.85      0.91        98
    
        accuracy                           0.99      2067
       macro avg       0.98      0.92      0.95      2067
    weighted avg       0.99      0.99      0.99      2067
    
    


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_44_3.png)
    


    === XGBoost (1:50) ===
                  precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00      4921
             1.0       0.96      0.79      0.87        98
    
        accuracy                           1.00      5019
       macro avg       0.98      0.89      0.93      5019
    weighted avg       1.00      1.00      0.99      5019
    
    


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_44_5.png)
    



```python
def eval_rf_scale(X, y, title_suffix):
    train_x, val_x, train_y, val_y = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=24
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,        # or a number if you want to regularize
        class_weight='balanced',  # optional, helps with imbalance
        random_state=24,
        n_jobs=-1
    )
    rf.fit(train_x, train_y)
    y_pred = rf.predict(val_x)

    print(f"=== Random Forest ({title_suffix}) ===")
    print(classification_report(val_y, y_pred))

    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(val_y, y_pred, normalize='true'),
        annot=True,
        ax=ax
    )
    ax.set_title(f'Confusion Matrix (RF, {title_suffix})')
    ax.set_ylabel('True Value')
    ax.set_xlabel('Predicted Value')
    plt.show()

# 1–10
eval_rf_scale(representation_x_1_to_10, representation_y_1_to_10, "1–10")

# 1–20
eval_rf_scale(representation_x_1_to_20, representation_y_1_to_20, "1–20")

# 1–50
eval_rf_scale(representation_x_1_to_50, representation_y_1_to_50, "1–50")
```

    === Random Forest (1–10) ===
                  precision    recall  f1-score   support
    
             0.0       0.98      1.00      0.99       985
             1.0       0.99      0.80      0.88        98
    
        accuracy                           0.98      1083
       macro avg       0.98      0.90      0.94      1083
    weighted avg       0.98      0.98      0.98      1083
    
    


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_45_1.png)
    


    === Random Forest (1–20) ===
                  precision    recall  f1-score   support
    
             0.0       0.99      1.00      0.99      1969
             1.0       0.99      0.81      0.89        98
    
        accuracy                           0.99      2067
       macro avg       0.99      0.90      0.94      2067
    weighted avg       0.99      0.99      0.99      2067
    
    


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_45_3.png)
    


    === Random Forest (1–50) ===
                  precision    recall  f1-score   support
    
             0.0       0.99      1.00      1.00      4921
             1.0       0.99      0.68      0.81        98
    
        accuracy                           0.99      5019
       macro avg       0.99      0.84      0.90      5019
    weighted avg       0.99      0.99      0.99      5019
    
    


    
![png](credit_card_fraud_detection_files/credit_card_fraud_detection_45_5.png)
    


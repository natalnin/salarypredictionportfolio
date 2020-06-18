# Salary Predictions Based on Job Descriptions

# Part 1 - DEFINE

### What are the salaries of new job postings? 

The purpose of this project is to make accurate salary predictions based on known salaries. This model will serve as a guide for offering competitive compensations based on years of experience, joy type, college major and industry. 


```python
#import your libraries
import pandas as pd
import sklearn as sk
import pandas as dp
import numpy as np      
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
%matplotlib inline

from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


#my info
__author__ = "Natalia"
__email__ = "natalianino06@gmailcom"
```

## Part 2 - DISCOVER

### Load raw data & verify data 


```python
train_features = pd.read_csv("/Users/natalianino/Downloads/data/train_features.csv")
train_salaries = pd.read_csv("/Users/natalianino/Downloads/data/train_salaries.csv")
test_features = pd.read_csv("/Users/natalianino/Downloads/data/test_features.csv")
```


```python
train_features.head(5)
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
      <th>jobId</th>
      <th>companyId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JOB1362684407687</td>
      <td>COMP37</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>MATH</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JOB1362684407688</td>
      <td>COMP19</td>
      <td>CEO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>WEB</td>
      <td>3</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JOB1362684407689</td>
      <td>COMP52</td>
      <td>VICE_PRESIDENT</td>
      <td>DOCTORAL</td>
      <td>PHYSICS</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JOB1362684407690</td>
      <td>COMP38</td>
      <td>MANAGER</td>
      <td>DOCTORAL</td>
      <td>CHEMISTRY</td>
      <td>AUTO</td>
      <td>8</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JOB1362684407691</td>
      <td>COMP7</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>PHYSICS</td>
      <td>FINANCE</td>
      <td>8</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_salaries.head(5)
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
      <th>jobId</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JOB1362684407687</td>
      <td>130</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JOB1362684407688</td>
      <td>101</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JOB1362684407689</td>
      <td>137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JOB1362684407690</td>
      <td>142</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JOB1362684407691</td>
      <td>163</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_features.head(5)
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
      <th>jobId</th>
      <th>companyId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JOB1362684407687</td>
      <td>COMP37</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>MATH</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JOB1362684407688</td>
      <td>COMP19</td>
      <td>CEO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>WEB</td>
      <td>3</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JOB1362684407689</td>
      <td>COMP52</td>
      <td>VICE_PRESIDENT</td>
      <td>DOCTORAL</td>
      <td>PHYSICS</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JOB1362684407690</td>
      <td>COMP38</td>
      <td>MANAGER</td>
      <td>DOCTORAL</td>
      <td>CHEMISTRY</td>
      <td>AUTO</td>
      <td>8</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JOB1362684407691</td>
      <td>COMP7</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>PHYSICS</td>
      <td>FINANCE</td>
      <td>8</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>



In order to have the 'picture' the training features & training salaries will be joined. The testing features will be used to test the model once completed. 


```python
train_data = pd.merge(train_features, train_salaries, on = "jobId" )
```


```python
train_data.head(5)
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
      <th>jobId</th>
      <th>companyId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JOB1362684407687</td>
      <td>COMP37</td>
      <td>CFO</td>
      <td>MASTERS</td>
      <td>MATH</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>83</td>
      <td>130</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JOB1362684407688</td>
      <td>COMP19</td>
      <td>CEO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>WEB</td>
      <td>3</td>
      <td>73</td>
      <td>101</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JOB1362684407689</td>
      <td>COMP52</td>
      <td>VICE_PRESIDENT</td>
      <td>DOCTORAL</td>
      <td>PHYSICS</td>
      <td>HEALTH</td>
      <td>10</td>
      <td>38</td>
      <td>137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JOB1362684407690</td>
      <td>COMP38</td>
      <td>MANAGER</td>
      <td>DOCTORAL</td>
      <td>CHEMISTRY</td>
      <td>AUTO</td>
      <td>8</td>
      <td>17</td>
      <td>142</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JOB1362684407691</td>
      <td>COMP7</td>
      <td>VICE_PRESIDENT</td>
      <td>BACHELORS</td>
      <td>PHYSICS</td>
      <td>FINANCE</td>
      <td>8</td>
      <td>16</td>
      <td>163</td>
    </tr>
  </tbody>
</table>
</div>



### ---- 3 Clean the data ----

In order to clean the data I will look to see if there missing data, any values in the salary variable that are not meaningful (0 or negative salaries) and check the type for each variable. 


```python
##missing data 
missing_data = train_data.isnull().sum()
print(missing_data)
```

    jobId                  0
    companyId              0
    jobType                0
    degree                 0
    major                  0
    industry               0
    yearsExperience        0
    milesFromMetropolis    0
    salary                 0
    dtype: int64



```python
##checking for negative or 0 value in salaries 
zero_salary = train_data["salary"]<=0
invalid_salary = train_data[zero_salary]
invalid_salary
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
      <th>jobId</th>
      <th>companyId</th>
      <th>jobType</th>
      <th>degree</th>
      <th>major</th>
      <th>industry</th>
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30559</th>
      <td>JOB1362684438246</td>
      <td>COMP44</td>
      <td>JUNIOR</td>
      <td>DOCTORAL</td>
      <td>MATH</td>
      <td>AUTO</td>
      <td>11</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>495984</th>
      <td>JOB1362684903671</td>
      <td>COMP34</td>
      <td>JUNIOR</td>
      <td>NONE</td>
      <td>NONE</td>
      <td>OIL</td>
      <td>1</td>
      <td>25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>652076</th>
      <td>JOB1362685059763</td>
      <td>COMP25</td>
      <td>CTO</td>
      <td>HIGH_SCHOOL</td>
      <td>NONE</td>
      <td>AUTO</td>
      <td>6</td>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <th>816129</th>
      <td>JOB1362685223816</td>
      <td>COMP42</td>
      <td>MANAGER</td>
      <td>DOCTORAL</td>
      <td>ENGINEERING</td>
      <td>FINANCE</td>
      <td>18</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>828156</th>
      <td>JOB1362685235843</td>
      <td>COMP40</td>
      <td>VICE_PRESIDENT</td>
      <td>MASTERS</td>
      <td>ENGINEERING</td>
      <td>WEB</td>
      <td>3</td>
      <td>29</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
##drop invalid salaries 
train_data = train_data.drop(invalid_salary.index.tolist())
```


```python
print(train_data.dtypes)
```

    jobId                  object
    companyId              object
    jobType                object
    degree                 object
    major                  object
    industry               object
    yearsExperience         int64
    milesFromMetropolis     int64
    salary                  int64
    dtype: object



```python
##transformation of variable types 
train_data['jobId'] = pd.Categorical(train_data['jobId'])
train_data['companyId'] = pd.Categorical(train_data['companyId'])
train_data['jobType'] = pd.Categorical(train_data['jobType'])
train_data['degree'] = pd.Categorical(train_data['degree'])
train_data['major'] = pd.Categorical(train_data['major'])
train_data['industry'] = pd.Categorical(train_data['industry'])
train_data.dtypes
```




    jobId                  category
    companyId              category
    jobType                category
    degree                 category
    major                  category
    industry               category
    yearsExperience           int64
    milesFromMetropolis       int64
    salary                    int64
    dtype: object



# Part 3: Exploratory Data Analysis 

Descriptive Statistics on quantitative data: 


```python
#summarize int variables including salary (target variable)
train_data.describe()
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
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>999995.000000</td>
      <td>999995.000000</td>
      <td>999995.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>11.992407</td>
      <td>49.529381</td>
      <td>116.062398</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.212390</td>
      <td>28.877721</td>
      <td>38.717163</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.000000</td>
      <td>25.000000</td>
      <td>88.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>12.000000</td>
      <td>50.000000</td>
      <td>114.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>18.000000</td>
      <td>75.000000</td>
      <td>141.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>24.000000</td>
      <td>99.000000</td>
      <td>301.000000</td>
    </tr>
  </tbody>
</table>
</div>



The count has 999,995 which makes sense, since invalid salaries were removed. Mean and standard deviation seem reasonable to continue and min and max are also wihtin in reasonable values. 


```python
#look for correlation between each feature and the target
train_data.corr()
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
      <th>yearsExperience</th>
      <th>milesFromMetropolis</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>yearsExperience</th>
      <td>1.000000</td>
      <td>0.000672</td>
      <td>0.375013</td>
    </tr>
    <tr>
      <th>milesFromMetropolis</th>
      <td>0.000672</td>
      <td>1.000000</td>
      <td>-0.297686</td>
    </tr>
    <tr>
      <th>salary</th>
      <td>0.375013</td>
      <td>-0.297686</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Miles from metrolopis and salary have a negative correlation (-0.297) 
Salary and years of experience have a positive correlation (0.375)

Normal distribution of salary: 


```python
sns.distplot(train_data['salary'], fit=norm, label='Salary')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a262c1510>




![png](output_28_1.png)


Using years of experience as a predictor of salary: 


```python
# Plot regression line on years of experience
width = 6
height = 5
plt.figure(figsize=(width, height))
sns.regplot(x='yearsExperience', y='salary', data=train_data, line_kws={'color':'red'})
plt.ylim(0,)
```




    (0, 315.21335540838845)




![png](output_30_1.png)


The positive correlation is not as strong, which was suspected from the correlation above and the data seems to be widely spread


```python
# Plot regression line on miles from metropolis 
width = 6
height = 5
plt.figure(figsize=(width, height))
sns.regplot(x='milesFromMetropolis', y='salary', data=train_data, line_kws={'color':'red'})
plt.ylim(0,)
```




    (0, 315.21335540838845)




![png](output_32_1.png)


Salary by company ID (Are there differences in salary based on company and job type?)


```python
train_data_pivots = train_data.pivot_table(index='companyId', columns='jobType', values='salary')
train_data_pivots
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
      <th>jobType</th>
      <th>CEO</th>
      <th>CFO</th>
      <th>CTO</th>
      <th>JANITOR</th>
      <th>JUNIOR</th>
      <th>MANAGER</th>
      <th>SENIOR</th>
      <th>VICE_PRESIDENT</th>
    </tr>
    <tr>
      <th>companyId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>COMP0</th>
      <td>144.693320</td>
      <td>135.460674</td>
      <td>135.174960</td>
      <td>70.612705</td>
      <td>96.569827</td>
      <td>115.838305</td>
      <td>105.330697</td>
      <td>124.635732</td>
    </tr>
    <tr>
      <th>COMP1</th>
      <td>144.917731</td>
      <td>134.557271</td>
      <td>135.584980</td>
      <td>70.978899</td>
      <td>95.817444</td>
      <td>115.510802</td>
      <td>104.245377</td>
      <td>125.650939</td>
    </tr>
    <tr>
      <th>COMP10</th>
      <td>146.515657</td>
      <td>134.690890</td>
      <td>135.153025</td>
      <td>70.883848</td>
      <td>95.133198</td>
      <td>114.698522</td>
      <td>106.229459</td>
      <td>125.651568</td>
    </tr>
    <tr>
      <th>COMP11</th>
      <td>144.134702</td>
      <td>135.304603</td>
      <td>136.367988</td>
      <td>71.392911</td>
      <td>95.804192</td>
      <td>114.172959</td>
      <td>105.741629</td>
      <td>125.935533</td>
    </tr>
    <tr>
      <th>COMP12</th>
      <td>146.007000</td>
      <td>134.625456</td>
      <td>135.704691</td>
      <td>70.646912</td>
      <td>94.764895</td>
      <td>116.148390</td>
      <td>104.990447</td>
      <td>125.020070</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>COMP61</th>
      <td>147.183528</td>
      <td>135.274145</td>
      <td>135.769309</td>
      <td>71.840039</td>
      <td>96.243816</td>
      <td>115.450409</td>
      <td>105.400408</td>
      <td>124.993921</td>
    </tr>
    <tr>
      <th>COMP62</th>
      <td>146.728414</td>
      <td>134.937028</td>
      <td>135.486680</td>
      <td>70.740488</td>
      <td>96.560976</td>
      <td>117.280284</td>
      <td>105.392273</td>
      <td>124.758483</td>
    </tr>
    <tr>
      <th>COMP7</th>
      <td>145.142637</td>
      <td>134.875712</td>
      <td>134.577020</td>
      <td>70.025154</td>
      <td>95.476697</td>
      <td>115.290256</td>
      <td>105.232394</td>
      <td>126.241888</td>
    </tr>
    <tr>
      <th>COMP8</th>
      <td>145.349329</td>
      <td>136.195396</td>
      <td>135.003447</td>
      <td>71.060926</td>
      <td>94.951621</td>
      <td>114.773489</td>
      <td>105.011777</td>
      <td>125.141299</td>
    </tr>
    <tr>
      <th>COMP9</th>
      <td>145.632353</td>
      <td>135.139546</td>
      <td>136.658286</td>
      <td>70.755522</td>
      <td>94.803571</td>
      <td>115.933494</td>
      <td>107.143808</td>
      <td>125.808065</td>
    </tr>
  </tbody>
</table>
<p>63 rows × 8 columns</p>
</div>



The pivot table makes it hard to visualize any differences in job type and company. A box blot will be created to further visualize. 


```python
train_data_pivots.describe()
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
      <th>jobType</th>
      <th>CEO</th>
      <th>CFO</th>
      <th>CTO</th>
      <th>JANITOR</th>
      <th>JUNIOR</th>
      <th>MANAGER</th>
      <th>SENIOR</th>
      <th>VICE_PRESIDENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>63.000000</td>
      <td>63.000000</td>
      <td>63.000000</td>
      <td>63.000000</td>
      <td>63.000000</td>
      <td>63.000000</td>
      <td>63.000000</td>
      <td>63.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>145.313407</td>
      <td>135.458834</td>
      <td>135.480841</td>
      <td>70.812473</td>
      <td>95.332708</td>
      <td>115.366690</td>
      <td>105.488889</td>
      <td>125.370796</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.855174</td>
      <td>0.759772</td>
      <td>0.747521</td>
      <td>0.527044</td>
      <td>0.622318</td>
      <td>0.685821</td>
      <td>0.712628</td>
      <td>0.613353</td>
    </tr>
    <tr>
      <th>min</th>
      <td>143.252708</td>
      <td>133.896570</td>
      <td>133.918050</td>
      <td>69.481382</td>
      <td>93.744734</td>
      <td>114.172959</td>
      <td>103.640387</td>
      <td>123.956219</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>144.899535</td>
      <td>134.937870</td>
      <td>134.848773</td>
      <td>70.559691</td>
      <td>94.868538</td>
      <td>114.844998</td>
      <td>105.113722</td>
      <td>124.936427</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>145.362887</td>
      <td>135.345038</td>
      <td>135.555171</td>
      <td>70.740627</td>
      <td>95.242316</td>
      <td>115.382411</td>
      <td>105.500000</td>
      <td>125.341850</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>145.779116</td>
      <td>135.893979</td>
      <td>135.963034</td>
      <td>71.124256</td>
      <td>95.807800</td>
      <td>115.846214</td>
      <td>106.100909</td>
      <td>125.797808</td>
    </tr>
    <tr>
      <th>max</th>
      <td>147.632727</td>
      <td>137.330107</td>
      <td>137.123925</td>
      <td>71.996976</td>
      <td>96.569827</td>
      <td>117.280284</td>
      <td>107.143808</td>
      <td>126.814834</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data_pivots = train_data.pivot_table(index='jobType', columns='companyId', values='salary')
train_data_pivots.plot(kind='box', figsize = [16, 8])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a26cf8b50>




![png](output_37_1.png)



```python
train_data_pivots = train_data.pivot_table(index='companyId', columns='jobType', values='salary')
train_data_pivots.plot(kind='box', figsize = [16, 8])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a30cd5410>




![png](output_38_1.png)


Company ID does not seem to have any significance in salary but job type does. The highest salaries are for CEO, CFO and CTO. The lowest is Janitor. The distribution for salaries by job type seem to be normally distributed. 

How is the salary based on the following degree, major and industry distributed? 


```python
violinplot = sns.violinplot(x="degree", y="salary", data=train_data)
```


```python
violinplot = sns.violinplot(x="industry", y="salary", data=train_data)
```


```python
violinplot = sns.violinplot(x="major", y="salary", data=train_data)
```


```python
## dummy variables for categorical variables in train_data
train_data= pd.get_dummies(train_data)
train_data.head()
```


```python
## plot heatmap of all correlation coefficients 
train_data_corr = train_data.corr()
plt.subplots(figsize=(40,30))
sns.heatmap(train_data_corr, cmap='YlGnBu', linewidth=.005, annot=True)
```

### ---- 5 Establish a baseline ----

A reasonable metric (MSE will be used in this case)



```python
## creating a simple model and measuring its efficacy 
```


```python
X = train_data[train_data.loc[ : , train_data.columns != 'salary'].columns]
y = train_data['salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```


```python
##Linear regression 
lm = LinearRegression()
lm.fit(X_train, y_train)
lm
```


```python
print(lm.intercept_)
print(lm.coef_)
```


```python
## MSE of training data 
print("The MSE of prediction model is:", mean_square_error(y_train, yhat))
```


```python
 Print accuracy score using 5-fold cross validation
scores = cross_val_score(lm, X_train, y_train, cv=5)
print("5-Fold Cross Validation Accuracy (train data):", (np.mean(scores)), (np.std(scores)))

```


```python
##actual salaries vs predicted 
Title = 'Distribution  Plot of Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat, "Actual Values (Train)", "Predicted Values (Train)", Title)
```

### ---- 6 Hypothesize solution ----


```python
#brainstorm 3 models that you think may improve results over the baseline model based
#on your 
```

Brainstorm 3 models that you think may improve results over the baseline model based on your EDA and explain why they're reasonable solutions here.

Also write down any new features that you think you should try adding to the model based on your EDA, e.g. interaction variables, summary statistics for each group, etc

## Part 3 - DEVELOP

You will cycle through creating features, tuning models, and training/validing models (steps 7-9) until you've reached your efficacy goal

#### Your metric will be MSE and your goal is:
 - <360 for entry-level data science roles
 - <320 for senior data science roles

### ---- 7 Engineer features  ----


```python
#make sure that data is ready for modeling
#create any new features needed to potentially enhance model
```

### ---- 8 Create models ----


```python
#create and tune the models that you brainstormed during part 2
```

### ---- 9 Test models ----


```python
#do 5-fold cross validation on models and measure MSE
```

### ---- 10 Select best model  ----


```python
#select the model with the lowest error as your "prodcuction" model
```

## Part 4 - DEPLOY

### ---- 11 Automate pipeline ----


```python
#write script that trains model on entire training set, saves model to disk,
#and scores the "test" dataset
```

### ---- 12 Deploy solution ----


```python
#save your prediction to a csv file or optionally save them as a table in a SQL database
#additionally, you want to save a visualization and summary of your prediction and feature importances
#these visualizations and summaries will be extremely useful to business stakeholders
```

### ---- 13 Measure efficacy ----

We'll skip this step since we don't have the outcomes for the test data


```python

```


```python

```


```python

```


```python

```

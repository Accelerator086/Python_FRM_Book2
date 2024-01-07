# B2_Ch9_1.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

## 9.3 Codes

B2_Ch9_1_A.py 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import plotly.io as pio
import chart_studio.plotly
from plotly.offline import plot
import plotly.graph_objects as go
from  matplotlib.ticker import FuncFormatter

data = pd.read_csv(r'./cs-training.csv')
data = data.iloc[:,1:]

data.shape
data.info()


B2_Ch9_1_B.py 
# make a fancy table for the corresponding Chinese translation of the column names
translation_map = {'SeriousDlqin2yrs':'两年内是否违约',
                   'RevolvingUtilizationOfUnsecuredLines':'可用额度比值',
                   'age':'年龄',
                   'NumberOfTime30-59DaysPastDueNotWorse':'借贷逾期30-59天数目',
                   'DebtRatio':'负债率',
                   'MonthlyIncome':'月收入',
                   'NumberOfOpenCreditLinesAndLoans':'借贷数量',
                   'NumberOfTimes90DaysLate':'借贷逾期90天数目',
                   'NumberRealEstateLoansOrLines':'固定资产贷款量',
                   'NumberOfTime60-89DaysPastDueNotWorse':'借贷逾期60-89天数目',
                   'NumberOfDependents':'家属人数'}

pio.renderers.default = "browser"
df_transmap = pd.DataFrame.from_dict(translation_map, orient='index').reset_index()
df_transmap.columns = ['English', 'Chinese']
pio.renderers.default = "browser"
fig = go.Figure(data=[go.Table(
                                header = dict(values=list(df_transmap.columns),
                                            fill_color = 'paleturquoise',
                                            align='left'),
                                cells = dict(values=[df_transmap.English, df_transmap.Chinese],
                                           fill_color='lavender',
                                           align='left'))
])

fig.show()

# Monthly income Data Missing Ratio
print("missing ratio:{:.2%}".format(data['MonthlyIncome'].isnull().sum()/data.shape[0]))

B2_Ch9_1_C.py 
# fill NA by random forest
data_process = data.iloc[:,[5,0,1,2,3,4,6,7,8,9]]
# split to known and unknown
known = data_process[data_process.MonthlyIncome.notnull()].values
unknown = data_process[data_process.MonthlyIncome.isnull()].values
# training set
X = known[:,1:]
y = known[:,0]
# fitting model
model = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
model.fit(X,y)
# pridict missing data
pred = model.predict(unknown[:,1:]).round(0)
# fill missing data
data.loc[data['MonthlyIncome'].isnull(),'MonthlyIncome'] = pred


B2_Ch9_1_D.py
# distribution of number of dependents
sns.set(style="darkgrid")
ax = sns.countplot(x='NumberOfDependents', data = data)
total = float(len(data))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:.0f}%'.format(100 * p.get_height()/total),
            ha="center") 
ax.set_title('Number of dependents count')    
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))



B2_Ch9_1_E.py 
# fill missing values of NumberOfDependents
num_Dependents = pd.Series([0,1,2,3,4]).copy()
for i in data['NumberOfDependents'][data['NumberOfDependents'].isnull()].index:
    data['NumberOfDependents'][i] = num_Dependents.sample(1)

# missing value and duplicate value deletion
data=data.dropna()
data=data.drop_duplicates()



B2_Ch9_1_F.py 
# age analysis
fig = plt.figure()
ax = plt.subplot()
ax.boxplot(data['age'])
ax.set_xticklabels(['age'])
ax.set_ylabel('age (years old)')

# remove outlier of age
data = data[data['age']>18]
data = data[data['age']<100]



B2_Ch9_1_G.py 
# analysis for RevolvingUtilizationOfUnsecuredLines and DebtRatio
fig, (ax1, ax2) = plt.subplots(1,2)
x1 = data['RevolvingUtilizationOfUnsecuredLines'].astype('float') 
x2 = data['DebtRatio'].astype('float')
ax1.boxplot(x1)
ax2.boxplot(x2)
ax1.set_xticklabels(['RevolvingUtilizationOfUnsecuredLines'])
ax2.set_xticklabels(['DebtRatio'])
ax1.set_ylabel('ratio')

# remove outlier of RevolvingUtilizationOfUnsecuredLines and DebtRatio                    
data = data[(data['RevolvingUtilizationOfUnsecuredLines']>=0)&(data['RevolvingUtilizationOfUnsecuredLines']<=1)]
data = data[(data['DebtRatio']>=0)&(data['DebtRatio']<=1)]



B2_Ch9_1_K.py 
# analysis for number of time of default/past due
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
x1 = data['NumberOfTime30-59DaysPastDueNotWorse']
x2 = data['NumberOfTime60-89DaysPastDueNotWorse']
x3 = data['NumberOfTimes90DaysLate']
ax1.boxplot(x1)
ax2.boxplot(x2)
ax3.boxplot(x3)
ax1.set_xticklabels(['NumberOfTime30-59DaysPastDueNotWorse'])
ax2.set_xticklabels(['NumberOfTime60-89DaysPastDueNotWorse'])
ax3.set_xticklabels(['NumberOfTimes90DaysLate'])
ax1.set_ylabel('Number of times of past due')

# remove outlier of number of times past due                    
data = data[data['NumberOfTime30-59DaysPastDueNotWorse']<20]
data = data[data['NumberOfTime60-89DaysPastDueNotWorse']<20]
data = data[data['NumberOfTimes90DaysLate']<20]



B2_Ch9_1_L.py 
# client with good credit: 1; client with bad credit: 0
data['SeriousDlqin2yrs'] = 1-data['SeriousDlqin2yrs'] 
client_group = data['SeriousDlqin2yrs'].groupby(data['SeriousDlqin2yrs']).count()
good_client_percentage = client_group[1]/(client_group[0]+client_group[1])
bad_client_percentage = client_group[0]/(client_group[0]+client_group[1])
ax = client_group.plot(kind='bar')
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height}', (x + width/2, y + height*1.02), ha='center')    
ax.set_ylabel('Client number')
print("percentage of good clients: ", format(good_client_percentage*100, '.2f'),"%")
print("percentage of bad clients: ", format(bad_client_percentage*100, '.2f'),"%")



B2_Ch9_1_M.py
# age distribution
ax = sns.distplot(data['age'])
ax.set(xlabel='Age', ylabel='Distribution', title='Age distribution')



B2_Ch9_1_N.py 
# monthly income distribution
ax = sns.distplot(data[data['MonthlyIncome']<30000]['MonthlyIncome'])
ax.set(xlabel='Monthly income', ylabel='Distribution', title='Monthly income distribution')



B2_Ch9_1_O.py 
# heatmap: correlation of columns
corr = data.corr()
fig = plt.figure(figsize=(14, 12))
sns.heatmap(corr,annot = True, cmap="YlGnBu")
fig.tight_layout()

## 9.4 Codes

B2_Ch9_1_P.py 
from sklearn.model_selection import train_test_split

Y = data['SeriousDlqin2yrs']
X = data.iloc[:,1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, random_state=0)
train = pd.concat([Y_train,X_train], axis =1)
test = pd.concat([Y_test,X_test], axis =1)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
# save test data to a file
test.to_csv('test.csv', index=False)



B2_Ch9_1_Q.py 
import scipy.stats as stats
import numpy as np
def monotone_optimal_binning(X, Y, n):
    r = 0
    total_good = Y.sum()
    total_bad = Y.count() - total_good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.min().X, columns = ['min_' + X.name])
    d3['min_' + X.name] = d2.min().X
    d3['max_' + X.name] = d2.max().X
    d3[Y.name] = d2.sum().Y
    d3['total'] = d2.count().Y
    # calculate WOE
    d3['Goodattribute']=d3[Y.name]/total_good
    d3['badattribute']=(d3['total']-d3[Y.name])/total_bad
    d3['woe'] = np.log(d3['Goodattribute']/d3['badattribute'])
    # calculate IV
    iv = ((d3['Goodattribute']-d3['badattribute'])*d3['woe']).sum()
    d4 = (d3.sort_values(by = 'min_' + X.name)).reset_index(drop = True)
    print ("=" * 80)
    print (d4)
    cut = []
    cut.append(float('-inf'))
    for i in range(1,n+1):
        qua =X.quantile(i/(n+1))
        cut.append(round(qua,4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4, iv, cut, woe

dfx1, ivx1, cutx1, woex1 = monotone_optimal_binning(data['RevolvingUtilizationOfUnsecuredLines'], data['SeriousDlqin2yrs'], n = 10)
dfx2, ivx2, cutx2, woex2 = monotone_optimal_binning(data['age'], data['SeriousDlqin2yrs'], n = 10)
dfx4, ivx4, cutx4, woex4 = monotone_optimal_binning(data['DebtRatio'], data['SeriousDlqin2yrs'], n = 20)



B2_Ch9_1_R.py 
def self_binning(Y, X, cat):
    good = Y.sum()
    bad = Y.count()-good
    d1 = pd.DataFrame({'X':X,'Y':Y,'Bucket':pd.cut(X,cat)})
    d2 = d1.groupby('Bucket', as_index = True)
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / (1 - d3['rate'])) / (good / bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = d3.sort_values(by='min')
    print("=" * 60)
    print(d4)
    woe = list(d4['woe'].round(3))
    return d4, iv,woe

pinf = float('inf')
ninf = float('-inf')
cutx3 = [ninf, 0, 1, 3, 5, pinf]
cutx5 = [ninf,1000,2000,3000,4000,5000,6000,7500,9500,12000,pinf]
cutx6 = [ninf, 1, 2, 3, 5, pinf]
cutx7 = [ninf, 0, 1, 3, 5, pinf]
cutx8 = [ninf, 0,1,2, 3, pinf]
cutx9 = [ninf, 0, 1, 3, pinf]
cutx10 = [ninf, 0, 1, 2, 3, 5, pinf]
dfx3, ivx3,woex3 = self_binning(data['SeriousDlqin2yrs'],data['NumberOfTime30-59DaysPastDueNotWorse'],cutx3)
dfx5, ivx5,woex5 = self_binning(data['SeriousDlqin2yrs'],data['MonthlyIncome'],cutx5)
dfx6, ivx6,woex6 = self_binning(data['SeriousDlqin2yrs'],data['NumberOfOpenCreditLinesAndLoans'],cutx6) 
dfx7, ivx7,woex7 = self_binning(data['SeriousDlqin2yrs'],data['NumberOfTimes90DaysLate'],cutx7)
dfx8, ivx8,woex8 = self_binning(data['SeriousDlqin2yrs'],data['NumberRealEstateLoansOrLines'],cutx8) 
dfx9, ivx9,woex9 = self_binning(data['SeriousDlqin2yrs'],data['NumberOfTime60-89DaysPastDueNotWorse'],cutx9)
dfx10, ivx10,woex10 = self_binning(data['SeriousDlqin2yrs'],data['NumberOfDependents'],cutx10)



B2_Ch9_1_S.py 
ivlist=[ivx1,ivx2,ivx3,ivx4,ivx5,ivx6,ivx7,ivx8,ivx9,ivx10]
index=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
sns.set(style="darkgrid")
fig, ax = plt.subplots(1)
x = np.arange(len(index))+1
ax.bar(x, ivlist, width=0.4)
ax.set_xticks(x)
ax.set_xticklabels(index, rotation=0, fontsize=12)
ax.set_xlabel('Variable', fontsize=14)
ax.set_ylabel('Information value', fontsize=14)

for a, b in zip(x, ivlist):
    plt.text(a, b+0.01, '%.2f'%b, ha='center', va='bottom', fontsize=10)



B2_Ch9_1_T.py 
# woe conversion
def woe_conversion(series,cut,woe):
    list=[]
    i=0
    while i<len(series):
        try:
            value=series[i]
        except:
            i += 1
            continue
        j=len(cut)-2
        m=len(cut)-2
        while j>=0:
            if value>=cut[j]:
                j=-1
            else:
                j -=1
                m -= 1
        list.append(woe[m])
        i += 1
    return list

train['RevolvingUtilizationOfUnsecuredLines'] = pd.Series(woe_conversion(train['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
train['age'] = pd.Series(woe_conversion(train['age'], cutx2, woex2))
train['NumberOfTime30-59DaysPastDueNotWorse'] = pd.Series(woe_conversion(train['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
train['DebtRatio'] = pd.Series(woe_conversion(train['DebtRatio'], cutx4, woex4))
train['MonthlyIncome'] = pd.Series(woe_conversion(train['MonthlyIncome'], cutx5, woex5))
train['NumberOfOpenCreditLinesAndLoans'] = pd.Series(woe_conversion(train['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
train['NumberOfTimes90DaysLate'] = pd.Series(woe_conversion(train['NumberOfTimes90DaysLate'], cutx7, woex7))
train['NumberRealEstateLoansOrLines'] = pd.Series(woe_conversion(train['NumberRealEstateLoansOrLines'], cutx8, woex8))
train['NumberOfTime60-89DaysPastDueNotWorse'] = pd.Series(woe_conversion(train['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
train['NumberOfDependents'] = pd.Series(woe_conversion(train['NumberOfDependents'], cutx10, woex10))
train.dropna(how = 'any')
train.to_csv('WoeData.csv', index=False)

test['RevolvingUtilizationOfUnsecuredLines'] = pd.Series(woe_conversion(test['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
test['age'] = pd.Series(woe_conversion(test['age'], cutx2, woex2))
test['NumberOfTime30-59DaysPastDueNotWorse'] = pd.Series(woe_conversion(test['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
test['DebtRatio'] = pd.Series(woe_conversion(test['DebtRatio'], cutx4, woex4))
test['MonthlyIncome'] = pd.Series(woe_conversion(test['MonthlyIncome'], cutx5, woex5))
test['NumberOfOpenCreditLinesAndLoans'] = pd.Series(woe_conversion(test['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
test['NumberOfTimes90DaysLate'] = pd.Series(woe_conversion(test['NumberOfTimes90DaysLate'], cutx7, woex7))
test['NumberRealEstateLoansOrLines'] = pd.Series(woe_conversion(test['NumberRealEstateLoansOrLines'], cutx8, woex8))
test['NumberOfTime60-89DaysPastDueNotWorse'] = pd.Series(woe_conversion(test['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
test['NumberOfDependents'] = pd.Series(woe_conversion(test['NumberOfDependents'], cutx10, woex10))
test.dropna(how = 'any')

train_X =train.drop(['NumberRealEstateLoansOrLines','NumberOfDependents','NumberOfOpenCreditLinesAndLoans','DebtRatio','MonthlyIncome'],axis=1)
test_X =test.drop(['NumberRealEstateLoansOrLines','NumberOfDependents','NumberOfOpenCreditLinesAndLoans','DebtRatio','MonthlyIncome'],axis=1)



B2_Ch9_1_U.py 
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm

X_train =train_X.drop(['SeriousDlqin2yrs'],axis =1)
y_train =train_X['SeriousDlqin2yrs']
y_test = test_X['SeriousDlqin2yrs']
X_test = test_X.drop(['SeriousDlqin2yrs'],axis =1)
X_train = sm.add_constant(X_train)
model = sm.Logit(y_train,X_train)
result = model.fit()
print(result.summary2())



B2_Ch9_1_V.py 
from sklearn.metrics import roc_curve, auc

X2 = sm.add_constant(X_test)
resu = result.predict(X2)
FPR,TPR,threshold = roc_curve(y_test,resu)
ROC_AUC = auc(FPR,TPR)
plt.plot(FPR, TPR, 'b', label='AUC = %0.2f' % ROC_AUC)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('TPR')
plt.xlabel('FPR')



B2_Ch9_1_W.py 
# regression coeficients
coe = result.params

import numpy as np
# set benchmark score as 600; PDO as 10; Odds as 10
benchmark = 600
pdo = 10
odds = 10
factor = pdo/np.log(2)
offset = benchmark-factor*np.log(odds)
baseScore = round(offset + factor * coe[0], 0)



B2_Ch9_1_X.py 
# function to calculate addon score for a single variable
def score_addon(coe,woe,factor):
    addon = []
    for w in woe:
        score = round(coe*w*factor,0)
        addon.append(score)
    return addon

# calculate addon score 
x1 = score_addon(coe[1], woex1, factor)
x2 = score_addon(coe[2], woex2, factor)
x3 = score_addon(coe[3], woex3, factor)
x7 = score_addon(coe[4], woex7, factor)
x9 = score_addon(coe[5], woex9, factor)
print('x1: ', x1)
print('x2: ', x2)
print('x3: ', x3)
print('x7: ', x7)
print('x9: ', x9)


B2_Ch9_1_Y.py 
# compute score for single variable
def single_variable_score(series,cut,score):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(score[m])
        i += 1
    return list



B2_Ch9_1_Z.py 
from pandas import Series

test = pd.read_csv('test.csv')
test['BaseScore']=Series(np.zeros(len(test))) + baseScore
test['x1'] = Series(single_variable_score(test['RevolvingUtilizationOfUnsecuredLines'], cutx1, x1))
test['x2'] = Series(single_variable_score(test['age'], cutx2, x2))
test['x3'] = Series(single_variable_score(test['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, x3))
test['x7'] = Series(single_variable_score(test['NumberOfTimes90DaysLate'], cutx7, x7))
test['x9'] = Series(single_variable_score(test['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, x9))
test['Score'] = test['x1'] + test['x2'] + test['x3'] + test['x7'] +test['x9']  + baseScore
test.to_csv('ScoreData.csv', index=False)

filtered_columns = ['SeriousDlqin2yrs','BaseScore', 'x1', 'x2', 'x3', 'x7', 'x9', 'Score']
displaytable = test.reindex(columns = filtered_columns)
displaytable.head()




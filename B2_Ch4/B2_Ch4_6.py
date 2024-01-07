# B2_Ch4_6.py

###############
# Prepared by Ran An, Wei Lu, and Feng Zhang
# Editor-in-chief: Weisheng Jiang, and Sheng Tu
# Book 2  |  Financial Risk Management with Python
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2021
###############

B2_Ch4_6_A.py
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix


B2_Ch4_6_B.py
bankdata = pd.read_csv(r'.\BankTeleCompaign.csv')
bankdata = bankdata.dropna()
bankdata.head()

B2_Ch4_6_C.py
# plot related item/column 
sns.set(palette="pastel")
fig, ax = plt.subplots(3, 2, figsize=(6, 8))
sns.countplot(y="job",  data=bankdata, ax=ax[0, 0])
sns.countplot(x="marital", data=bankdata, ax=ax[0, 1])
sns.countplot(x="default", data=bankdata, ax=ax[1, 0])
sns.countplot(x="housing", data=bankdata, ax=ax[1, 1])
sns.countplot(x="loan", data=bankdata, ax=ax[2, 0])
sns.countplot(x="poutcome", data=bankdata, ax=ax[2, 1])
plt.tight_layout()

B2_Ch4_6_D.py
# create dunny variables with only two values: 0 or 1
data = pd.get_dummies(bankdata, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
# drop unknow columns
data.drop([col for col in data.columns if 'unknow' in col], axis=1, inplace=True)
# plot correlation heatmap
sns.heatmap(data.corr(), square=True, cmap="YlGnBu", linewidths=.01, linecolor='lightgrey', cbar_kws={"orientation": "horizontal", "shrink": 0.3, "pad": 0.25})

B2_Ch4_6_E.py
# split data into training and test sets
X = data.iloc[:,1:]
y = data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# implement logistic regression model
modelclassifier = LogisticRegression(random_state=0)
modelclassifier.fit(X_train, y_train)


B2_Ch4_6_F.py
# evaluate model via confusion matrix 
# evaluate performance of classification model on a set of test dataset with known true values 
y_pred = modelclassifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


B2_Ch4_6_G.py
# evaluate model by accuracy
model_score = modelclassifier.score(X_test, y_test)
print('Model accuracy on test set: {:.2f}'.format(model_score))

#!/usr/bin/env python
# coding: utf-8

# In[17]:


From sklearn import preprocessing

X = data #your data

Normalized_x_value = preprocessing.normalize(x)


# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Imputing missing values
from sklearn.impute import KNNImputer

from scipy.stats import chi2_contingency
# Feature engineering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# Model processing and testing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, plot_roc_curve, precision_score, recall_score
from sklearn.feature_selection import RFE
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier #Decision tree
from sklearn.naive_bayes import GaussianNB #Naive_bayes
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# In[22]:


df=pd.read_csv("C:/Users/deeti/Downloads/healthcare-dataset-stroke-data.csv")   
df.head()


# In[23]:


df.isnull().sum()


# In[24]:


#Replacing the special character to nan and then drop the columns
df['bmi'] = df['bmi'].replace('?',np.nan)
#Dropping the NaN rows now 
df.dropna(how='any',inplace=True)
df.isnull().sum()


# In[25]:


#Assigning the numeric values to the string type variables
number = LabelEncoder()
df['ever_married'] = number.fit_transform(df['ever_married'])
df['work_type'] = number.fit_transform(df['work_type'])
df['Residence_type'] = number.fit_transform(df['Residence_type'])
df['smoking_status'] = number.fit_transform(df['smoking_status'])
df['gender'] = number.fit_transform(df['gender'])


# In[26]:


df.head()


# In[27]:


X = df.drop("stroke",1)
y = df["stroke"]


# In[28]:


#Declaring the train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=0)


# In[29]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)


# In[30]:


X_train.head()


# In[31]:


# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_classif, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
 


# In[32]:


# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()


# In[33]:


# Logistic Regression
LR = LogisticRegression()
LR.fit(X_train_fs, y_train)
y_pred = LR.predict(X_test_fs)
score_LR = LR.score(X_test_fs,y_test)
print('The accuracy of the Logistic Regression model is', score_LR)
targets = ['0' , '1']
print(classification_report(y_test, y_pred,target_names=targets))


# In[34]:


# Support Vector Classifier (SVM/SVC)
from sklearn.svm import SVC
svc = SVC(gamma=0.22)
svc.fit(X_train_fs, y_train)
y_pred = svc.predict(X_test_fs)
score_svc = svc.score(X_test_fs,y_test)
print('The accuracy of SVC model is', score_svc)
targets = ['0' , '1']
print(classification_report(y_test, y_pred,target_names=targets))


# In[35]:


# Random Forest Classifier
RF = RandomForestClassifier()
RF.fit(X_train_fs, y_train)
y_pred = RF.predict(X_test_fs)
score_RF = RF.score(X_test_fs,y_test)
print('The accuracy of the Random Forest Model is', score_RF)
targets = ['0' , '1']
print(classification_report(y_test, y_pred,target_names=targets))


# In[36]:


# Decision Tree
DT = DecisionTreeClassifier()
DT.fit(X_train_fs,y_train)
y_pred = DT.predict(X_test_fs)
score_DT = DT.score(X_test_fs,y_test)
print("The accuracy of the Decision tree model is ",score_DT)
targets = ['0' , '1']
print(classification_report(y_test, y_pred,target_names=targets))


# In[37]:


# Gaussian Naive Bayes
GNB = GaussianNB()
GNB.fit(X_train_fs, y_train)
y_pred = GNB.predict(X_test_fs)
score_GNB = GNB.score(X_test_fs,y_test)
print('The accuracy of Gaussian Naive Bayes model is', score_GNB)
targets = ['0' , '1']
print(classification_report(y_test, y_pred,target_names=targets))


# In[38]:


# K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train_fs, y_train)
y_pred = knn.predict(X_test_fs)
score_knn = knn.score(X_test_fs,y_test)
print('The accuracy of the KNN Model is',score_knn)
targets = ['0' , '1']
print(classification_report(y_test, y_pred,target_names=targets))


# In[39]:


# define the evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the pipeline to evaluate
model = LogisticRegression(solver='liblinear')
fs = SelectKBest(score_func=f_classif)
pipeline = Pipeline(steps=[('anova',fs), ('lr', model)])
# define the grid
grid = dict()
grid['anova__k'] = [i+1 for i in range(X.shape[1])]
# define the grid search
search = GridSearchCV(pipeline, grid, scoring='accuracy', n_jobs=-1, cv=cv)
# perform the search
results = search.fit(X, y)
# summarize best
print('Best Mean Accuracy: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)


# In[42]:


import numpy as np 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

random_search = {'criterion': ['entropy', 'gini'],
               'max_depth': list(np.linspace(10, 1200, 10, dtype = int)) + [None],
               'max_features': ['auto', 'sqrt','log2', None],
               'min_samples_leaf': [4, 6, 8, 12],
               'min_samples_split': [5, 7, 10, 14],
               'n_estimators': list(np.linspace(151, 1200, 10, dtype = int))}

clf = RandomForestClassifier()
model = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 80, 
                               cv = 4, verbose= 5, random_state= 101, n_jobs = -1)
model.fit(X_train,y_train)


# In[43]:


predictionforest = model.best_estimator_.predict(X_test)
print(confusion_matrix(y_test,predictionforest))
print(classification_report(y_test,predictionforest))
acc3 = accuracy_score(y_test,predictionforest)


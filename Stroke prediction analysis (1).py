#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import the required libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action='ignore') #filter all the warnings


# In[4]:


#read the dataset to find useful information
data=pd.read_csv("C:/Users/deeti/Downloads/healthcare-dataset-stroke-data.csv")   
data


# In[6]:


#data exploration
data.head()


# In[7]:


data.info()


# In[8]:


data.drop("id",axis=1,inplace=True)


# In[9]:


data.describe()


# In[10]:


print("Unique Values per Variable")
for col in data.columns:
    un=data[col].unique()
    print("\n\nUnique Values in {}:\n{}".format(col,un))


# In[11]:


#preprocessing the data to learn business problems and work on analysis
#since the value is 1 for gender we will drop it.

(data["gender"]=="Other").sum()


# In[12]:


#we will check which row has this data
data[data["gender"]=="Other"]


# In[13]:


data=data.drop(3116,axis=0)


# In[14]:


data.iloc[3114:3118,:]


# In[15]:


index=[i for i in range(data.shape[0])]
data.index=index
data.iloc[3114:3118,:]


# In[17]:


#encoding the data to convert them into numerical values
#models can only impute numerical values
from category_encoders.target_encoder import TargetEncoder


# In[18]:


enc=TargetEncoder()
to_encode="work_type"
enc.fit(X=data[to_encode],y=data["stroke"])
encoded = enc.transform(data[to_encode])


# In[19]:


data["work_type"] = encoded["work_type"]


# In[20]:


data[["ever_married","Residence_type","gender"]]=pd.get_dummies(data[["ever_married","Residence_type","gender"]],drop_first=True)


# In[21]:


data.head()


# In[22]:


#Check for missing values 
print("Proportions of 'smoking' categories:")
data["smoking_status"].value_counts()/data.shape[0]


# In[23]:


smoking_mapper={"never smoked":0,"formerly smoked":1,"smokes":2,"Unknown":np.nan}


# In[24]:


for i in range(data.shape[0]):
    status=data["smoking_status"][i]
    data["smoking_status"][i]=smoking_mapper[status]


# In[25]:


data["smoking_status"].unique()


# In[26]:


#multiple imputations by chained equations (MICE)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


# In[27]:


estimator=RandomForestRegressor(max_depth=8)
mice = IterativeImputer(estimator=estimator,random_state=11,skip_complete=True)


# In[28]:


impdata=mice.fit_transform(data)


# In[29]:


impdata=pd.DataFrame(impdata,columns=data.columns)


# In[30]:


impdata.isnull().sum()


# In[31]:


impdata.head()


# In[32]:


for i in range(impdata.shape[0]):
    if impdata.loc[i,"smoking_status"]<0.5:
        impdata.loc[i,"smoking_status"]=0
    elif impdata.loc[i,"smoking_status"] <1.5:
        impdata.loc[i,"smoking_status"]=1
    else:
        impdata.loc[i,"smoking_status"]=2


# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
style.use('seaborn-darkgrid')


# In[34]:


fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(16,10))
fig.suptitle("Effect of MICE on Distributions\n",fontsize=25)
sns.histplot(x=data["bmi"],ax=axes[0,0],color="mediumspringgreen")
axes[0,0].set_title("BMI before MICE")
axes[0,0].set_xlabel(None)
sns.histplot(x=impdata["bmi"],ax=axes[0,1],color="mediumspringgreen")
axes[0,1].set_title("BMI after MICE")
axes[0,1].set_xlabel(None)
sns.countplot(x=data["smoking_status"],ax=axes[1,0],palette="cool")
axes[1,0].set_title("Smoking Status before MICE")
axes[1,0].set_xlabel(None)
sns.countplot(x=impdata["smoking_status"],ax=axes[1,1],palette="cool")
axes[1,1].set_title("Smoking Status after MICE")
axes[1,1].set_xlabel(None)
plt.show()


# # baseline model
# 

# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
rf=RandomForestClassifier(n_jobs=-1,max_depth=7)
x=impdata.drop('stroke',axis=1)
y=impdata["stroke"]


# This indicates imbalance in target variable

# In[36]:


xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=2)
rf.fit(xtrain,ytrain)
y_pred_tr=rf.predict(xtrain)
y_pred_ts=rf.predict(xtest)
train_mat=classification_report(ytrain,y_pred_tr)
test_mat=classification_report(ytest,y_pred_ts)
print("Baseline Random Forest Results:")
print("Training Classification_Report:\n{}".format(train_mat))
print("Testing Classification_Report:\n{}".format(test_mat))


# The model scored a very low recall and 1 in precision for the stroke class on the training data.
# This shows that the dataset is seriously imbalanced.
# The results on the testing data are even worse: the model is classifying everything as without stroke.

# In[51]:


stl.use("ggplot")
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=10)
from sklearn.feature_selection import mutual_info_classif


# In[40]:


plt.figure(figsize=(10,6))
cp=sns.countplot(x=data["stroke"],palette="seismic")
plt.title("Imbalance in the Target Variable\n",fontsize=30)
plt.xlabel("Stroke",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.show()


# SMOTE and its variants are common techniques for oversampling.
# Many of them are quite sensitive to outliers though.
# Let's make a quick EDA to search for outliers

# In[41]:


#work type is an encoded categorical feature, not a continuous one
continuous=["bmi","avg_glucose_level","age"] 


# In[42]:


fig,axes=plt.subplots(nrows=3,ncols=1,figsize=(12,20))
fig.suptitle("Distributions of Continuous Features",fontsize=25)


label1='Mean = {}\nMedian = {}\nStandard Deviation = {}'.format("%.2f"%data["bmi"].mean(),
                                                               data["bmi"].median(),
                                                               "%.2f"%data["bmi"].std())
sns.histplot(x=data["bmi"],ax=axes[0],color='crimson',label=label1).legend(loc='best',fontsize=15)
axes[0].set_title("BMI: Worrisome Outliers")
axes[0].set_xlabel(None)


label2='Mean = {}\nMedian = {}\nStandard Deviation = {}'.format("%.2f"%data["avg_glucose_level"].mean(),
                                                                "%.2f"%data["avg_glucose_level"].median(),
                                                                "%.2f"%data["avg_glucose_level"].std())
sns.histplot(x=data["avg_glucose_level"],ax=axes[1],color="crimson", label=label2).legend(loc='best',fontsize=15)
axes[1].set_title("Average Glucose Level: Somewhat Skewed, but Nothing Awful")
axes[1].set_xlabel(None)

label3='Mean = {}\nMedian = {}\nStandard Deviation = {}'.format("%.2f"%data["age"].mean(),
                                                                data["age"].median(),
                                                                "%.2f"%data["age"].std())
sns.histplot(x=data["age"],ax=axes[2],color="crimson",label=label3).legend(loc='best',fontsize=15)
axes[2].set_title("Age: No Outliers")
axes[2].set_xlabel(None)

plt.show()


# In[43]:


categoricals = []
for col in data.drop("stroke",axis=1):
    if not(col in continuous):
        categoricals.append(col)


# In[44]:


fig,axes=plt.subplots(nrows=7,ncols=1,figsize=(13,50))
fig.suptitle("Distributions of Categorical Features",fontsize=40)
i=0
for col in (data.drop("stroke",axis=1).columns):
    if not(col in continuous):
        sns.countplot(x=data[col],ax=axes[i],palette="cool")
        axes[i].set_title(col,fontsize=20)
        axes[i].set_xlabel(None)
        axes[i].set_ylabel("Count",fontsize=15)
        i+=1
plt.show()


# In[45]:


for col in categoricals:
    impdata[col] = impdata[col].astype("category")


# In[53]:


plt.figure(figsize=(20,15)) 
cat_mi = pd.DataFrame(np.zeros((7,7)),columns=categoricals,index=categoricals)
for i in range(7): 
    for j in range(7): print(data.columns[i]+" vs "+data.columns[j]) 
cat_mi.iloc[i,j] = mutual_info_classif(impdata[data.columns[i]].values.reshape(-1,1),impdata[data.columns[j]].values.reshape(-1,),random_state=11) 
print("finished")


# In[55]:


from sklearn.preprocessing import RobustScaler
from umap import UMAP
rb=RobustScaler()
scaled_data = rb.fit_transform(x)
ump=UMAP(random_state=11,n_neighbors=5,min_dist=0.5)
umap_data = ump.fit_transform(scaled_data)


# In[56]:


plt.figure(figsize=(16,10))
sns.scatterplot(x=umap_data[:,0],y=umap_data[:,1],hue=y,palette="seismic")
plt.title('UMAP',fontsize=45)
plt.show()


# Patients in the "1" class don't seem to be well differentiated in one separate cluster.
# 
# 

# In[57]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_data=pca.fit_transform(scaled_data)
pca_data=pd.DataFrame(pca_data,columns=["PC1","PC2"])
pca_data["Stroke"]=y
pca_data.head()


# In[58]:


#PCA principle component analysis
plt.figure(figsize=(16,8))
sns.scatterplot(x="PC1",y="PC2",hue="Stroke",data=pca_data, palette="seismic")
plt.title("Distribution across Top 2 PCA Components",fontsize=30)
plt.show()


# In[59]:


#random sampling
from imblearn.over_sampling import RandomOverSampler
proportions = [0.1,0.33,0.5,0.66,1]
oversampled_data = {}
for i in proportions:
    oversampler = RandomOverSampler(sampling_strategy=i)
    x_ros, y_ros = oversampler.fit_resample(x, y)
    x_ros = rb.fit_transform(x_ros)
    oversampled_data[i] = [x_ros,y_ros]


# In[60]:


fig,axes=plt.subplots(nrows=5,ncols=2,figsize=(20,35))
fig.suptitle("Random Oversampling Results\nWith Different Minority Class Proportions",fontsize=40)
for i in range(5):
    proportion = proportions[i]
    x_ros, y_ros = oversampled_data[proportion]
    ros_umap = ump.fit_transform(x_ros)
    sns.scatterplot(x=ros_umap[:,0],y=ros_umap[:,1],hue=y_ros,palette="seismic",ax=axes[i,0])
    axes[i,0].set_title(f"UMAP\nMinority Class Proportion = {proportion}")
    pca_ros=pca.fit_transform(x_ros)
    pca_ros=pd.DataFrame(pca_ros,columns=["PC1","PC2"])
    pca_ros["Stroke"]=y_ros
    sns.scatterplot(x="PC1",y="PC2",hue="Stroke",data=pca_ros, palette="seismic",ax=axes[i,1])
    axes[i,1].set_title(f"Top 2 PCA Components\nMinority Class Proportion = {proportion}")
plt.show()


# Training an SVM Classifier on the output of UMAP
# That last UMAP scatterplot seems to somewhat separate the classes.
# An SVM classifer with an RBF kernel might do a decent job.

# In[61]:


from sklearn.svm import SVC
sv=SVC()


# In[62]:


xg, yg = oversampled_data[1]
xgumap = ump.fit_transform(xg)
sv.fit(xgumap,yg)
xump = ump.transform(x)
y_pred = sv.predict(xump)
test_mat_ros=classification_report(y,y_pred)
print(f"Gaussian SVM on UMAP output (Testing Results):\n{test_mat_ros}")


# In[63]:


for i in range(5):
    proportion = proportions[i]
    x_ros, y_ros = oversampled_data[proportion]
    rf.fit(x_ros,y_ros)
    y_pred_ts=rf.predict(x)
    test_mat_ros=classification_report(y,y_pred_ts)
    print("Random Forest Results with Random Oversampling:")
    print("Proportion = {}\n{}\n\n".format(proportion,test_mat_ros))


# In[64]:


#random undersampling
from imblearn.under_sampling import RandomUnderSampler
undersampled_data = {}
for i in proportions:
    undersampler = RandomUnderSampler(sampling_strategy=i)
    x_rus, y_rus = undersampler.fit_resample(x, y)
    x_rus = rb.fit_transform(x_rus)
    undersampled_data[i] = [x_rus,y_rus]


# In[65]:


fig,axes=plt.subplots(nrows=5,ncols=2,figsize=(20,35))
fig.suptitle("Random Undersampling Results\nWith Different Minority Class Proportions",fontsize=40)
for i in range(5):
    proportion = proportions[i]
    x_rus, y_rus = undersampled_data[proportion]
    rus_umap = ump.fit_transform(x_rus)
    sns.scatterplot(x=rus_umap[:,0],y=rus_umap[:,1],hue=y_rus,palette="seismic",ax=axes[i,0])
    axes[i,0].set_title(f"UMAP\nProportion = {proportion}")
    pca_rus=pca.fit_transform(x_rus)
    pca_rus=pd.DataFrame(pca_rus,columns=["PC1","PC2"])
    pca_rus["Stroke"]=y_rus
    sns.scatterplot(x="PC1",y="PC2",hue="Stroke",data=pca_rus, palette="seismic",ax=axes[i,1])
    axes[i,1].set_title(f"Top 2 PCA Components\nProportion = {proportion}")
plt.show()


# In[66]:


#model evalusation using random undersampling
for i in range(5):
    proportion = proportions[i]
    x_rus, y_rus = undersampled_data[proportion]
    rf.fit(x_rus,y_rus)
    y_pred_ts=rf.predict(x)
    test_mat_ros=classification_report(y,y_pred_ts)
    print("Random Forest Results with Random Undersampling:")
    print("Proportion = {}\n{}\n\n".format(proportion,test_mat_ros))


# In[67]:


#detailed feature extraction and Selection
#combining features using oversampling dataset to enhance the work we have done so far
xfe, yfe = RandomOverSampler(sampling_strategy=0.25, random_state=11).fit_resample(x, y)


# In[68]:


xref = xfe.copy(deep=True)
xtest = x.copy(deep=True)


# In[69]:


xfe["Blood&Heart"]=xfe["hypertension"]*xfe["heart_disease"]
xtest["Blood&Heart"]=xtest["hypertension"]*xtest["heart_disease"]
xfe["Effort&Duration"] = xfe["work_type"]*(xfe["age"])
xtest["Effort&Duration"] = xtest["work_type"]*(xtest["age"])
xfe["Obesity"] = xfe["bmi"]*xfe["avg_glucose_level"]/1000
xtest["Obesity"] = xtest["bmi"]*xtest["avg_glucose_level"]/1000
xfe["AwfulCondition"] = xfe["Obesity"] * xfe["Blood&Heart"] * xfe["smoking_status"]
xtest["AwfulCondition"] = xtest["Obesity"] * xtest["Blood&Heart"] * xtest["smoking_status"]
xfe["AwfulCondition"].unique()


# In[70]:


#effect of residence type on Effort&Duration
xfe.head()


# In[71]:


#principle component analysis
from sklearn.decomposition import PCA
pca = PCA()
pca_feats = pca.fit_transform(rb.fit_transform(xref))
pca.explained_variance_ratio_


# In[72]:


list(range(1,11))


# In[73]:


fig=plt.figure(figsize=(20,9))
sns.barplot(x=list(range(1,11)),y=pca.explained_variance_,palette = 'Reds_r')
plt.ylabel('Variation',fontsize=15)
plt.xlabel('PCA Components',fontsize=15)
plt.title("PCA Components\nRanked by Variation",fontsize=25)
plt.show()


# In[74]:


xfe["PC1"], xfe["PC2"] = pca_feats[:,0], pca_feats[:,1]
xtestpca = pca.transform(rb.transform(x))
xtest["PC1"], xtest["PC2"] = xtestpca[:,0], xtestpca[:,1]
xfe.head()


# In[75]:


#independent component analysis
from sklearn.decomposition import FastICA as ICA
ica = ICA(random_state=11)
xica = ica.fit_transform(X=rb.fit_transform(xref))
ncomp = ica.components_.shape[0]


# In[76]:


fig,axes=plt.subplots(ncols=1,nrows=ncomp,figsize=(20,10*ncomp))
fig.suptitle("Target Distributions\nAcross ICA Components",fontsize=40)
for i in range(ncomp):
    sns.boxenplot(y=xica[:,i], x=yfe, palette="seismic",showfliers=True,ax=axes[i])
    axes[i].set_xlabel("Stroke",fontsize=15)
    axes[i].set_ylabel(f"IC{i+1}",fontsize=25)
plt.show()


# In[77]:


xfe["ICA"] = xica[:,3]
xtest["ICA"] = ica.transform(rb.transform(x))[:,3]


# In[78]:


#linear Discriminant analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
xlda = lda.fit_transform(rb.fit_transform(xref),yfe)
xlda = xlda.reshape((xlda.shape[0],))
plt.figure(figsize=(20,8))
sns.boxenplot(y=xlda, x=yfe, color='crimson',showfliers=True)
plt.title("Separation of Classes with LDA",fontsize=30)
plt.xlabel("Stroke",fontsize=20)
plt.show()


# In[79]:


xfe["LDA"] = xlda
xtest["LDA"] = lda.transform(rb.transform(x)).reshape((x.shape[0],))


# The 2 following sections are an attempt at using cluster labels as features.
# KMeans did a decent job.
# DBSCAN didn't.

# In[80]:


from sklearn.cluster import KMeans


# In[82]:


inertias = []

ks=list(range(1,10))

#xkm = rb.fit_transform(xref)

xfesc = rb.fit_transform(xfe)
xtsc = rb.transform(xtest)

for k in ks:
    
    model=KMeans(n_clusters=k)
    
    model.fit(xfesc)
    
    inertias.append(model.inertia_)


# In[ ]:





#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


#Importing dataset
dataset=pd.read_csv('winequality-red.csv')
X=dataset.iloc[:,:11].values
y=dataset.iloc[:,11].values


#Exploratory data analysis
for i in range(len(dataset.columns)):
    print("Column",i+1,":",list(dataset.columns)[i])
unique,counts=np.unique(y,return_counts=True)
unique_values=['Very Poor','Poor','Normal','Good','Very good','Excellent']
plt.figure(num=1)
sns.barplot(x=unique,y=counts,data=dataset)
plt.title('Wine quality')
plt.xlabel('Wine quality numbers')
plt.ylabel('Number of wines')
plt.show()

plt.figure(num=2)
cor=dataset.corr().iloc[:11,11:12]
plt.title("The Correlation Map between quality and all components")
sns.heatmap(cor,cmap='gray',annot=True)



#Splitting dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


#Optiizing and Scaling 
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


#Training Models without Dimension Reductionality Using Grid_Search_CV
dic={}
#Training Logistic Regression
Logic=LogisticRegression()
parameters={"penalty":['l2'],'C':[1.0,0.5,2.0,0.7],'solver':('newton-cg','lbfgs', 'liblinear', 'sag', 'saga')}
grid_Logistic=GridSearchCV(Logic,parameters,cv=5,scoring='accuracy',n_jobs=-1)
gd_logic=grid_Logistic.fit(X_train,y_train)
Logic_final=gd_logic.best_estimator_
Logic_final.fit(X_train,y_train)
cm=confusion_matrix(y_true=y_test,y_pred=Logic_final.predict(X_test))
acc=accuracy_score(y_test,Logic_final.predict(X_test))
dic["Logistic Regression"]=acc



#Training KNN classifier
knn=KNeighborsClassifier()
parameters_1={'n_neighbors':[4,5,6,7,8,9,10],'weights':['uniform','distance'],'algorithm':['auto', 'ball_tree','kd_tree','brute'],'leaf_size':[30,20,10,40]}
grid_knn=GridSearchCV(knn,parameters_1,cv=5,scoring='accuracy',n_jobs=-1)
gd_knn=grid_knn.fit(X_train,y_train)
knn_final=gd_knn.best_estimator_
knn_final.fit(X_train,y_train)
cm_1=confusion_matrix(y_true=y_test,y_pred=knn_final.predict(X_test))
acc_1=accuracy_score(y_test,knn_final.predict(X_test))
dic["K-Neighbors"]=acc_1


#Traininig SVM classifier
svm=SVC()
parameters_2={'C':[7.0],'kernel':['linear', 'poly', 'rbf', 'sigmoid']}
grid_svm=GridSearchCV(svm,parameters_2,cv=5,scoring='accuracy',n_jobs=-1)
gd_svm=grid_svm.fit(X_train,y_train)
svm_final=gd_svm.best_estimator_
svm_final.fit(X_train,y_train)
cm_2=confusion_matrix(y_true=y_test,y_pred=svm_final.predict(X_test))
acc_2=accuracy_score(y_test,svm_final.predict(X_test))
dic["SVM classifier"]=acc_2

#Training Naive Bayes Theorem
gauss=GaussianNB()
gauss.fit(X_train,y_train)
cm_3=confusion_matrix(y_true=y_test,y_pred=gauss.predict(X_test))
acc_3=accuracy_score(y_test,gauss.predict(X_test))
dic["Naive Bayes"]=acc_3

#Training Decision Tree Classifier
tree=DecisionTreeClassifier()
parameters_3={"criterion":['gini','entropy'],"splitter":['best','random'],}
grid_tree=GridSearchCV(tree,parameters_3,cv=5,scoring='accuracy',n_jobs=-1)
grid_tree.fit(X_train,y_train)
tree_final=grid_tree.best_estimator_
tree_final.fit(X_train,y_train)
cm_4=confusion_matrix(y_true=y_test,y_pred=tree_final.predict(X_test))
acc_4=accuracy_score(y_test,tree_final.predict(X_test))
dic["Decision Tree"]=acc_4



#Training Random Forest Classifier
random=RandomForestClassifier()
parameters_4={'n_estimators':[5,10,15,20,17,25,24,30],'criterion':['gini','entropy']}
grid_random=GridSearchCV(random,parameters_4,cv=5,scoring='accuracy',n_jobs=-1)
grid_random.fit(X_train,y_train)
random_final=grid_random.best_estimator_
random_final.fit(X_train,y_train)
cm_5=confusion_matrix(y_true=y_test,y_pred=random_final.predict(X_test))
acc_5=accuracy_score(y_test,random_final.predict(X_test))
dic["Random Classifier"]=acc_5


#Training XGD boost
model=xgb.XGBClassifier()
model.fit(X_train,y_train)
cm_6=confusion_matrix(y_true=y_test,y_pred=model.predict(X_test))
acc_6=accuracy_score(y_test,model.predict(X_test))
dic["XG Boost"]=acc_6

#Analysing the results of the Results
Estimators=[]
Accuracy=[]
for i in dic:
    Estimators.append(i)
    Accuracy.append(dic[i]*100)
d={'Estimators':Estimators,"Accuracy":Accuracy}
df=pd.DataFrame(data=d)
plt.figure(num=3)
plt.ylim(0,100)
plt.title("All classification estimators with accuracy score")
sns.barplot(x='Estimators',y='Accuracy',data=df)
plt.show()



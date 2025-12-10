import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier,BaggingClassifier,AdaBoostClassifier,StackingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
data_dict={
    'Maths':[85,78,92,65,70,88,90,55,60,75,80,95,50,68,72],
    'English':[80,82,88,70,75,90,85,60,62,78,74,91,55,66,73],
    'Science':[88,76,95,60,68,85,89,58,63,70,77,92,54,67,71],
    'History':[90,74,94,67,69,87,88,59,64,72,75,93,53,65,70],
    'Target':[1,1,1,0,0,1,1,0,0,1,1,1,0,0,0]
}
data=pd.DataFrame(data_dict)
print("DATABASE TABLE")
print(tabulate(data,headers='keys',tablefmt='psql'))
X=data.drop('Target',axis=1)
y=data['Target']
X_train,X_test,y_train,y_test=train_test_split(X, y, random_state=42, test_size=0.2)
df1=DecisionTreeClassifier(random_state=42)
df2=KNeighborsClassifier(n_neighbors=3)
df3=SVC(probability=True, random_state=42)
df4=LogisticRegression(max_iter=200,random_state=42)
voting_df=VotingClassifier(estimators=[
    ('dt',df1),
    ('knn', df2),
    ('svc', df3),
    ('lr', df4)
], voting='soft')
voting_df.fit(X_train, y_train)
y_pred_voting=voting_df.predict(X_test)
acc_voting=accuracy_score(y_test,y_pred_voting)

bagging_df=BaggingClassifier(estimator=df1,n_estimators=50,random_state=42)
bagging_df.fit(X_train,y_train)
y_pred_bagging=bagging_df.predict(X_test)
acc_bagging = accuracy_score(y_test, y_pred_bagging)

boosting_df = AdaBoostClassifier(estimator=df1, n_estimators=50, random_state=42)
boosting_df.fit(X_train,y_train)
y_pred_boosting=boosting_df.predict(X_test)
acc_boosting=accuracy_score(y_test,y_pred_boosting)

Stacking_df=StackingClassifier(
    estimators=[('df',df1),('knn',df2),('svc',df3)],
    final_estimator=LogisticRegression(max_iter=200),
    passthrough=True
)
Stacking_df.fit(X_train,y_train)
y_pred_stacking = Stacking_df.predict(X_test)
acc_stacking=accuracy_score(y_test,y_pred_stacking)

print("\nEnsembling Method Performance on sample Maths/English Dataset")
print(f"Voting classifier Accuracy: {acc_voting:.2f}")
print(f"Bagging classifier Accuracy: {acc_bagging:.2f}")
print(f"Boosting classifier Accuracy: {acc_boosting:.2f}")
print(f"Stacking classifier Accuracy: {acc_stacking:.2f}")
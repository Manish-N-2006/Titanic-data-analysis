import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
#print(df.head())
#print(df.info())
#print(df.isnull().sum())

data = df.copy()

data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1,inplace=True)
x = data['Age'].median()
data['Age'] = data['Age'].fillna(x)
data = data.dropna(subset=['Embarked'])
#print(data.isnull().sum())
#print(data.head())

from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])
le_embark = LabelEncoder()
data['Embarked'] = le_embark.fit_transform(data['Embarked'])
#print(data.head())

X = data.drop(['Survived'], axis=1)
y = data['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state = 42, max_iter=1000)
model.fit(X_train,y_train)
y_pr = model.predict(X_test)

from sklearn.metrics import accuracy_score
sc = accuracy_score(y_test,y_pr)
print(f"Logistic Regression: \n Accuracy: {sc*100:.2f}%")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

modtree = DecisionTreeClassifier(random_state=42)
modtree.fit(X_train,y_train)
y_dt_pr = modtree.predict(X_test)
acc_dt = accuracy_score(y_test,y_dt_pr)
print(f"Decision Tree: \n Accuracy: {acc_dt*100:.2f}%")

modrand = RandomForestClassifier(random_state=42)
modrand.fit(X_train,y_train)
y_rf_pr = modrand.predict(X_test)
acc_rd = accuracy_score(y_test,y_rf_pr)
print(f"Random Forest: \n Accuracy: {acc_rd*100:.2f}%")

import matplotlib.pyplot as plt
#plt.figure(figsize=(2,2))
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accur = [sc,acc_dt,acc_rd]
'''plt.bar(models,accur)
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.title("Model comparision")'''

feature_importance = modrand.feature_importances_
features = X.columns
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
})
plt.figure(figsize=(6,8))
plt.barh(importance_df['Feature'],importance_df['Importance'])
plt.ylabel('Feature')
plt.xlabel('Importance Score')
plt.title('Feature Importance (Random Forest)')
plt.show()

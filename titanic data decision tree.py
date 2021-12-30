import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Import data 
titanic_data = pd.read_csv("C:\Amit AE\Documents/titanic.csv")
titanic_data.head(5)
titanic_data.tail(5)

# data information
titanic_data.info()
titanic_data["age"].plot.hist()
plt.hist(titanic_data["age"])

# convert "age" column object to float type 
titanic_data["age"] = pd.to_numeric(titanic_data.age, errors='coerce')
titanic_data.info()

titanic_data["age"].plot.hist()

#Converting var "fare" from object type to float type
titanic_data["fare"] = pd.to_numeric(titanic_data.fare, errors='coerce')
titanic_data.info()


titanic_data["fare"].plot.hist()

# finding any null value 
titanic_data.isnull()
titanic_data.isnull().sum()

sns.heatmap(titanic_data.isnull(),yticklabels=False, cmap="viridis")


titanic_data.dropna(subset=['fare'],inplace=True)
sns.heatmap(titanic_data.isnull(),yticklabels=False)

# Replace age missing value by age mean value
titanic_data["age"].fillna(titanic_data["age"].mean(), inplace=True)
sns.heatmap(titanic_data.isnull(),yticklabels=False)


titanic_data.isnull().sum()


titanic_data.info()

#create "sex" column to dummies column
pd.get_dummies(titanic_data["sex"])
pd.get_dummies(titanic_data["sex"],drop_first=True)

Sex_Dummy = pd.get_dummies(titanic_data["sex"],drop_first=True)
Sex_Dummy.head(5)

# create "embarked" column to dummies column
pd.get_dummies(titanic_data["embarked"])
Embardked_Dummy = pd.get_dummies(titanic_data["embarked"],drop_first=True)
Embardked_Dummy.head(5)

pd.get_dummies(titanic_data["pclass"])
PClass_Dummy = pd.get_dummies(titanic_data["pclass"],drop_first=True)
PClass_Dummy.head(5)


titanic_data = pd.concat([titanic_data,Sex_Dummy,PClass_Dummy,Embardked_Dummy],axis=1)
titanic_data.head(5)

titanic_data.drop(["sex","embarked","pclass","Passenger_id","name","ticket"],axis=1,inplace=True)
titanic_data.head(5)


x=titanic_data.drop("survived",axis=1)
y=titanic_data["survived"]

# split data in train & test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# import model 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy',max_depth = 4, min_samples_leaf=4)

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
# accuracy=====79.8%


from sklearn import tree

#Simple Decision Tree
tree.plot_tree(classifier)

tree.plot_tree(classifier,filled = True)
#Although the Decision tree shows class name & leafs are colred but still its view is blurred.

#Lets create a blank chart of desired size using matplotlib library and place our Decision tree there.
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=1000)
#The above line is used to set the pixels of the Decision Trees nodes so that
#the content mentioned in each node of Decision tree is visible.
cn=['0','1']
tree.plot_tree(classifier,class_names=cn,filled = True)





















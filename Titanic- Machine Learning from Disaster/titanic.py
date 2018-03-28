# TITANIC SURVIVAL

# importing libraries
import pandas as pd
from pandas import Series,DataFrame

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#importing the dataset
titanic_df = pd.read_csv("train.csv")
test_df    = pd.read_csv("test.csv")
te=pd.read_csv("gender_submission.csv")

# visualization of datasets
titanic_df.info()
print("----------------------------")
test_df.info()

# drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_df = titanic_df.drop(['PassengerId'], axis=1)

# Ticket
# majority of tickets have their first digit = 1, 2, or 3, which probably also represent different classes. So I just keep the
# first element (a letter or a single-digit number) of these ticket names
titanic_df.Ticket = titanic_df.Ticket.map(lambda x: x[0])
test_df.Ticket = test_df.Ticket.map(lambda x: x[0])

# inspect the correlation between Ticket and Survived
titanic_df[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean()

# inspect the amount of people for each type of tickets
titanic_df['Ticket'].value_counts()

# The main categories of Ticket are "1", "2", "3", "P", "S", and "C", so I will combine all the others into "4"
titanic_df['Ticket'] = titanic_df['Ticket'].replace(['A','W','F','L','5','6','7','8','9'], '4')
test_df['Ticket'] = test_df['Ticket'].replace(['A','W','F','L','5','6','7','8','9'], '4')

# dummy encoding
titanic_df = pd.get_dummies(titanic_df,columns=['Ticket'])
test_df = pd.get_dummies(test_df,columns=['Ticket'])

# Name
titanic_df.Name.head(5)

# extract the titles from the names like Mr where first split by , and take 1 i.e second string then split second string by . 
# and take 0 i.e. first string 
# strip removes space
titanic_df['Title'] = titanic_df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())
test_df['Title'] = test_df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())

# inspect the amount of people for each title
titanic_df['Title'].value_counts()

# above we can see some have less number so some can be be merged into some of most occured categories. For the rest, 
# store in 'Others'
titanic_df['Title'] = titanic_df['Title'].replace('Mlle', 'Miss')
titanic_df['Title'] = titanic_df['Title'].replace(['Mme','Lady','Ms'], 'Mrs')
titanic_df.Title.loc[ (titanic_df.Title !=  'Master') & (titanic_df.Title !=  'Mr') & (titanic_df.Title !=  'Miss') 
                    & (titanic_df.Title !=  'Mrs')] = 'Others'

test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace(['Mme','Lady','Ms'], 'Mrs')
test_df.Title.loc[ (test_df.Title !=  'Master') & (test_df.Title !=  'Mr') & (test_df.Title !=  'Miss') 
                    & (test_df.Title !=  'Mrs')] = 'Others'

# inspect the correlation between Title and Survived
titanic_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# drop name
titanic_df = titanic_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

# dummy encoding
titanic_df = pd.get_dummies(titanic_df,columns=['Title'])
test_df = pd.get_dummies(test_df,columns=['Title'])

# Embarked
# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

# inspect the correlation between Embarked and Survived as well as some other features
titanic_df[['Embarked', 'Survived','Pclass','Fare', 'Age', 'Sex']].groupby(['Embarked'], as_index=False).mean()

# plots the point estimate and confidence interval. size is for stretching of plot and aspect for
# stretching of x-axis
sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
# counts each category occurance
sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
# counts each category occurance according to survival
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)
# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

# Either to consider Embarked column in predictions,
# and remove "S" dummy variable, 
# and leave "C" & "Q", since they seem to have a good rate for Survival.
# OR
# don't create dummy variables for Embarked column, just drop it, 
# because logically, Embarked doesn't seem to be useful in prediction.

# get_dummies produces table where each column title is categories in embarked and in table each row 
# tells which category is there in that specific row by putting 0 or 1
#  s1 = ['a', 'b', np.nan]
# get_dummies(s1, dummy_na=True)
#    a  b  NaN
# 0  1  0    0
# 1  0  1    0
# 2  0  0    1

# drop all passengers who have S embarkment
# When inplace=True is passed, the data is renamed in place (it returns nothing)
# When inlace=False is passed (this is the default value, so isn't necessary), 
# performs the operation and returns a copy of the object
embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

titanic_df = titanic_df.join(embark_dummies_titanic)
test_df    = test_df.join(embark_dummies_test)

titanic_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)

# Fare
# only for test_df, since there is a missing "Fare" values
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# convert from float to int
titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)

# plot frequency vs fare
# bins for linespacong, xlim is x-aixs limits
titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))

# Divide 'fare' into groups
f = pd.qcut(titanic_df['Fare'], 4)
print (titanic_df.groupby(f).Survived.mean())

# Assign number to fare limits
titanic_df.Fare.loc[ (titanic_df.Fare <= 7) ]= 0
titanic_df.Fare.loc[ (titanic_df.Fare > 7) & (titanic_df.Fare <=  14) ]= 1
titanic_df.Fare.loc[ (titanic_df.Fare > 14) & (titanic_df.Fare <=  31) ]= 2
titanic_df.Fare.loc[ (titanic_df.Fare > 31) ]= 3


test_df.Fare.loc[ (test_df.Fare <= 7) ]= 0
test_df.Fare.loc[ (test_df.Fare > 7) & (test_df.Fare <=  14) ]= 1
test_df.Fare.loc[ (test_df.Fare > 14) & (test_df.Fare <=  31) ]= 2
test_df.Fare.loc[ (test_df.Fare > 31) ]= 3

# Age 

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# get average, std, and number of NaN values in titanic_df
average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
# get random(from a, to b, n numbers wanted)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# plot original Age values
# NOTE: drop all null values, and convert to int
titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# fill NaN values in Age column with random values generated
titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

# convert from float to int
titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age']    = test_df['Age'].astype(int)
        
# plot new Age Values
titanic_df['Age'].hist(bins=70, ax=axis2)

# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, titanic_df['Age'].max()))
facet.add_legend()

# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = titanic_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)

# Divide 'Age' into groups
a = pd.cut(titanic_df['Age'], 5)
print (titanic_df.groupby(a).Survived.mean())

# Assign number to Age limits
titanic_df.Age.loc[ (titanic_df.Age <= 16) ]= 0
titanic_df.Age.loc[ (titanic_df.Age > 16) & (titanic_df.Age <=  32) ]= 1
titanic_df.Age.loc[ (titanic_df.Age > 32) & (titanic_df.Age <=  48) ]= 2
titanic_df.Age.loc[ (titanic_df.Age > 48) & (titanic_df.Age <=  64) ]= 3
titanic_df.Age.loc[ (titanic_df.Age > 64) ]= 4

test_df.Age.loc[ (test_df.Age <= 16) ]= 0
test_df.Age.loc[ (test_df.Age > 16) & (test_df.Age <=  32) ]= 1
test_df.Age.loc[ (test_df.Age > 32) & (test_df.Age <=  48) ]= 2
test_df.Age.loc[ (test_df.Age > 48) & (test_df.Age <=  64) ]= 3
test_df.Age.loc[ (test_df.Age > 64) ]= 4

# Cabin
# It has a lot of NaN values, so it won't cause a remarkable impact on prediction
titanic_df.drop("Cabin",axis=1,inplace=True)
test_df.drop("Cabin",axis=1,inplace=True)

# Family

# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"] + 1
test_df['Family'] =  test_df["Parch"] + test_df["SibSp"] + 1

# inspect the correlation between Family and Survived
titanic_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()

# inspect the amount of people for each Family size
titanic_df['Family'].value_counts()

# survival rate increases with the family size, but not beyond Family = 4. I will combine all the data with Family > 4 into one
# category i.e Family = 0, such that the survival rate always increases as Family increases.
titanic_df.Family = titanic_df.Family.map(lambda x: 0 if x > 4 else x)
test_df.Family = test_df.Family.map(lambda x: 0 if x > 4 else x)

# drop Parch & SibSp
titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)
test_df    = test_df.drop(['SibSp','Parch'], axis=1)

# Sex

# inspect the correlation between Sex and Survived
titanic_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()

# As we see, children(age < ~16 i.e we assigned it =0) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age ==0 else sex
    
titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)
test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)

# No need to use Sex column since we created Person column
titanic_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

titanic_df = titanic_df.join(person_dummies_titanic)
test_df    = test_df.join(person_dummies_test)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Person', data=titanic_df, ax=axis1)

# average of survived for each Person(male, female, or child)
person_perc = titanic_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])

titanic_df.drop(['Person'],axis=1,inplace=True)
test_df.drop(['Person'],axis=1,inplace=True)

# Pclass

# get mean of survival for each category in Pclass
# as_index=false is used to display serial number
titanic_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5)

# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
# naming each column in pclass_dummies_titanic
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

titanic_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)

titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df    = test_df.join(pclass_dummies_test)

X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
logreg.score(X_train, Y_train)

# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
coeff_df

cm=confusion_matrix(te['Survived'],Y_pred)
cm

# Support Vector Machines
svc = SVC(kernel='linear',C=0.025)
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
svc.score(X_train, Y_train)

cm=confusion_matrix(te['Survived'],Y_pred)
cm

# KNN Neighbours
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
knn.score(X_train, Y_train)

cm=confusion_matrix(te['Survived'],Y_pred)
cm

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
gaussian.score(X_train, Y_train)

cm=confusion_matrix(te['Survived'],Y_pred)
cm

# Random Forests
random_forest = RandomForestClassifier(n_estimators=500,criterion='entropy',random_state=0,n_jobs=-1,warm_start=True,
                                       max_depth=6,min_samples_leaf=2,verbose=0)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)

cm=confusion_matrix(te['Survived'],Y_pred)
cm

# ExtraTreesClassifier
ec = ExtraTreesClassifier(n_jobs=-1,n_estimators=500,max_depth=8,min_samples_leaf=2,verbose=0)
ec.fit(X_train, Y_train)
Y_pred = ec.predict(X_test)
ec.score(X_train, Y_train)

cm=confusion_matrix(te['Survived'],Y_pred)
cm

# Submission
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic_predicted.csv', index=False)

# AdaBoostClassifier
ad = AdaBoostClassifier(n_estimators=500,learning_rate=0.75)
ad.fit(X_train, Y_train)
Y_pred = ad.predict(X_test)
ad.score(X_train, Y_train)

cm=confusion_matrix(te['Survived'],Y_pred)
cm

#GradientBooster
gb = GradientBoostingClassifier(n_estimators=500,max_depth=5,min_samples_leaf=2,verbose=0,random_state=0)
gb.fit(X_train, Y_train)
Y_pred = gb.predict(X_test)
gb.score(X_train, Y_train)

cm=confusion_matrix(te['Survived'],Y_pred)
cm

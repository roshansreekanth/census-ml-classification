#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:56:57 2020

@Student name: Roshan Sreekanth

@Student ID: R00170592

@Student Course Name: SDH3B


"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection._validation import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def clean_data(df):

    df['workType'] = df['workType'].str.strip()
    df['workType'] = df['workType'].replace('?', 'Unknown')
    df['workType'] = df['workType'].fillna(df['workType'].mode().values[0]) # All the empty cells from workType attributes are filled in with the value with the highest frequency.

    df['education'] = df['education'].str.strip()
    df['education'] = df['education'].fillna(df['education'].mode().values[0]) # All the empty cells from education attributes are filled in with the value with the highest frequency.
    
    df['job'] = df['job'].str.strip()
    df['job'] = df['job'].replace('?', np.NaN)
    df['job'] = df['job'].replace(np.NaN, 'Other')

    df['Income'] = df['Income'].str.strip()

    replace_chars = [" ->", " <=", "."]

    for char in replace_chars:
        df['Income'] = df['Income'].str.replace(char, "")
    
    df['Income'] = df['Income'].fillna(df['Income'].mode().values[0]) # All the empty cells from Income attributes are filled in with the value with the highest frequency

    df['Gender'] = df['Gender'].str.strip()
 
def encode_data(df):
    # Convert the categorical values in workType attribute to numerical values
    df['workType'] = df['workType'].map(dict(zip(df['workType'].unique(), np.arange(len(df['workType'].unique())))))
    df['workType'] = df['workType'].astype(int)

    # Convert the categorical values in education attribute to numerical values
    df['education'] = df['education'].map(dict(zip(df['education'].unique(), np.arange(len(df['education'].unique())))))
    df['education'] = df['education'].astype(int)

    # Convert the categorical values in job attribute to numerical values
    df['job'] = df['job'].map(dict(zip(df['job'].unique(), np.arange(len(df['job'].unique())))))
    df['job'] = df['job'].astype(int)

    # Convert the categorical values in education attribute to numerical values
    df['Income'] = df['Income'].map(dict(zip(df['Income'].unique(), np.arange(len(df['Income'].unique())))))
    df['Income'] = df['Income'].astype(int)

 
def task1():

    df = pd.read_csv('people.csv')
    
    clean_data(df)
    encode_data(df)
    
    younger_df = df[df['age'] < 50] # Filter by people younger than 50

    X = younger_df[['education', 'workType', 'job']]
    y = younger_df[['Income']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3) # Using 33% of the data individuals as test set
    
    clfDT = DecisionTreeClassifier()
    clfDT.fit(X_train, y_train)
    print("Train Accuracy 33%: ", clfDT.score(X_train, y_train))
    print("Test Accuracy 33%: ", clfDT.score(X_test, y_test))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.95) # Using 95% of the data individuals as test set
    clfDT.fit(X_train, y_train)
    print("Train Accuracy 95%", clfDT.score(X_train, y_train))
    print("Test Accuracy 95%", clfDT.score(X_test, y_test))

    '''
    Observations

    First 33% of the data was used for testing, and then 95% of the data was used for testing.

    A larger testing data cna cause greater variance, and a larger amount of training data can cacuse overfitting.

    In this case, the dataset has a pretty large amount of data. It would be better to use a 33% test percentage, as
    the risk of overfitting is not very high. Using 95% test data can cause more innacuracy and variance. 
    '''

def task2():
    
    df = pd.read_csv('people.csv')  
    
    clean_data(df)

    # Convert the categorical values in Income attribute to numerical values.
    df['Income'] = df['Income'].map(dict(zip(df['Income'].unique(), np.arange(len(df['Income'].unique())))))
    df['Income'] = df['Income'].astype(int)

    twoColumns = df[['workType', 'Income']]
    groups = twoColumns.groupby(['workType', 'Income'])

    attribute_value = "Private"

    sumVal1 = groups.size()[attribute_value].sum()    

    pi = groups.size()[attribute_value]/sumVal1        
    log2s = np.log2((groups.size()[attribute_value]/sumVal1))    
    entropies =  log2s.multiply( -1* pi)    
    print("Entropy for Private: ", entropies.sum())

    attribute_value = "State-gov"

    sum_val_2 = groups.size()[attribute_value].sum()    

    pi = groups.size()[attribute_value]/sum_val_2        
    log2s = np.log2((groups.size()[attribute_value]/sum_val_2))    
    entropies =  log2s.multiply( -1* pi)    
    print("Entropy for State-gov: ", entropies.sum())

    '''
    Observations

    The entropy for attribute workType with value "Private" is 0.7562417707440523
    The entropy for attribute workType with value "State-gov" is 0.8379148918407011

    The entropy value ranges from 0 (perfectly classified) to log(k) (random)

    The workType "Private" activity has a lower level of uncertainity.
    The workType "State-gov" has a higher level of uncertainity.
    '''

def task3():
    df = pd.read_csv('people.csv')

    # This should go in the clean data function but I wanted to show it explicitly
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age'] = df['age'].fillna(round(df['age'].mean()))
    df['age'] = df['age'].astype(int)

    df = df.dropna()

    clean_data(df)
    encode_data(df)

    older_females = df[(df['Gender'] == 'Female') & (df['age'] > 30)] # Dataset for females older than 30

    X = older_females[['education', 'age', 'job']]
    y = older_females[['workType']].values.ravel()
    
    models = []

    models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
    models.append(('Naive Bayes', GaussianNB()))
    models.append(('Random Forest Classifier', RandomForestClassifier()))

    results = {}

    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=5, scoring='accuracy', return_train_score=True) # Setting Cross Validation to 5
        results[name] = cv_results
    
    for models in results:
        print(models)       
        print('Training:',results[models]['train_score'].mean())        
        print('Test: ',results[models]['test_score'].mean())

    labels = ['Decision Tree Classifier', 'Naive Bayes', 'Random Forest Classifier']
    train_means = [results[models]['train_score'].mean() for models in results]
    test_means = [results[models]['test_score'].mean() for models in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, train_means, width, label="Training Accuracy")
    rects2 = ax.bar(x + width/2, test_means, width, label = "Testing Accuracy")

    ax.set_xlabel('Classifier')
    ax.set_ylabel('Average Accuracy')
    ax.set_title('Accuracy Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.show()

    '''
    Observations

    All three classifiers are close in accuracy.

    Naive Bayes Classifier has higher testing accuracy than Decision Tree and Random Forset Classifiers.

    Decision Tree and Random Forest Classifiers have higher training accuracy than Naive Bayes Classifier.

    Naive Bayes Classifier is more consistent and the training accuracy is similar to the testing accuracy.
    
    '''

def task4():
    df = pd.read_csv('people.csv')
    df = df.dropna()

    clean_data(df)
    encode_data(df)

    three_columns = df[['Income', 'education', 'job']]
    
    scaler = MinMaxScaler()
    three_columns = scaler.fit_transform(three_columns)

    costs_three_column = []

    print("Clustering Costs for Income, education, and job") # For up to 7 clusters
    for i in range(7):
        kmeans = KMeans(n_clusters = i + 1).fit(three_columns)
        costs_three_column.append(kmeans.inertia_)
        print(str(i + 1) + " cluster(s): " + str(kmeans.inertia_))
    

    fig, axs = plt.subplots(2)

    indexes = np.arange(1, 8)    
    axs[0].plot(indexes, costs_three_column)
    axs[0].set_title("Clustering Costs for Income, education, and job")
    axs[0].set_xlabel("Number of clusters")
    axs[0].set_ylabel("Cost")

    two_columns = df[['education', 'job']]
    
    scaler = MinMaxScaler()
    two_columns = scaler.fit_transform(two_columns)

    costs_two_column = []

    print("Clustering Costs for education and job") # For up to 7 clusters
    for i in range(7):
        kmeans = KMeans(n_clusters = i + 1).fit(two_columns)
        costs_two_column.append(kmeans.inertia_)
        print(str(i + 1) + " cluster(s): " + str(kmeans.inertia_))
    

    indexes = np.arange(1, 8)    
    axs[1].plot(indexes, costs_two_column)
    axs[1].set_title("Clustering Costs for education and job")
    axs[1].set_xlabel("Number of clusters")
    axs[1].set_ylabel("Cost")
    
    plt.tight_layout()
    plt.show()

    '''
    Observations

    The best number of clusters for each dataset can be seen by the elbow method in the graph.

    For the dataset with Income, education and job, the elbow method shows that the inflection point is at 2 clusters
    For the dataset with education and job, the elbow method shows that the inflection point is at 3 clusters.
    '''

task4()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:49:15 2018

@author: kirktsui
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None)


#print head and tail of data frame
print(df_wine.head())
print(df_wine.tail())
#print summary of data frame
summary = df_wine.describe()
print(summary) 
 
#PAIRWISE GRAPH
sns.pairplot(df_wine)
plt.tight_layout()
plt.show()


#HEATMAP
cm = np.corrcoef(df_wine.values.T)
sns.set(font_scale=1.2)
hm = sns.heatmap(cm, 
                 cbar=True,
                 annot=True, 
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 7},
                 yticklabels=df_wine.columns,
                 xticklabels=df_wine.columns)
plt.show()


#BOXPLOT
array = df_wine.iloc[0:177,0:13].values
plt.boxplot(array)
plt.ylabel(("Quartile Ranges")) 
plt.show()


X, y = df_wine.iloc[:,1:].values,  df_wine.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42,stratify = y)
#print( X_train.shape, y_train.shape)


# standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


#Baseline lr
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit (X_train_std, y_train)

cv_score_train_lr = np.average(cross_val_score(lr, X_train_std, y_train, cv = 5))
cv_score_test_lr = np.average(cross_val_score(lr, X_test_std, y_test, cv = 5))


#Baseline SVM
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

cv_score_train_svm = np.average(cross_val_score(svm, X_train_std, y_train, cv = 5))
cv_score_test_svm = np.average(cross_val_score(svm, X_test_std, y_test, cv = 5))

print("\t\t\t\tLogistic\tSVM")
print('Baseline_train\t\t\t%.4f\t\t%.4f'%(cv_score_train_lr,cv_score_train_svm))
print('Baseline_test\t\t\t%.4f\t\t%.4f'%(cv_score_test_lr,cv_score_test_svm))

#PCA transform
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

#PCA lr
lr.fit(X_train_pca, y_train)
cv_score_train_lr_pca = np.average(cross_val_score(lr, X_train_pca, y_train, cv = 5))
cv_score_test_lr_pca = np.average(cross_val_score(lr, X_test_pca, y_test, cv = 5))

#PCA SVM
svm.fit(X_train_pca, y_train)
cv_score_train_svm_pca = np.average(cross_val_score(svm, X_train_pca, y_train, cv = 5))
cv_score_test_svm_pca = np.average(cross_val_score(svm, X_test_pca, y_test, cv = 5))


print('PCA_train\t\t\t%.4f\t\t%.4f'%(cv_score_train_lr_pca,cv_score_train_svm_pca))
print('PCA_test\t\t\t%.4f\t\t%.4f'%(cv_score_test_lr_pca,cv_score_test_svm_pca))

#LDA transform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

#LDA lr
lr.fit(X_train_lda, y_train)
cv_score_train_lr_lda = np.average(cross_val_score(lr, X_train_lda, y_train, cv = 5))
cv_score_test_lr_lda = np.average(cross_val_score(lr, X_test_lda, y_test, cv = 5))

#LDA SVM
svm.fit(X_train_lda, y_train)
cv_score_train_svm_lda = np.average(cross_val_score(svm, X_train_lda, y_train, cv = 5))
cv_score_test_svm_lda = np.average(cross_val_score(svm, X_test_lda, y_test, cv = 5))


print('LDA_train\t\t\t%.4f\t\t%.4f'%(cv_score_train_lr_lda,cv_score_train_svm_lda))
print('LDA_test\t\t\t%.4f\t\t%.4f'%(cv_score_test_lr_lda,cv_score_test_svm_lda))


#kPCA transform
from sklearn.decomposition import KernelPCA

for gm in [0.0001, 0.001, 0.01, 0.1, 1, 2]:

    kpca = KernelPCA(n_components=2, kernel='rbf', gamma = gm)
    X_train_kpca = kpca.fit_transform(X_train_std)
    X_test_kpca = kpca.transform(X_test_std)
    
    #kPCA lr
    lr.fit(X_train_kpca, y_train)
    cv_score_train_lr_kpca = np.average(cross_val_score(lr, X_train_kpca, y_train, cv = 5))
    cv_score_test_lr_kpca = np.average(cross_val_score(lr, X_test_kpca, y_test, cv = 5))
    
    #kPCA SVM
    svm.fit(X_train_kpca, y_train)
    cv_score_train_svm_kpca = np.average(cross_val_score(svm, X_train_kpca, y_train, cv = 5))
    cv_score_test_svm_kpca = np.average(cross_val_score(svm, X_test_kpca, y_test, cv = 5))
    
    print('kPCA_train (γ=%.4f)\t\t%.4f\t\t%.4f'%(gm,cv_score_train_lr_kpca,cv_score_train_svm_kpca))
    print('kPCA_test  (γ=%.4f)\t\t%.4f\t\t%.4f'%(gm, cv_score_test_lr_kpca,cv_score_test_svm_kpca))



print("\nMy name is Jianhao Cui")
print("My NetID is: jianhao3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
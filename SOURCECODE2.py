#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Kelompok 05
"""

import pandas as pd
import seaborn as sns
import scipy


#1.4.
#Membuat Heatmap dan mencari korelasi
df = pd.read_csv('df_cleaned_9Des.csv')
df2 = df[['score','user score', 'critics', 'users']]
corr2 = df2.corr()
print(df2)
print(corr2)
print(sns.heatmap(corr2, cmap = "YlGnBu", annot=True))

#3.2.
#Mencari nilai chi square dari score dengan jumlah game yang release di setiap tahun
score = df['score']
bins = [-1,19,39,59,79,100]
groups = ['Overwhelming dislike', 'Generally unfavorable reviews', 'Mixed or average reviews', 'Generally favorable reviews', 'Universal acclaim']
scoreCat = pd.cut(score,bins,labels = groups)
df['score category'] = scoreCat

print('score')
table = pd.crosstab(df['score category'], df['year'])
print(table)

from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(table.values)
ct = scipy.stats.chi2.ppf(1-.05, df=dof)
print('critical value ', ct)
print('degree of freedom ' , dof)
print('chi square ' , chi2)

print()


#3.3.
#Mencari nilai chi square dari user score dengan jumlah game yang release di setiap tahun
userScore = df['user score']
bins = [-1,19,39,59,79,100]
groups = ['Overwhelming dislike', 'Generally unfavorable reviews', 'Mixed or average reviews', 'Generally favorable reviews', 'Universal acclaim']
userScoreCat = pd.cut(userScore,bins,labels = groups)
df['user score category'] = userScoreCat

print('user score')
table = pd.crosstab(df['user score category'], df['year'])
print(table)

from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(table.values)
ct = scipy.stats.chi2.ppf(1-.05, df=dof)
print('critical value ', ct)
print('degree of freedom ' , dof)
print('chi square ' , chi2)

print()
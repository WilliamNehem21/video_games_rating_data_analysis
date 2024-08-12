# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 09:32:28 2021

@author: ASUS
"""
#Tugas Akhir PDS Kelompok 5 dengan Topik Video Games Rating

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from matplotlib import pyplot 
import plotly.express as px
import datetime
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.tree import export_graphviz
import pydotplus

df_games = pd.read_csv('https://raw.githubusercontent.com/BrunoBVR/projectGames/main/games-data.csv')

# EKSPLORASI DATA
#Melihat atribut/kolom yang dimiliki oleh dataset
df_games.columns
#Melihat jumlah baris dan kolom yang dimiliki oleh dataset | format : (baris,kolom)
df_games.shape
#Melihat tipe data yang dimiliki setiap atribut/kolom
df_games.dtypes
#Melihat banyaknya elemen yang ada pada dataset
df_games.size
#Melihat apakah terdapat missing value pada dataset
df_games.isnull().values.any()

#Describe dataframenya
df_games.describe()

#Mencari jumlah sum missin values
df_games.isna().sum()

# Lihat Distribusi Attributes
df_games.hist()
pyplot.show()

#hapus null dari kolom players
df_games = df_games.dropna()

df_games.drop_duplicates(subset = ['name','platform','r-date','score','user score'],keep = 'first',inplace = True)

#Mencari Korelasi
import seaborn as sns
corr = df_games[df_games.columns].corr()
sns.heatmap(corr, cmap="YlGnBu", annot = True)

#Ubah r-date menjadi tahun dan bulan
def convert_year(obj):
    dto = datetime.datetime.strptime(obj, '%B %d, %Y')
    return dto.year

def convert_month(obj):
    dto = datetime.datetime.strptime(obj, '%B %d, %Y')
    return dto.month

def convert_day(obj):
    dto = datetime.datetime.strptime(obj, '%B %d, %Y')
    return dto.day

df_games['year'] = df_games['r-date'].apply(convert_year)
df_games['month'] = df_games['r-date'].apply(convert_month)
df_games['day'] = df_games['r-date'].apply(convert_day)

#Bersihin duplicate dataset df_games_cleaned

#Hitung jumlah duplicate
df_games.name.duplicated().sum()
df_games.name.duplicated().sum()

#temuin yang duplicate
df_duplicate = df_games.loc[df_games.name.duplicated(), :]

#Save data Ganes yg sudah dibersihkan

df_games.to_csv('cleaned_data.csv', index = False)

#Import data yang ud disave

df_games_cleaned = pd.read_csv('cleaned_data.csv')

#ubah kolom user score jadi numeric dan dikali 10 supaya bisa dicompare

df_games_cleaned['user score'] = df_games_cleaned['user score'].apply(lambda score: 0 if score == 'tbd' else score)
df_games_cleaned['user score'] = pd.to_numeric(df_games_cleaned['user score'])
df_games_cleaned['user score'] = df_games_cleaned['user score']*10

#mencari korelasi
df_games_cleaned_noDate = df_games_cleaned[['score','user score','critics']]
import seaborn as sns
corr = df_games_cleaned_noDate[df_games_cleaned_noDate.columns].corr()
sns.heatmap(corr, cmap="YlGnBu", annot = True)

#bikin selisih score dan user score
df_games_cleaned['score difference'] = df_games_cleaned['score'] - df_games_cleaned['user score']

#Explore genre unique
df_games_cleaned['genre'].nunique()
#Ubah ke lower case dan tambahkan ,
df_games_cleaned['genre'] = df_games_cleaned['genre'].apply(lambda s: s.lower().replace(' ',''))
#bikin kolom baru
df_games_cleaned['genre_list'] = df_games_cleaned['genre'].apply(lambda s: list(set(s.split(','))) )


#Jumlah genre unique
unique_genres = []
for i in df_games_cleaned['genre_list']:
    unique_genres += i
    
unique_genres = list(set(unique_genres))
print('Number of unique genres: ',len(unique_genres))


#info berdasarkan platform dan genre
genre_plat = {'genre':[]}

for p in df_games_cleaned['platform'].unique():
    genre_plat[p] = []

for g in unique_genres:
    genre_plat['genre'].append(g)
    
    for p in df_games_cleaned['platform'].unique():        
        g_p = []
        for i in df_games_cleaned[df_games_cleaned['platform'] == p]['genre_list']:
            g_p += i
            
        genre_plat[p].append( g_p.count(g) )
        
df_genre_final = pd.DataFrame(genre_plat)

#Save data-dataframe yang sudah diclean
df_games_cleaned.to_csv('df_games_cleaned.csv', index = False)
df_genre_final.to_csv('df_genre_final.csv', index = False)

#Predictive Analysis

#Buat features dari 3 kolom 
games_3kol_prediktor = df_games_cleaned[['critics','score','user score']]
#Buat array Numpy utk features
games_3kol_prediktor = np.array(games_3kol_prediktor.values)

#Buat label kelas dari kolom developer
games_labels = df_games_cleaned[['developer']]  # hasil: 1 kolom 
#Buat array Numpy utk kelas/label
developer_games_label = np.array(games_labels.values) # numpy array 

X = games_3kol_prediktor
Y = developer_games_label


# Split dataset into training set and test set: 70% training and 30% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3) 

from sklearn.tree import export_graphviz
import pydotplus

# Create/initiate the Decision Tree classifer model
DT_model_games = tree.DecisionTreeClassifier(criterion='entropy')
# Train Decision Tree Classifer using the 70% of the dataset
DT_model_games.fit(X_train,Y_train)

#Predict the response for test dataset
Y_pred = DT_model_games.predict(X_test)


# Evaluate model using test (30%) dataset, print the accuracy
print("Model accuracy:",metrics.accuracy_score(Y_test, Y_pred))

games_classes = games_labels.developer.unique()
print(games_classes)

# Hitung dan tampilkan metriks evaluator model klasifikasi
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, target_names = games_classes))


#======jika sudah memperoleh kriteria model yg akurasinya tinggi,
#buat model finalnya dengan menggunakan seluruh data input yg dimiliki======

DT_model_games_final = tree.DecisionTreeClassifier(criterion='entropy')
DT_model_games_final.fit(X,Y)

int_class_names=DT_model_games_final.classes_
str_class_names = int_class_names.astype(str)

#Visualize the Decision Tree model 
#visualisasi model DT iris.
#Nama2 fitur/prediktor di DT diambil dari nama atribut prediktor yg dipakai membuat model
#Begitupun dengan kelas2 targetnya

dot_data = export_graphviz(DT_model_games_final,feature_names=df_games_cleaned.columns, class_names=str_class_names, filled=True,rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
#Create and save the graph of tree as image (PNG format)
graph.write_png("Dtree_games_model.png")
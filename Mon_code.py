import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data= pd.read_csv('Tobacco3482.csv')
sns.countplot(data=data,y='label') 
plt.show()
classes = pd.read_csv('Tobacco3482.csv', sep = ",")
classes.sample(10)

print("Les étiquettes manquantes :", 1.0 - classes.shape[0] / classes.dropna().shape[0])
s = 0
for i in range(classes.shape[0]):
    s += classes["img_path"][i].split("/")[0] == classes["label"][i]
print("Les documents mal classé :", classes.shape[0] - s)

data_txt=[]
data_txt=data

NB = data_txt.shape[0]
for i in range (NB):
    A = data_txt.get_value(i, 'img_path')
    data_txt.set_value(i, 'img_path', 'Tobacco3482-OCR/'+A)
    data_txt.set_value(i, 'img_path', data_txt.get_value(i, 'img_path').split('.jpg')[0]+'.txt')
    data_txt.set_value(i, 'img_path',open(data_txt.get_value(i, 'img_path'), "r",encoding="utf8").read())
data_txt.columns = ['text','label']
print(data_txt.head())

X_train, X_app, y_train, y_app = train_test_split(data['text'], data['label'], test_size=0.4)
X_test,X_dev,y_test,y_dev=train_test_split(X_app, y_app, test_size=0.5)

print('train data size=',X_train.shape[0],'X_test data size=',X_test.shape[0],'dev data size=',X_dev.shape[0])

vectorizer = CountVectorizer(max_features=2000)
vectorizer.fit(X_train)
X_train_counts = vectorizer.transform(X_train)
X_dev_counts = vectorizer.transform(X_dev)
X_test_counts = vectorizer.transform(X_test)  

parameters = {'alpha' : [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0]}

nb_classifier = MultinomialNB()
grid_search_clf = GridSearchCV(nb_classifier, parameters, cv=5, return_train_score=True)
grid_search_clf.fit(X_train_counts, y_train)
res = grid_search_clf.cv_results_

for i in range(1,11):
    print('Rang', i)
    ind = np.where(res['rank_test_score'] == i)
    print('Alpha : {}'.format(res['params'][ind[0][0]]['alpha']))
    print('La précision moyenne : {}\n'.format(round(res['mean_test_score'][ind][0], 3)))



nb_classifier = MultinomialNB(alpha=1.)
nb_classifier.fit(X_train_counts, y_train)

pred_train = nb_classifier.predict(X_train_counts)
pred_dev = nb_classifier.predict(X_dev_counts)
pred_test = nb_classifier.predict(X_test_counts)

print("La précision sur les données d'entrainement est : ", metrics.accuracy_score(y_train, pred_train))
print("La précision sur les données de validation est : ", metrics.accuracy_score(y_dev, pred_dev))
print("La précision sur les données de test est : ", metrics.accuracy_score(y_test, pred_test))



tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_dev_tf = tf_transformer.transform(X_dev_counts)
X_test_tf = tf_transformer.transform(X_test_counts)


nb_classifier.fit(X_train_tf, y_train)

pred_train_tf = nb_classifier.predict(X_train_tf)
pred_dev_tf = nb_classifier.predict(X_dev_tf)
pred_test_tf = nb_classifier.predict(X_test_tf)

print("La précision sur les données d'entrainement est: ", metrics.accuracy_score(y_train, pred_train_tf))
print("La précision sur les données de validation est : ", metrics.accuracy_score(y_dev, pred_dev_tf))
print("La précision sur les données de test est : ", metrics.accuracy_score(y_test, pred_test_tf))


mlp_clf = MLPClassifier(activation='relu', alpha=1.0, verbose=2, batch_size=50)
mlp_clf.fit(X_train_counts, y_train)

pred_train_mlp = mlp_clf.predict(X_train_counts)
pred_dev_mlp = mlp_clf.predict(X_dev_counts)
pred_test_mlp = mlp_clf.predict(X_test_counts)

print("La précision sur les données d'entrainement est : ", metrics.accuracy_score(y_train, pred_train_mlp))
print("La précision sur les données de validation est : ", metrics.accuracy_score(y_dev, pred_dev_mlp))
print("La précision sur les données de test est : ", metrics.accuracy_score(y_test, pred_test_mlp))


print('------------------  Naive Bayse Bag Of  Word  ------------------\n')
print("La précision sur les données d'entrainement est : ", metrics.accuracy_score(y_train, pred_train))
print("La précision sur les données de validation est : ", metrics.accuracy_score(y_dev, pred_dev))
print("La précision sur les données de test est : ", metrics.accuracy_score(y_test, pred_test))
print('\n\n Matrice de Classification \n')
print(classification_report(y_test, pred_test))
print('Matrice de Confusion Bag Of Word')
print(confusion_matrix(y_test, pred_test))



print('---------------------- TF-IDF -----------------------\n')
print("La précision sur les données d'entrainement es: ", metrics.accuracy_score(y_train, pred_train_tf))
print("La précision sur les données de validation est : ", metrics.accuracy_score(y_dev, pred_dev_tf))
print("La précision sur les données de test est : ", metrics.accuracy_score(y_test, pred_test_tf))
print('\n\n Matrice de Classification \n')
print(classification_report(y_test, pred_test_tf))
print()
print('Matrice de Confusion TF-IDF')
print(confusion_matrix(y_test, pred_test_tf))


print('---------------------- MLP Classifier with TF-IDF representation -----------------------\n')

print("La précision sur les données d'entrainement est : ", metrics.accuracy_score(y_train, pred_train_mlp))
print("La précision sur les données de validation est : ", metrics.accuracy_score(y_dev, pred_dev_mlp))
print("La précision sur les données de test est : ", metrics.accuracy_score(y_test, pred_test_mlp))
print('\n\n Matrice de Classification \n')
print(classification_report(y_test, pred_test_mlp))
print()
print('Matrice de Confusion MLP-BoW')
print(confusion_matrix(y_test, pred_test_mlp))
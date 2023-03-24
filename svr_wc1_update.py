reset()
y
import numpy as np
import pandas as pd
from nilearn.image import get_data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error  

all_filenames = pd.read_csv(
        #"Agewell_IMAP_data.csv",
         "/netapp/vol4_agewell/pro/AGEWELL/imap_sh/Doctorat/python/Algo_test_BT/training_ADNI_IMAP_wc1.csv",
        dtype=object,
        keep_default_na=False,
        na_values=[]).to_numpy()
        
new_filenames = pd.read_csv(
        #"Agewell_IMAP_data.csv",
         "/netapp/vol4_agewell/pro/AGEWELL/imap_sh/Doctorat/python/Algo_test_BT/sujets_age_training_wc1.csv",
        dtype=object,
        keep_default_na=False,
        na_values=[]).to_numpy()

images = []
labels = []
listNameFile = []
groupe=[]

for f in all_filenames:
    rowBdd = f[0].split(";")
    subject_id = rowBdd[0]
    pathIRM = rowBdd[0]
    t1 = pathIRM
    images.append(t1)
    age = rowBdd[2]
    labels.append(float(age))
    groupe.append(rowBdd[3])

new_images = []
new_labels = []
new_listNameFile = []

for f in new_filenames:
    new_rowBdd = f[0].split(";")
    new_subject_id = new_rowBdd[0]
    new_pathIRM = new_rowBdd[0]
    new_t1 = new_pathIRM
    new_images.append(new_t1)
    new_age = new_rowBdd[2]
    new_labels.append(float(new_age)) 
    
age = np.array(labels)
gm_imgs = np.array(images)
groupe = np.array(groupe)    
new_age = np.array(new_labels)
new_gm_imgs = np.array(new_images)

from nilearn.maskers import NiftiMasker
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

nifti_masker = NiftiMasker(standardize=False, smoothing_fwhm=12, memory='nilearn_cache')

gm_maps_masked = nifti_masker.fit_transform(gm_imgs)
new_gm_maps_masked = nifti_masker.transform(new_gm_imgs)

gss = GroupShuffleSplit(n_splits=20, train_size=.8, random_state=42) 
gss.get_n_splits()

clf = Pipeline(
    [
        ("reg", SelectPercentile(f_regression)),
        ("scaler", StandardScaler()),
        ("svr", SVR(gamma="auto")),
    ]
)


from sklearn.model_selection import cross_val_score

score_means = list()
score_stds = list()
percentiles = (1,)#3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

#CROSS VAL pour PERCENTILE
#for percentile in percentiles:
#    clf.set_params(reg__percentile=float(percentile))
#    this_scores = cross_val_score(clf, new_gm_maps_masked, age, cv=gss, groups=groupe, scoring='neg_mean_absolute_error')
#    score_means.append(this_scores.mean())
#    score_stds.append(this_scores.std())

#CROSS VALIDATION pour estimer les hyper paramètres CAS 1 : On a 1 cohorte dans ce cas, le split est un hyper paramètre, sinon estimer les autres genres smooth, pct features, kernel
#partie pour CV les kernels, reste à changer scoring
from sklearn.model_selection import GridSearchCV
parameters = {'svr__kernel':('linear', 'rbf','poly'), 'svr__degree':(2,3,4,5,6,7)}  #à ajouter qd poly -->besoin du degré
gscv = GridSearchCV(clf, parameters, cv=gss, verbose=10, scoring='neg_mean_absolute_error') #GridSeaarchCV va appliquer la CV en étages, CV sur un paramètre puis sur l'autre, ex : d'abord sur kernel linéaire puis rbf, verbose=a check
gscv.fit(gm_maps_masked, age, groups=groupe)
print("Best parameter (CV score=%0.3f):" % gscv.best_score_)  #ligne pour savoir quel kernel est le meilleur
print(gscv.best_params_)

#re pipeline modifié selon résultat CV précédente
clf = Pipeline(
    [
        ("reg", SelectPercentile(f_regression)),
        ("scaler", StandardScaler()),
        ("svr", SVR(gamma="auto", kernel=gscv.best_params_["svr__kernel"],degree=gscv.best_params_["svr__degree"])),
    ]
)

for percentile in percentiles:
    clf.set_params(reg__percentile=float(percentile))
    clf.fit(gm_maps_masked, age)
    pred=clf.predict(new_gm_maps_masked)
  
mse=mean_squared_error(new_age,pred)
mae=mean_absolute_error(new_age, pred)
print(mae)

import matplotlib.pyplot as plt
#fig = plt.figure() #pour faire une nouvelle figure et éviter superposition des résultats 
#fig=plt.errorbar(l_smooth, score_means, np.array(score_stds))
#fig=plt.title("Performance of the SVM-Anova varying the percentile of features selected")
#fig=plt.xticks(np.linspace(0, 100, 11, endpoint=True))
#fig=plt.xlabel("smooth")
#fig=plt.ylabel("Accuracy Score")
#fig=plt.axis("tight")
#fig=plt.show()
#plt.savefig("/netapp/vol4_agewell/pro/AGEWELL/imap_sh/Doctorat/python/Algo_test_BT/test_alt_4.png")


import csv
objetFichier = open('BT_pred_rbf.csv', 'w')
writer = csv.writer(objetFichier)
writer.writerow(pred)
writer.writerow(new_age)
objetFichier.close()
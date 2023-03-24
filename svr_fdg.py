reset()
y
import numpy as np
import pandas as pd
from nilearn.image import get_data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error  

all_filenames = pd.read_csv(
        #"Agewell_IMAP_data.csv",
         "/netapp/vol4_agewell/pro/AGEWELL/imap_sh/Doctorat/python/Algo_test_BT/training_ADNI_IMAP_tep.csv",
        dtype=object,
        keep_default_na=False,
        na_values=[]).to_numpy()

new_filenames = pd.read_csv(
        #"Agewell_IMAP_data.csv",
         "/netapp/vol4_agewell/pro/AGEWELL/imap_sh/Doctorat/python/Algo_test_BT/sujets_age_training_tep.csv",
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
    pathfdg = rowBdd[0]
    fdg = pathfdg
    images.append(fdg)
    age = rowBdd[2]
    labels.append(float(age))
    groupe.append(rowBdd[3])    

new_images = []
new_labels = []
new_listNameFile = []

for f in new_filenames:
    new_rowBdd = f[0].split(";")
    new_subject_id = new_rowBdd[0]
    new_pathfdg = new_rowBdd[0]
    new_fdg = new_pathfdg
    new_images.append(new_fdg)
    new_age = new_rowBdd[2]
    new_labels.append(float(new_age)) 
    
age = np.array(labels)
fdg_imgs = np.array(images)
groupe = np.array(groupe)
new_age = np.array(new_labels)
new_fdg_imgs = np.array(new_images)

from nilearn.maskers import NiftiMasker
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

nifti_masker = NiftiMasker(standardize=False, smoothing_fwhm=8, memory='nilearn_cache')

fdg_maps_masked = nifti_masker.fit_transform(fdg_imgs)
new_fdg_maps_masked = nifti_masker.transform(new_fdg_imgs)

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
#l_smooth = (2,3,4,5,6,7,8)

#for smooth in l_smooth:
#nifti_masker = NiftiMasker(standardize=False,smoothing_fwhm=float(smooth),memory='nilearn_cache')
#fdg_maps_masked = nifti_masker.fit_transform(fdg_imgs)
#for percentile in percentiles:
#    clf.set_params(reg__percentile=float(percentile))
#    this_scores = cross_val_score(clf, fdg_maps_masked, age, cv=gss, groups=groupe, scoring='neg_mean_absolute_error')
#    score_means.append(this_scores.mean())
#    score_stds.append(this_scores.std())

#CROSS VALIDATION pour estimer les hyper paramètres CAS 1 : On a 1 cohorte dans ce cas, le split est un hyper paramètre, sinon estimer les autres genres smooth, pct features, kernel
#partie pour CV les kernels, reste à changer scoring
from sklearn.model_selection import GridSearchCV
parameters = {'svr__kernel':('linear', 'rbf','poly'), 'svr__degree':(2,3,4,5,6,7)}  #à ajouter qd poly -->besoin du degré
gscv = GridSearchCV(clf, parameters, cv=gss, verbose=10, scoring='neg_mean_absolute_error') #GridSeaarchCV va appliquer la CV en étages, CV sur un paramètre puis sur l'autre, ex : d'abord sur kernel linéaire puis rbf, verbose=a check
gscv.fit(fdg_maps_masked, age, groups=groupe)
print("Best parameter (CV score=%0.3f):" % gscv.best_score_) #ligne pour savoir quel kernel est le meilleur
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
    clf.fit(fdg_maps_masked, age)
    pred=clf.predict(new_fdg_maps_masked)

mse=mean_squared_error(new_age,pred)
mae=mean_absolute_error(new_age, pred)   
print(mae)
 
import matplotlib.pyplot as plt
#fig = plt.figure()       
#plt.errorbar(l_smooth, score_means, np.array(score_stds))
#plt.title("Performance of the SVM-Anova varying the percentile of features selected")
#plt.xticks(np.linspace(0, 100, 11, endpoint=True))
#plt.xlabel("smooth")
#plt.ylabel("Accuracy Score")
#plt.axis("tight")
#plt.show()
#plt.savefig("/netapp/vol4_agewell/pro/AGEWELL/imap_sh/Doctorat/python/Algo_test_BT/test_alt_6.png")

import csv
objetFichier = open('BT_pred_fdg_2.csv', 'w')
writer = csv.writer(objetFichier)
writer.writerow(pred)
writer.writerow(new_age)
objetFichier.close()
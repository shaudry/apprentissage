import numpy as np
import pandas as pd
from nilearn.image import get_data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 

from nilearn.maskers import NiftiMasker
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import csv


all_filenames = pd.read_csv(
        #"Agewell_IMAP_data.csv",
         "/netapp/vol4_agewell/pro/AGEWELL/imap_sh/Doctorat/python/Algo_test_BT/training_ADNI_IMAP_wc1.csv", #fdg ou multi à la place de wc1
        dtype=object,
        keep_default_na=False,
        na_values=[]).to_numpy() # , à la place de ) quand ligne suivante
        #sep=";") #ligne pour éviter à avoir à faire le bloc suivant à tester voir si déjà variable avec ça
        
new_filenames = pd.read_csv(
        #"Agewell_IMAP_data.csv",
         "/netapp/vol4_agewell/pro/AGEWELL/imap_sh/Doctorat/python/Algo_test_BT/sujets_age_training_wc1.csv",  #fdg ou multi à la place de wc1
        dtype=object,
        keep_default_na=False,
        na_values=[]).to_numpy()
        #sep=";")
        
#all_filenames['bdd'] #selectionne une colonnes

images = []
labels = []
listNameFile = []
groupe=[]

for f in all_filenames:
    rowBdd = f[0].split(";")
    subject_id = rowBdd[0]
    path_img = rowBdd[0]
    img = path_img
    images.append(img)
    age = rowBdd[2]
    labels.append(float(age))
    groupe.append(rowBdd[3])

new_images = []
new_labels = []
new_listNameFile = []

for f in new_filenames:
    new_rowBdd = f[0].split(";")
    new_subject_id = new_rowBdd[0]
    new_path_img = new_rowBdd[0]
    new_img = new_path_img
    new_images.append(new_img)
    new_age = new_rowBdd[2]
    new_labels.append(float(new_age)) 
    
age = np.array(labels)
modal_imgs = np.array(images)
groupe = np.array(groupe)    
new_age = np.array(new_labels)
new_modal_imgs = np.array(new_images)

nifti_masker = NiftiMasker(standardize=False, smoothing_fwhm=12, memory='nilearn_cache')

modal_maps_masked = nifti_masker.fit_transform(modal_imgs)
new_modal_maps_masked = nifti_masker.transform(new_modal_imgs)

gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=42) 
gss.get_n_splits()

clf = Pipeline(
    [
        ("reg", SelectPercentile(f_regression)),
        ("scaler", StandardScaler()),
        ("RF", RandomForestRegressor(random_state=0)),
    ]
)



from sklearn.model_selection import GridSearchCV
parameters = {'RF__n_estimators':(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000),'RF__max_depth':(5, 10, 20), 'RF__min_samples_leaf':(10,15,20), 'RF__max_leaf_nodes':(64,32,16,8)}#, 'svr__degree':('2','3')}  #à ajouter qd poly -->besoin du degré
gscv = GridSearchCV(clf, parameters, cv=gss, verbose=10, scoring='neg_mean_absolute_error') #GridSeaarchCV va appliquer la CV en étages, CV sur un paramètre puis sur l'autre, ex : d'abord sur kernel linéaire puis rbf, verbose=a check
gscv.fit(modal_maps_masked, age, groups=groupe)
print("Best parameter (CV score=%0.3f):" % gscv.best_score_)  #ligne pour savoir quel kernel est le meilleur
print(gscv.best_params_)


clf = Pipeline(
    [
        ("reg", SelectPercentile(f_regression)),
        ("scaler", StandardScaler()),
        ("RF", RandomForestRegressor(random_state=0, n_estimators=gscv.best_params_["RF__n_estimators"], max_depth=gscv.best_params_["RF__max_depth"], min_samples_leaf=gscv.best_params_["RF__min_samples_leaf"],max_leaf_nodes=gscv.best_params_["RF__max_leaf_nodes"])),

percentiles = (1,)
for percentile in percentiles:
    clf.set_params(reg__percentile=float(percentile))
    clf.fit(modal_maps_masked, age)
    pred=clf.predict(new_modal_maps_masked)
  
mse=mean_squared_error(new_age,pred)
mae=mean_absolute_error(new_age, pred)
print(mae)

  
#regr = RandomForestRegressor(max_depth=2, random_state=0, verbose=10)
#regr.fit(modal_maps_masked, age)
#pred_rf = regr.predict(new_modal_maps_masked)

#mse=mean_squared_error(new_age, pred)
#mae=mean_absolute_error(new_age, pred)
#print(mae)

#print(regr.predict([[0, 0, 0, 0]]))

from sklearn.ensemble import GradientBoostingRegressor

#reg = GradientBoostingRegressor(random_state=0, verbose=10)
#reg.fit(modal_maps_masked, age)
#reg.predict(new_modal_maps_masked)

#decoder=DecoderRegressor(estimator='svr', mask=mask,scoring='neg_mean_absolute_error',screening_percentile=1,n_jobs=1)
#decoder.fit(modal_maps_masked, age)
#pred = decoder.predict(new_modal_maps_masked)

#decoder=DecoderRegressor(estimator='ridge', mask=mask,scoring='neg_mean_absolute_error',screening_percentile=1,n_jobs=1)
#decoder.fit(modal_maps_masked, age)
#pred = decoder.predict(new_modal_maps_masked)

#clf = Ridge(alpha=1.0)
#clf.fit(modal_maps_masked, age)
#clf.predict(new_modal_maps_masked

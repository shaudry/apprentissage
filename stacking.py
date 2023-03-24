from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from nilearn.maskers import NiftiMasker
from sklearn.utils import check_random_state

import numpy as np
import pandas as pd
from nilearn.image import get_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupShuffleSplit

all_filenames = pd.read_csv(
        #"Agewell_IMAP_data.csv",
         "/netapp/vol4_agewell/pro/AGEWELL/imap_sh/Doctorat/python/Brain_age_prediction-main/training_ADNI_IMAP_multi.csv",
        dtype=object,
        keep_default_na=False,
        na_values=[]).to_numpy()

images = []
images_fdg = []
labels = []
listNameFile = []
groupe=[]
for f in all_filenames:
    rowBdd = f[0].split(";")
    subject_id = rowBdd[0]
    pathIRM = rowBdd[0]
    pathfdg = rowBdd[1]
    t1 = pathIRM
    fdg = pathfdg
    images.append(t1)
    images_fdg.append(fdg)
    age = rowBdd[3]
    labels.append(float(age))
    groupe.append(rowBdd[4])
    

age = np.array(labels)
gm_imgs = np.array(images)
fdg_imgs = np.array(images_fdg)
groupe = np.array(groupe)  

nifti_masker = NiftiMasker(standardize=False, smoothing_fwhm=12, memory='nilearn_cache')
gm_maps_masked = nifti_masker.fit_transform(gm_imgs)

nifti_masker = NiftiMasker(standardize=False,smoothing_fwhm=8,memory='nilearn_cache')
fdg_maps_masked = nifti_masker.fit_transform(fdg_imgs)

#test = pd.DataFrame({'t1' : gm_maps_masked,'fdg' : fdg_maps_masked}) #erreur ?
test = pd.DataFrame()
test['gm_maps_masked'] = gm_maps_masked.tolist()
test['fdg_maps_masked'] = fdg_maps_masked.tolist()
#test = pd.DataFrame([gm_maps_masked,fdg_maps_masked])
X=test.to_numpy()

#rng = check_random_state(42)
#X_train, X_test, age_train, age_test = train_test_split(X, age, train_size=.8, random_state=rng)

#gss = GroupShuffleSplit(n_splits=20, train_size=.8, random_state=42)
#gss.get_n_splits()

estimators = [
     ('lr', RidgeCV()),
     ('svr', LinearSVR(random_state=42))
 ]

reg = StackingRegressor(
     estimators=estimators,
     final_estimator=RandomForestRegressor(n_estimators=2,
                                           random_state=42)
 ) 

reg.fit(gm_maps_masked, age)
reg.fit(gm_maps_masked, age).score(X_test, age_test)
#pred=reg.fit(gm_maps_masked, age).predict(new_gm_maps_masked)
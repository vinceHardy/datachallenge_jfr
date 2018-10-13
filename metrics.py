#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:20:03 2018

@author: theoestienne
"""
import pandas as pd
from sklearn import metrics
import numpy as np
import SimpleITK as sitk
import os

#%%

def foie_metrics(reference_path, prediction_path):

    pred = pd.read_csv(prediction_path, index_col=0)
    ref = pd.read_csv(reference_path, index_col=0)
    
    assert pred.shape[0] == ref.shape[0]
    assert pred.shape[1] == 8
    
    pred = pred.loc[ref.index]
    pred = pred.fillna(0.5)
    
    # Lesion
    y_true = ref['Lesion']
    y_pred = pred['Lesion']
    
    auc_detection = metrics.roc_auc_score(y_true, y_pred)
    
    # Malin
    y_true = ref.loc[ref['Lesion'] == True,'Malin']
    y_pred = pred.loc[ref['Lesion'] == True,'Malin']
    
    auc_malin = metrics.roc_auc_score(y_true, y_pred)
    
    
    lesions = ['Kyste', 'HNF', 'Angiome', 'Metastase', 'CHC']
    
    auc_lesions = {}
    
    for lesion in lesions:
        try:
            y_true = ref.loc[ref['Lesion'] == True,'Type de lesion'] == lesion
            y_pred = pred.loc[ref['Lesion'] == True,lesion]
            
            auc = metrics.roc_auc_score(y_true, y_pred)
            
            auc_lesions[lesion] = auc
        except Exception as e:
            print(e)
            print(lesion)
    
    auc_lesion = np.mean(list(auc_lesions.values()))
    total_score = 0.5 * auc_detection + 0.3 * auc_malin + 0.2 * auc_lesion
    
    return total_score


# print(foie_metrics(train_csv_path, test_csv_path))


#%%
def menisque_metrics(reference_path, prediction_path):
    
    pred = pd.read_csv(prediction_path, index_col=0)
    ref = pd.read_csv(reference_path, index_col=0)
    
    assert pred.shape[0] == ref.shape[0]
    assert pred.shape[1] == 5
    
    pred = pred.loc[ref.index]
    pred = pred.fillna(0.5)
    
    # Fissure
    ref['Fissure'] = np.logical_or(ref['Corne anterieure'], ref['Corne posterieure'])
    
    y_true= ref['Fissure']
    y_pred = pred['Fissure']
    
    auc_detection = metrics.roc_auc_score(y_true, y_pred)
    
    # Localisation
    y_true = ref.loc[ref['Fissure'] == True,'Corne anterieure']
    y_pred = pred.loc[ref['Fissure'] == True,'Corne anterieure']
    
    auc_position = metrics.roc_auc_score(y_true, y_pred)
    
    # Orientation
    ante = ref.loc[ref['Fissure'] == True,'Orientation anterieure'] == 'Horizontale' 
    poste = ref.loc[ref['Fissure'] == True,'Orientation posterieure'] == 'Horizontale'
    
    y_true = np.logical_or(ante,poste) 
    y_pred = pred.loc[ref['Fissure'] == True,'Orientation horizontale']
    
    auc_orientation = metrics.roc_auc_score(y_true, y_pred)
    
    total_score = 0.4 * auc_detection + 0.3 * auc_position + 0.3 *auc_orientation
    
    return total_score

# reference_path = '/home/theoestienne/Documents/JFR/menisque/menisque_train_set.csv'
# prediction_path = '/home/theoestienne/Documents/JFR/menisque/menisque_exemple.csv' 

# print(menisque_metrics(reference_path, prediction_path))

#%%
def dice_calcul(pred, ref):
    
    pred = pred.flatten()
    ref = ref.flatten()
    
    numerateur = np.sum(np.logical_and(pred, ref))
    denominateur = np.sum(pred) + np.sum(ref)
    
    dice = 2 * numerateur / denominateur
    
    return dice
 

def cortex_metrics(reference_folder, prediction_folder):
    
    patients = [patient for patient in os.listdir(reference_folder)
                if patient.startswith('mask')]
    
    dices = []
    for patient in patients:
        prediction_path = prediction_folder + patient
        reference_path = reference_folder + patient 
        
        try:
            pred = sitk.GetArrayFromImage(sitk.ReadImage(prediction_path))
            ref = sitk.GetArrayFromImage(sitk.ReadImage(reference_path))
            
            assert np.array_equal(pred, pred.astype(bool))
            
            dices.append(dice_calcul(pred,ref))
        except:
            dices.append(0)
        
    total_score = np.mean(dices)
    
    return total_score

#%%

def sein_metrics(reference_path, prediction_path):
    
    pred = pd.read_csv(prediction_path, index_col=0)
    ref = pd.read_csv(reference_path, index_col=0)
    
    assert pred.shape[0] == ref.shape[0]
    assert pred.shape[1] == 18
    
    pred = pred.loc[ref.index]
    pred = pred.fillna(0.5)
    
    # Malin
    y_true= ref['Malin']
    y_pred = pred['Malin']
    
    auc_malin = metrics.roc_auc_score(y_true, y_pred)
    
    # Tissu glandulaire 
    y_true = ref['Type de lesion'] == 'Tissu glandulaire'
    y_pred = pred['Tissu glandulaire']
    
    auc_tissu = metrics.roc_auc_score(y_true, y_pred)
        
    # Carcinome canalaire infiltrant
    y_true = ref['Type de lesion'] == 'Carcinome canalaire infiltrant'
    y_pred = pred['Carcinome canalaire infiltrant']
    
    auc_carcinome = metrics.roc_auc_score(y_true, y_pred)
    
    # Autres Benins
    autres_benins = ['Adenose sclerosante', 'Autre lesion proliferante',
    'Cicatrice radiaire', 'Fibroadenome','Galactophorite', 
    'Hyperplasie canalaire sans atypie', 'Kyste', 'PASH',
    'Papillome', 'cytosteatonecrose',
    'ganglio intra-mammaire']
    
    y_true = ref['Type de lesion'].isin(autres_benins)
    y_pred = pred[autres_benins].sum(axis=1)
    
    auc_autre_benins = metrics.roc_auc_score(y_true, y_pred)
    
    # Autres malins
    autres_malins = ['Carcinome lobulaire infiltrant', 'Cancer triple negatif', 
                      'Carcinome intracanalaire', 'Carcinome mucineux']
    
    y_true = ref['Type de lesion'].isin(autres_malins)
    y_pred = pred[autres_malins].sum(axis=1)
    
    auc_autres_malins = metrics.roc_auc_score(y_true, y_pred)
    
    total_score = 0.6 * auc_malin + 0.4/4 * (auc_tissu + auc_carcinome + auc_autre_benins + auc_autres_malins)
    
    return total_score

#%%
def thyroide_metrics(reference_path, prediction_path):
    
    pred = pd.read_csv(prediction_path, index_col=0)
    ref = pd.read_csv(reference_path, index_col=0)
    
    assert pred.shape[0] == ref.shape[0]
    assert pred.shape[1] == 1
    
    
    pred = pred.loc[ref.index]
    pred = pred.fillna(0.5)
    
    y_true = ref['anormale']
    y_pred = pred['anormale']
    
    auc = metrics.roc_auc_score(y_true, y_pred)
    
    return auc
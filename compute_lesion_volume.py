import os 
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
#os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install platipy dicom2nifti rt_utils")
from get_all_terminal_subfolders import get_all_terminal_subfolders
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
from skimage import morphology, measure  
import nibabel as nib 
from copy import deepcopy
import glob 
from nifti_roi_tools import determine_if_rt_should_be_included, determine_if_rt_is_equivocal
from pydicom import dcmread
import pandas as pd 
from copy import deepcopy

def compute_MTV(dicom_series_path, rtstruct_path, nifti_img_path):
    nii = nib.load(nifti_img_path)
    PET_volume = nii.get_fdata().copy()
    voxel_size = nii.header.get_zooms() # mm
    voxel_size = [ii/10 for ii in voxel_size] # mm -> cm 
    PET_volume = np.transpose(PET_volume, [1,0,2])
    rtstruct = RTStructBuilder.create_from(
    dicom_series_path=dicom_series_path, 
    rt_struct_path=rtstruct_path)
  
    # View all of the ROI names from within the image
    roi_name_list = rtstruct.get_roi_names()
    ignore_strings = ['Struct_XD', 'Struct_YD', 'Reference', 'Burden', 'DS1', 'marrow', 'osseous', 'marow', 'bone', 'marro']
    #print(roi_name_list)
    total_lesion_volume = 0
    # Loading the 3D Mask from within the RT Struct
    for roi_name in roi_name_list:
        ignore_rt = False 
        for ignore in ignore_strings:
            if ignore.lower() in ('struct_'+roi_name).lower():
                ignore_rt = True # remove ignored strings 
        if not determine_if_rt_should_be_included(roi_name):
            ignore_rt = True # remove XD (without label being changed), YD 

        is_equivocal = determine_if_rt_is_equivocal(roi_name)

        if ignore_rt or is_equivocal:
            continue 
        
        mask_3d = rtstruct.get_roi_mask_by_name(roi_name) > 0 # phycisian's annotations 
        total_lesion_volume += np.sum(mask_3d) * np.prod(voxel_size) # cm3 or ml 
    return total_lesion_volume

def generate_dataframe(filenames, volumes, hodgkin, num_fold):
    np.random.seed(1)
    df_ori = pd.DataFrame()
    df_ori['PatientID'] = filenames
    df_ori['Total_lesion_volume_ml'] = volumes
    df_ori['Hodgkin_or_not'] = hodgkin
    
    df_sort = deepcopy(df_ori)
    sort_index = np.argsort(volumes) 
    df_sort = df_sort.loc[sort_index]
    df_sort = df_sort.reset_index(drop=True)

    df_shuffle = deepcopy(df_ori)
    greater_0_index = np.where(volumes > 0)[0]
    df_shuffle = df_shuffle.loc[greater_0_index]
    df_shuffle = df_shuffle.reset_index(drop=True)
    sort_index = np.argsort(df_shuffle['Total_lesion_volume_ml'].to_list()) 
    sort_index = np.array_split(sort_index, num_fold)
    sort_index = [np.random.permutation(i) for i in sort_index] 
    sort_index_num = [len(i) for i in sort_index] 
    sort_index_new = [] 
    for index in range(np.min(sort_index_num)): 
        for sort_index_ in sort_index: 
            sort_index_new.append(sort_index_[index]) 

    for partial_sort_index in sort_index: 
        try:
            sort_index_new.append(partial_sort_index[np.min(sort_index_num)]) 
        except:
            continue 
    df_shuffle = df_shuffle.loc[np.array(sort_index_new)]
    df_shuffle = df_shuffle.reset_index(drop=True)
    return df_ori, df_sort, df_shuffle


def process_cases(root_dir, nifti_path, num_fold = 5):
    data = pd.read_excel('./data/UW_data_with_Hodgkin_DLBCL_labels.xlsx') 
    filename = data.Filenames.tolist()
    Disease = data.Disease.tolist() 

    folder_list = get_all_terminal_subfolders(root_dir)
    cases_list = np.arange(4540, 4800)
    cases_list = [ii for ii in cases_list if np.mod(ii, 2)==0] # only process baseline right now 
    cases_list = [str(ii) for ii in cases_list]
    subfolders_PTs = [subfolder for subfolder in folder_list if ("_pt_" in subfolder.lower())]
    subfolders_RTs = [subfolder for subfolder in folder_list if ("scho" in subfolder.lower())]
    nifti_folders = get_all_terminal_subfolders(nifti_path)
    
    file_hodgkins = []
    volume_hodgkins = []
    file_dlbcl = []
    volume_dlbcl = []

    for case in tqdm(cases_list):
        #print('Processing Case# {}'.format(case))
        subfolders_PT = [subfolder for subfolder in subfolders_PTs if subfolder.find('PETLYMPH.{}_'.format(case))>-1]
        subfolders_RT = [subfolder for subfolder in subfolders_RTs if subfolder.find('PETLYMPH.{}_'.format(case))>-1]
        nifti_folder =  [subfolder for subfolder in nifti_folders if subfolder.find('petlymph_{}'.format(case))>-1]
        if len(subfolders_PT) >= 1:
            dicom_series_path = subfolders_PT[0]
        else:
            continue 

        nifti_file_list = glob.glob(os.path.join(nifti_folder[0], "*.nii.gz"))
        nifti_file_list = [nifti_file for nifti_file in nifti_file_list if '_suv.nii.gz' in nifti_file.lower()]
        nifti_img_path = nifti_file_list[0]
        if len(subfolders_RT) >= 1:
            for subfolder_RT in subfolders_RT:
                if "adj" in subfolder_RT.lower():
                    rtstruct_folder = subfolder_RT
                else:
                    rtstruct_folder = subfolder_RT
            rtstruct_path_list = glob.glob(os.path.join(rtstruct_folder, "*.dcm"))
            if len(rtstruct_path_list) == 0:
                continue 
            else:
                 rtstruct_path = rtstruct_path_list[0]
        else: 
            continue
        
        if 'PETLYMPH_{}'.format(case) in filename:
            index = ([filename.index(i) for i in filename if(i == 'PETLYMPH_{}'.format(case))])
            if Disease[index[0]] == 'Hodgkins':
                try:
                    total_volume = compute_MTV(dicom_series_path, rtstruct_path, nifti_img_path)
                    #print(total_volume)
                except:
                    total_volume = -1 
                file_hodgkins.append('PETLYMPH_{}'.format(case))
                volume_hodgkins.append(total_volume)
            elif Disease[index[0]] == 'DLBCL':
                try:
                    total_volume = compute_MTV(dicom_series_path, rtstruct_path, nifti_img_path)
                except:
                    total_volume = -1 
                file_dlbcl.append('PETLYMPH_{}'.format(case))
                volume_dlbcl.append(total_volume)
    
    df_ori_ho, df_sort_ho, df_shuffle_ho = generate_dataframe(file_hodgkins, np.array(volume_hodgkins), [1]*len(file_hodgkins), num_fold)
    df_ori_dl, df_sort_dl, df_shuffle_dl = generate_dataframe(file_dlbcl, np.array(volume_dlbcl), [0]*len(file_dlbcl), num_fold)
    df_ori = pd.concat([df_ori_ho, df_ori_dl]) 
    df_sort = pd.concat([df_sort_ho, df_sort_dl]) 
    df_ori.to_excel('././data/UW_data_with_lesion_volume_ori.xlsx', index=False, header=True)
    df_sort.to_excel('././data/UW_data_with_lesion_volume_sort.xlsx', index=False, header=True)
    df_shuffle = pd.concat([df_shuffle_ho, df_shuffle_dl])
    df_shuffle = df_shuffle.reset_index(drop=True)
    a1 = np.arange(len(df_shuffle_ho))
    a2 = np.arange(len(df_shuffle_dl)) + len(df_shuffle_ho)

    if len(a1) == 0 and len(a2) > 0:
        new_index = a2 
    elif len(a1) > 0 and len(a2) == 0:
        new_index = a1
    elif len(a1) > 0 and len(a2) > 0: 
        a1_ = a1[:min(len(a1), len(a2))]
        a2_ = a2[:min(len(a1), len(a2))] 
        new_index = np.concatenate((a1_.reshape(-1,1), a2_.reshape(-1,1)), 1).reshape(-1, 1)
        if len(a1_) < len(a1): 
            new_index = np.concatenate((new_index, np.arange(len(a1_), len(a1)).reshape(-1, 1)), 0)
    
        if len(a2_) < len(a2): 
            new_index = np.concatenate((new_index, np.arange(len(a2_)+ len(a1), len(a2)+len(a1)).reshape(-1, 1)), 0)
    
    new_index = new_index.reshape(1,-1) 
    new_index = new_index[0,]
    df_shuffle = df_shuffle.loc[new_index]
    df_shuffle = df_shuffle.reset_index(drop=True)
    print(df_shuffle)
    df_shuffle.to_excel('././data/UW_data_with_lesion_volume_random_sample.xlsx', index=False, header=True)

if "__main__" == __name__:     
    root_dir =  "/UserData/Lymphoma_UW_Retrospective/Data/uw_analyzed/dicom/"
    nifti_path = '/UserData/Lymphoma_UW_Retrospective/Data/uw_analyzed/Xin/nifti/voxel/'
    process_cases(root_dir, nifti_path, num_fold=2)
    data = pd.read_excel('./data/UW_data_with_Hodgkin_DLBCL_labels.xlsx') 
    filename = data.Filenames.tolist()
    Disease = data.Disease.tolist() 
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
from nifti_roi_tools import determine_if_rt_should_be_included
from pydicom import dcmread
'''
Threshold: Union of 40% SUV max and 4.0 SUV 
'''
def generate_contours(dicom_series_path, rtstruct_path, nifti_img_path, save_rtstruct_path, dil):
    nii = nib.load(nifti_img_path)
    PET_volume = nii.get_fdata().copy()
    PET_volume = np.transpose(PET_volume, [1,0,2])
    rtstruct = RTStructBuilder.create_from(
    dicom_series_path=dicom_series_path, 
    rt_struct_path=rtstruct_path)
  
    # View all of the ROI names from within the image
    roi_name_list = rtstruct.get_roi_names()
    ignore_strings = ['Struct_XD', 'Struct_YD', 'Reference', 'Burden', 'DS1', 'marrow', 'osseous', 'marow', 'bone', 'marro']

    #print(roi_name_list)
    rtstruct_new = RTStructBuilder.create_new(dicom_series_path=dicom_series_path)
    # Loading the 3D Mask from within the RT Struct
    for roi_name in roi_name_list:
        ignore_rt = False 
        for ignore in ignore_strings:
            if ignore.lower() in ('struct_'+roi_name).lower():
                ignore_rt = True # remove ignored strings 

        if not determine_if_rt_should_be_included(roi_name):
            ignore_rt = True # remove XD (without label being changed), YD 
        
        roi_new_name = 'ut2_dil{}_'.format(dil) + roi_name
        if dil == 1:
            color_code = [222,184,135] 
        elif dil == 2: 
            color_code = [34,139,34] 
        elif dil == 3: 
            color_code = [255,20,147] 
        # burlywood #DEB887 [222,184,135] 
        # forestgreen #228B22 [34,139,34] 
        # deeppink1 #FF1493  [255,20,147] 
        if ignore_rt:
            mask_3d = rtstruct.get_roi_mask_by_name(roi_name) > 0 # phycisian's annotations 
            if np.sum(mask_3d) == 0:
                continue  
            rtstruct_new.add_roi(
            mask=rtstruct.get_roi_mask_by_name(roi_name), 
            color=color_code, 
            name=roi_new_name)
            continue 
        
        mask_3d = rtstruct.get_roi_mask_by_name(roi_name) > 0 # phycisian's annotations 
        if np.sum(mask_3d) == 0:
            continue  
        dilated = morphology.binary_dilation(
            mask_3d, morphology.ball(radius=dil))
        mask = np.zeros_like(dilated)

        ###### Segmentation methods ######
        PET_ROI = deepcopy(PET_volume)
        PET_ROI[dilated<0.5] = 0
        SUV_max = PET_ROI.max()
        SUV_thres = min(SUV_max * 0.4, 4.0) 
        mask[PET_ROI > max(0, SUV_thres)] = 1 

        mask_segs = measure.label(mask, connectivity=2) 
        #print(mask_segs.max())
        for ii in range(mask_segs.max()):
            mask_seg = np.zeros_like(mask)
            mask_seg[mask_segs==ii+1] = 1 
            if np.sum(mask_3d * mask_seg)/np.sum(mask_seg) < 0.01: # 1/9
                mask[mask_seg==1] = 0  
                print(np.sum(mask_3d * mask_seg), np.sum(mask_seg)) 
            #print(np.sum(mask_seg)) 
            
        #ccolors = plt.get_cmap('tab10')(np.arange(10, dtype=int))
        ######
        if np.sum(mask) == 0:
            continue 
        rtstruct_new.add_roi(
            mask=mask, 
            color=color_code, 
            name=roi_new_name,
            use_pin_hole=True
            ) 

        # separate a single mask into several segments by connected componenet analysis 
        
    rtstruct_new.save(save_rtstruct_path)
    ds = dcmread(save_rtstruct_path)
    ds.SeriesDescription = 'Union_threshold_4.0_SUV_0.4_SUVmax_dilate_{}voxel'.format(dil)
    with open(save_rtstruct_path, 'wb') as outfile:
        ds.save_as(outfile)

def process_contouring(root_dir, nifti_path, dil):
    save_rtstruct_folder = '/UserData/Lymphoma_UW_Retrospective/Data/uw_analyzed/Xin/rtstruct/union_threshold_02_dilate_{}voxel/'.format(dil)
    os.makedirs(save_rtstruct_folder, exist_ok=True)
    # petlymph_4560_petlymph_4560/petlymph_4560_petlymph_4560_20130822_PT_WB_3D_MAC_SUV.nii.gz

    folder_list = get_all_terminal_subfolders(root_dir)
    cases_list = np.arange(4500, 4800)
    cases_list = [str(ii) for ii in cases_list]
    subfolders_PTs = [subfolder for subfolder in folder_list if ("_pt_" in subfolder.lower())]
    subfolders_RTs = [subfolder for subfolder in folder_list if ("scho" in subfolder.lower())]
    nifti_folders = get_all_terminal_subfolders(nifti_path)
    
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
        save_rtstruct_subfolder = os.path.join(save_rtstruct_folder, 'petlymph_{}_petlymph_{}'.format(case, case))
        os.makedirs(save_rtstruct_subfolder, exist_ok=True)
        save_rtstruct_path = os.path.join(save_rtstruct_subfolder, 'contouring.dcm') 
        try:
            generate_contours(dicom_series_path, rtstruct_path, nifti_img_path, save_rtstruct_path, dil)    
        except: 
            print("There is sth. wrong with Case# {}".format(case))

if "__main__" == __name__:     
    root_dir =  "/UserData/Lymphoma_UW_Retrospective/Data/uw_analyzed/dicom/"
    nifti_path = '/UserData/Lymphoma_UW_Retrospective/Data/uw_analyzed/Xin/nifti/voxel/'
    process_contouring(root_dir, nifti_path, 1)
    process_contouring(root_dir, nifti_path, 2)
    process_contouring(root_dir, nifti_path, 3)

          
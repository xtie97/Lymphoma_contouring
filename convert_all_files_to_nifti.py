import os
import pydicom
import dicom2nifti
from platipy.dicom.io import rtstruct_to_nifti
from get_all_terminal_subfolders import get_all_terminal_subfolders
import nibabel as nib
from datetime import datetime
import numpy as np
import glob

def isdicom(file_path):
    #borrowed from pydicom filereader.py
    with open(file_path, 'rb') as fp:
        preamble = fp.read(128)
        magic = fp.read(4)
        if magic != b"DICM":
            return False
        else:
            return True

def get_suv_conversion_factor(test_dicom):
    '''
        Get patients' weights (in kg, to g) / date difference (to second) / Half-life of a F-18 between the injection and acquisition 
        SUV = tracer uptake (or activity concentration) in ROI (Bq/ml) / (injected dose (Bq) * 2^(-T/tau)) / patient weight (g))
        water density = 1 g/ml 
        T: delay between the injection time and the scan time, tau: half-life of the radionuclides
    '''
    dicom_corrections = test_dicom['00280051'].value
    if 'ATTN' not in dicom_corrections and 'attn' not in dicom_corrections:
        print('Not attenuation corrected -- SUV factor set to 1')
        return 1
    try:
        dicom_weight = test_dicom['00101030'].value
        dicom_manufacturer = test_dicom['00080070'].value.lower()

        #scantime info
        if dicom_manufacturer[0:2] == 'ge' and '0009100D' in test_dicom:
            dicom_scan_datetime = test_dicom['0009100D'].value[0:14]  #need to check!
        else:
            dicom_scan_datetime = test_dicom['00080021'].value +  test_dicom['00080031'].value
            dicom_scan_datetime = dicom_scan_datetime[0:14]

        #radiopharmaceutical info
        radiopharm_object = test_dicom['00540016'][0]
        if '00181074' in radiopharm_object and '00181075' in radiopharm_object:
            dicom_half_life = radiopharm_object['00181075'].value
            dicom_dose = radiopharm_object['00181074'].value

            if '00181078' in radiopharm_object:
                if radiopharm_object['00181078'].value != None:
                    dicom_inj_datetime = radiopharm_object['00181078'].value
                else:
                    dicom_inj_datetime = dicom_scan_datetime[0:8] + radiopharm_object['00181072'].value
            else:
                dicom_inj_datetime = dicom_scan_datetime[0:8] + radiopharm_object['00181072'].value
        # sometimes tracer info is wiped, and if GE, can be found in private tags
        else:
            print('No dose information -- SUV factor set to 1')
            return 1

    except Exception:
        print('Problem reading SUV info -- SUV factor set to 1')
        return 1

    # date difference
    scan_datetime = datetime.strptime(dicom_scan_datetime, '%Y%m%d%H%M%S')
    if not 'philips' in dicom_manufacturer.lower():
        dicom_inj_datetime = dicom_inj_datetime[0:14]
    inj_datetime = datetime.strptime(dicom_inj_datetime, '%Y%m%d%H%M%S')
    diff_seconds = (scan_datetime - inj_datetime).total_seconds()
    if diff_seconds < 0:
        diff_seconds += 24*3600 
    #SUV factor
    dose_corrected = dicom_dose * 2**(- diff_seconds/dicom_half_life)
    suv_factor = 1/((dose_corrected/dicom_weight)*0.001) 
    return suv_factor

def convert_pet_nifti_to_suv_nifti(nifti_read_filename, test_dicom, nifti_save_filename):
    suv_factor = get_suv_conversion_factor(test_dicom)
    if suv_factor != 1:
        orig = nib.load(nifti_read_filename)
        data = orig.get_fdata()
        new_data = data.copy()
        new_data = new_data * suv_factor
        if np.max(new_data) < 1:
            print('*** PET image values seem low for {}. Check SUV conversion'.format(nifti_save_filename))
        suv_img = nib.Nifti1Image(new_data, orig.affine, orig.header)
        nib.save(suv_img, nifti_save_filename)
        return 1
    else:
        return 0

def find_path_to_dicom_image_that_corresponds_with_rtsrtuct(dir_i, dicom_id, dicom_study_date, modality_of_interest, subdirs):
    #just looks one folder up to see if any directories contain DICOM images of the modality of interest. Takes first one
    one_folder_up = os.path.dirname(dir_i)
    patient_folders = [s for s in subdirs if one_folder_up in s]
    corresponding_dicom = ''
    for dir_p in patient_folders:
        files2 = os.listdir(dir_p)
        temp_file = files2[0]
        if isdicom(os.path.join(dir_p, temp_file)) == False:
            continue
        test2_dicom = pydicom.dcmread(os.path.join(dir_p, temp_file))
        if  (test2_dicom['00080020'].value == dicom_study_date and
                test2_dicom['00080060'].value == modality_of_interest and
                test2_dicom['00100020'].value.lower() == dicom_id.lower()):
            corresponding_dicom = dir_p
    return corresponding_dicom

def convert_PT_CT_files_to_nifti(top_dicom_folder, top_nifti_folder):
    #modality of interest is the modality that will be the reference size for the RTSTRUCT contours, defined by DICOM
    #type ('PT, 'CT', 'MR')
    files = glob.glob(top_dicom_folder + "/*.dcm") 
    if len(files) < 1:
        print('Empty folder: ' + files)
        raise Exception("Fail to find DICOM files")

    # get dicom info for saving
    test_dicom = pydicom.dcmread(files[0])
    dicom_modality = test_dicom['00080060'].value
    dicom_name = str(test_dicom['00100010'].value).lower()
    dicom_id = test_dicom['00100020'].value.lower()
    dicom_study_date = test_dicom['00080020'].value
    dicom_series_description = test_dicom['0008103e'].value

    # unique names for subjects and scans
    subject_save_name = dicom_id + '_' + dicom_name.replace(' ', '_').replace('__', '_')
    subject_save_folder = os.path.join(top_nifti_folder, subject_save_name)
    os.makedirs(subject_save_folder, exist_ok=True)
    scan_save_name =  '{}_{}_{}_{}'.format(subject_save_name, dicom_study_date, dicom_modality, \
        dicom_series_description.replace(' ', '_'))
    
    if dicom_modality in ['CT', 'MR', 'NM']:
        dicom2nifti.dicom_series_to_nifti(top_dicom_folder, os.path.join(subject_save_folder, scan_save_name + '.nii.gz'), reorient_nifti=False)
    elif dicom_modality == 'PT':
        dicom2nifti.dicom_series_to_nifti(top_dicom_folder, os.path.join(subject_save_folder, scan_save_name + '.nii.gz'), reorient_nifti=False)
        convert_pet_nifti_to_suv_nifti(os.path.join(subject_save_folder, scan_save_name + '.nii.gz'), test_dicom,
                                           os.path.join(subject_save_folder, scan_save_name + '_SUV.nii.gz'))
       
def convert_rtstruct_to_nifti(annotator_dicom_folder: str, top_nifti_folder: str, mismatch_case: str, modality_of_interest: str='PT'):
    # modality of interest is the modality that will be the reference size for the RTSTRUCT contours
    
    files = glob.glob(annotator_dicom_folder + "/*.dcm") 
    if len(files) < 1:
        print('Empty folder: ' + files)
        raise Exception("Fail to find DICOM files")

    # get dicom info for saving
    test_dicom = pydicom.dcmread(files[0])
    dicom_modality = test_dicom['00080060'].value
    dicom_name = str(test_dicom['00100010'].value).lower()
    dicom_id = test_dicom['00100020'].value.lower()
        
    # unique names for subjects and scans
    subject_save_name = dicom_id + '_' + dicom_name.replace(' ', '_').replace('__', '_')
    subject_save_folder = os.path.join(top_nifti_folder, subject_save_name)
    os.makedirs(subject_save_folder, exist_ok=True)

    #if dicom_modality == 'RTSTRUCT' and rtstruct_string_identifier.lower() in dicom_series_description.lower():
    if dicom_modality == 'RTSTRUCT': 
        # might be multiple rtstructs in folder
        for file_i in files:
            if isdicom(file_i) == True:
                rt_dicom = pydicom.dcmread(file_i)
                if rt_dicom['00080060'].value == 'RTSTRUCT':
                    dicom_modality = rt_dicom['00080060'].value
                    dicom_name = str(rt_dicom['00100010'].value).lower()
                    dicom_id = rt_dicom['00100020'].value.lower()
                    dicom_study_date = rt_dicom['00080020'].value
                    dicom_series_description = rt_dicom['0008103e'].value

                    # unique names for subjects and scans
                    subject_save_name = dicom_id + '_' + dicom_name.replace(' ', '_').replace('__', '_')
                    subject_save_folder = os.path.join(top_nifti_folder, subject_save_name)
                    scan_save_name = dicom_study_date + '_' + dicom_modality + '_' + dicom_series_description.replace(
                        ' ', '_')

                    # find the corresponding DICOM series
                    one_folder_up = os.path.dirname(annotator_dicom_folder)
                    subdirs = os.listdir(one_folder_up)
                    subdirs = [os.path.join(one_folder_up, s) for s in subdirs if mismatch_case in s]
                    corresp_dicom_path = find_path_to_dicom_image_that_corresponds_with_rtsrtuct(annotator_dicom_folder,
                                                                                                     dicom_id,
                                                                                                     dicom_study_date,
                                                                                                     modality_of_interest,
                                                                                                     subdirs)
                    if corresp_dicom_path == '':
                        print('***!!! Unable to find correspoding DICOM images for %s. RTStruct will not be made ***!!'.format(subject_save_name))
                        continue
                    # save
                    rtstruct_nifti_save_path = os.path.join(subject_save_folder, scan_save_name)
                    if not os.path.exists(rtstruct_nifti_save_path):
                        os.makedirs(rtstruct_nifti_save_path)
                    try:
                        rtstruct_to_nifti.convert_rtstruct(corresp_dicom_path, file_i,
                                                           output_dir=rtstruct_nifti_save_path)
                    except:
                        print("!!!!!!!!!!!XXXXXXX   Problem with {}    XXXXXX!!!!!!!!!!!!!! ".format(file_i))

                    #check dimensions match
                    roi_files = os.listdir(rtstruct_nifti_save_path)
                    roi_file = os.path.join(rtstruct_nifti_save_path, roi_files[0])
                    roi_nii = nib.load(roi_file)
                    nii_dims = roi_nii.header.get_data_shape()
                    dicom_files = os.listdir(corresp_dicom_path)
                    dicom_file = os.path.join(corresp_dicom_path, dicom_files[0])
                    dicom_info = pydicom.dcmread(dicom_file)
                    rows = dicom_info['00280010'].value
                    if rows != nii_dims[0] or len(dicom_files) != nii_dims[2]:
                        print('!!!**** ROI nifti is not same shape as DICOM (maybe) for {}'.format(rtstruct_nifti_save_path))

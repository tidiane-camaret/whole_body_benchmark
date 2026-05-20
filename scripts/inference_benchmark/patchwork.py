# use conda activate jupyter
import sys
sys.path.append("/software/")
import patchwork2.model as patchwork
MODEL_PATH = "/nfs/data/nii/data1/Analysis/camaret___whole_body_benchmark/ANALYSIS_ana001/patchwork_oppscreen_all/model_patchwork.json"
IMG_PATH = "/nfs/data/nii/data1/Analysis/camaret___whole_body_benchmark/ANALYSIS_ana001/patchwork_inference_benchmark/imagesTs/100003.nii"
PRED_PATH = "/nfs/data/nii/data1/Analysis/camaret___whole_body_benchmark/ANALYSIS_ana001/patchwork_inference_benchmark/predsTs/100003.nii.gz"

model_kid = patchwork.PatchWorkModel.load(MODEL_PATH,immediate_init=True)

w = [0]*86
w[0] = -1.5
w[71] = 1.5
w[58] = 1.5
res,r = model_kid.apply_on_nifti(IMG_PATH, PRED_PATH,
  generate_type='random',
  augment={},
  #crop_fdim=[2,3],
  repetitions=4,
  num_chunks=20,
  scale_to_original=True,
  #lazyEval=0.7,
  branch_factor=[4,4,8],
  out_typ='atls',
  QMapply_paras={'weights':w},
  level='mix'
     )


"""
python scripts/nnunet_patchwork_comp/eval.py \
    --gt-pattern   /nfs/data/nii/data1/Analysis/camaret___whole_body_benchmark/ANALYSIS_ana001/nnunet/nnUNet_raw/Dataset003_oppscreen_20_subjects/labelsTs/{subject}.nii.gz \
    --pred-pattern /nfs/data/nii/data1/Analysis/camaret___whole_body_benchmark/ANALYSIS_ana001/nnunet/nnUNet_raw/Dataset003_oppscreen_20_subjects/predsTs/{subject}.nii.gz \
    --subjects     /nfs/data/nii/data1/Analysis/camaret___whole_body_benchmark/ANALYSIS_ana001/nnunet/nnUNet_raw/Dataset003_oppscreen_20_subjects/labelsTs \
    --output       results/results_nnunet_20_separate_dataset.json --no-nsd
"""
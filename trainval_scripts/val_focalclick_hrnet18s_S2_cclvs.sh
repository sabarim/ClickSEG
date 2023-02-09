python scripts/evaluate_model.py FocalClick\
  --model_dir=./weights/\
  --checkpoint=hr18ss2_cclvs\
  --logs-path=./experiments/evaluation_logs/focalclick_hrnet18s_cclvs_rebuttal/\
  --infer-size=256\
  --datasets=SBD,DAVIS17,COCOVal\
  --gpus=0\
  --n-clicks=10\
  --target-iou=0.9\
  --thresh=0.5\
  --multi_instance
  --n-clicks 10
  #--vis
  #--target-iou=0.95\

#--datasets=GrabCut,Berkeley,PascalVOC,COCO_MVal,SBD,DAVIS,D585_ZERO,D585_SP\
#--datasets=DAVIS_high,DAVIS_mid,DAVIS_low\
  


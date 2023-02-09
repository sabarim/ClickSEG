python scripts/evaluate_model.py FocalClick\
  --model_dir=./experiments/focalclick/hrnet32_S2_cclvs/000_hrnet32_S2_cclvs/checkpoints/\
  --checkpoint=weights/hrnet32.pth\
  --infer-size=256\
  --datasets=COCO\
  --gpus=0\
  --n-clicks=10\
  --target-iou=0.90\
  --thresh=0.5\
  --multi_instance
  #--vis
  #--target-iou=0.95\

#--datasets=GrabCut,Berkeley,PascalVOC,COCO_MVal,SBD,DAVIS,D585_ZERO,D585_SP\
#--datasets=DAVIS_high,DAVIS_mid,DAVIS_low\
  


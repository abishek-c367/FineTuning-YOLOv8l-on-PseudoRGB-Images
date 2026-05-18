# config.py
cyclegan_weights = "Cycle_gan/cyclegan_epoch_45.pth"
teacher_weights  = "Teacher/yolov8l.pt"
flir_root        = "FLIR_ADAS_v2"
split            = "images_thermal_train"
coco_json        = "coco.json"
imgsz            = 512
batch            = 8
workers          = 4
epochs           = 50
alpha            = 0.2
num_classes      = 3
device           = "cuda"
checkpoint_dir   = "checkpoints_distill"
pseudo_rgb_cache = "Cycle_gan/pseudo_rgb_cache"
# METRIC_LOG_INTERVAL = 100 
val_split_dir = "FLIR_ADAS_V2/images_thermal_val"
val_coco_json = "FLIR_ADAS_V2/images_thermal_val/coco.json"
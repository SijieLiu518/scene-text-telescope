conda activate textzoom
# CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=16 --STN --exp_name EXP_NAME --text_focus --gradient
python main.py --batch_size=48 --STN --exp_name EXP_NAME --text_focus --gradient
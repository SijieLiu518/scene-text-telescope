conda activate textzoom
# CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=16 --STN --exp_name EXP_NAME --text_focus --gradient
# train
# python main.py --batch_size=48 --STN --exp_name TBSRN_CPLoss3 --text_focus --gradient --mask
# test
python main.py --batch_size=48 --STN --exp_name TBSRN_CPLoss3 --text_focus --resume ./checkpoint/TBSRN_CPLoss3/model_best.pth --test --test_data_dir ./dataset/TextZoom/test --gradient --rec aster --mask
conda activate textzoom
# CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=16 --STN --exp_name EXP_NAME --text_focus --gradient
# train
python main.py --arch=tbsrn_seg_emb --batch_size=28 --STN --exp_name TBSRN_CPLoss11 --text_focus --gradient --mask --cp_loss --wgan_loss
# test
# python main.py --batch_size=48 --STN --exp_name TBSRN_CPLoss4 --text_focus --resume ./checkpoint/TBSRN_CPLoss4/model_best.pth --test --test_data_dir ./dataset/TextZoom/test --gradient --rec moran --mask
# fix up "fatal: unable to access 'https://github.com"
# git config --global --unset http.sslVerify false
# git config --global http.sslVerify false
# git config --global http.postBuffer 524288000
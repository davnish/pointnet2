model_name=2
epoch=200
embd=64

# python train.py --epoch $epoch --radius 30 --step_size 50 --points_taken 4096 --batch_size 32 --grid_size 25 --model_name $model_name --model pointnet2
python train.py --epoch 400 --lr 0.001 --step_size 20 --points_taken 4096 --batch_size 32 --grid_size 25 --model_name $model_name --model pointnet
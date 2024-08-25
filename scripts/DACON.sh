export CUDA_VISIBLE_DEVICES=0

python main.py \
	--anormly_ratio 1 --num_epochs 30 --batch_size 512 \
	--mode train --dataset PSM --data_path data/DACON \
	--input_c 51 --output_c 51 --step 1 \
	--win_size 128 --downsample 4

python main.py \
	--anormly_ratio 1 --num_epochs 10 --batch_size 256 \
	--mode test --dataset PSM --data_path data/DACON \
	--input_c 51 --output_c 51 --step 1 --pretrained_model 20 \
	--win_size 128 --downsample 4



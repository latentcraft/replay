mkdir -p ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k_02
mkdir -p ../logs/DeepEyes/DeepEyes-7B

CUDA_VISIBLE_DEVICES=0 nohup python eval/eval_hr4k_transformers.py --model_base ../model/DeepEyes-7B --checkpoint_dir ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k_02 --chunk_idx 0 > ../logs/DeepEyes/DeepEyes-7B/hr4k_gpu_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval/eval_hr4k_transformers.py --model_base ../model/DeepEyes-7B --checkpoint_dir ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k_02 --chunk_idx 1 > ../logs/DeepEyes/DeepEyes-7B/hr4k_gpu_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python eval/eval_hr4k_transformers.py --model_base ../model/DeepEyes-7B --checkpoint_dir ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k_02 --chunk_idx 2 > ../logs/DeepEyes/DeepEyes-7B/hr4k_gpu_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python eval/eval_hr4k_transformers.py --model_base ../model/DeepEyes-7B --checkpoint_dir ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k_02 --chunk_idx 3 > ../logs/DeepEyes/DeepEyes-7B/hr4k_gpu_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python eval/eval_hr4k_transformers.py --model_base ../model/DeepEyes-7B --checkpoint_dir ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k_02 --chunk_idx 4 > ../logs/DeepEyes/DeepEyes-7B/hr4k_gpu_4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python eval/eval_hr4k_transformers.py --model_base ../model/DeepEyes-7B --checkpoint_dir ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k_02 --chunk_idx 5 > ../logs/DeepEyes/DeepEyes-7B/hr4k_gpu_5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python eval/eval_hr4k_transformers.py --model_base ../model/DeepEyes-7B --checkpoint_dir ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k_02 --chunk_idx 6 > ../logs/DeepEyes/DeepEyes-7B/hr4k_gpu_6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python eval/eval_hr4k_transformers.py --model_base ../model/DeepEyes-7B --checkpoint_dir ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k_02 --chunk_idx 7 > ../logs/DeepEyes/DeepEyes-7B/hr4k_gpu_7.log 2>&1 &
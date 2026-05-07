mkdir -p ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k
mkdir -p ../logs/DeepEyes/DeepEyes-7B

HIP_VISIBLE_DEVICES=4 nohup /z_data/conda_envs/xy_repl_deepeyes/bin/python eval/eval_hr4k_transformers.py --model_base ../model/DeepEyes-7B --checkpoint_dir ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k --chunk_idx 0 > ../logs/DeepEyes/DeepEyes-7B/hr4k_gpu_0.log 2>&1 &
HIP_VISIBLE_DEVICES=5 nohup /z_data/conda_envs/xy_repl_deepeyes/bin/python eval/eval_hr4k_transformers.py --model_base ../model/DeepEyes-7B --checkpoint_dir ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k --chunk_idx 1 > ../logs/DeepEyes/DeepEyes-7B/hr4k_gpu_1.log 2>&1 &
HIP_VISIBLE_DEVICES=6 nohup /z_data/conda_envs/xy_repl_deepeyes/bin/python eval/eval_hr4k_transformers.py --model_base ../model/DeepEyes-7B --checkpoint_dir ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k --chunk_idx 2 > ../logs/DeepEyes/DeepEyes-7B/hr4k_gpu_2.log 2>&1 &
HIP_VISIBLE_DEVICES=7 nohup /z_data/conda_envs/xy_repl_deepeyes/bin/python eval/eval_hr4k_transformers.py --model_base ../model/DeepEyes-7B --checkpoint_dir ../mirror/DeepEyes/outputs/DeepEyes-7B/hr4k --chunk_idx 3 > ../logs/DeepEyes/DeepEyes-7B/hr4k_gpu_3.log 2>&1 &
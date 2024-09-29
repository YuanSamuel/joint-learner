python -m jl.data_engineering.add_labels --input collector_output/cache_accesses_xalan500m.csv --output data/xalan500m_labeled.csv --cache_size 4096

CUDA_VISIBLE_DEVICES=1 nohup python -m jl.train.train_joint_mlp --ip_history_window 15 --batch_size 256 --model_name joint_mlp_xalan --cache_data_path data/xalan_large_labeled.csv -p collector_output/prefetches_xalan.csv --dataset joint_xalan_large_15w &


# joint-learner

## Instructions for Training Models


### Using [ChampSim](https://github.com/ChampSim/ChampSim)  to Collect Data Traces

1. Enter the sim/ folder

```
cd sim
```

2. Set up the simulator by downloading dependencies:

```
git submodule update --init
vcpkg/bootstrap-vcpkg.sh
vcpkg/vcpkg install
```

ChampSim takes a JSON configuration script. Use champsim_config.json as a working example.

```
./config.sh <configuration file>
make
```

3. Download SPEC Benchmark traces from [this site](https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/)


4. Run the simulator, e.g.


```
bin/champsim --warmup_instructions 50000000 --simulation_instructions 50000000 --name sphinx ~/path/to/traces/482.sphinx3-234B.champsimtrace.xz
```

The number of warmup and simulation instructions given will be the number of instructions retired. 

### Data Preprocessing

In order to add labels to cache instructions for whether an instruction would be cached under Belady's OPT, run the `add_labels.py` script from the root directory, e.g.

```
python src/data_engineering/add_labels.py --input ~/path/to/simtraces/cache_accesses_sphinx.csv --output ~/path/to/output --cache_size 2048 * 2
```

The size of the cache is variable, and ChampSim's cache size is configurable in the `champsim_config.json`. The default is 2048 * 16, or the number of sets * number of ways in the LLC cache.


### Model Training

To train the joint model, firstly constrastively train the encoders using through the `train_embedders.py` script:

```
CUDA_VISIBLE_DEVICES=1 python src/train_embedders.py --prefetch_data_path ~path/to/labeled_data/prefetches_sphinx.csv --cache_data_path ~/path/to/labeled_data/cache_accesses_sphinx.csv --model_name contrastive_encoder_sphinx -l 0.0001 --config ./configs/base_voyager_cont.yaml
```

The models should be saved to the `./data/model/` folder.

To train the cache replacement model individually, use `train_mlp.py`:

```
CUDA_VISIBLE_DEVICES=1 python src/train_mlp.py --ip_history_window 10 --batch_size 256 --model_name mlp_sphinx --cache_data_path ~/path/to/labeled_data/labeled_cache_sphinx.csv
```

Use the `encoder_name` flag to use the contrastively trained encoder:

```
CUDA_VISIBLE_DEVICES=1 python src/train_mlp.py --ip_history_window 10 --batch_size 256 --model_name mlp_sphinx --cache_data_path ~/path/to/labeled_data/labeled_cache_sphinx.csv --encoder_name contrastive_encoder_sphinx
```

To train the voyage prefetcher model, firstly download the traces from [this link](https://utexas.app.box.com/s/2k54kp8zvrqdfaa8cdhfquvcxwh7yn85/folder/132805020714) for the appropriate benchmark (using the ChampSim trace instead is WIP), and use `train_voyager.py`:

```
CUDA_VISIBLE_DEVICES=1 python src/train_voyager.py --prefetch_data_path ~/path/to/data/482.sphinx3-s0.txt.xz --model_name voyager_base
```

Or, specify the `encoder_name` to use the contrastively trained encoder as above.
# Marked Line-art Vectorization

  * Model: https://huggingface.co/nopperl/marked-lineart-vectorizer
  * Demo: https://huggingface.co/spaces/nopperl/lineart-vectorizer
  * Paper: https://github.com/nopperl/marked-lineart-vectorization-thesis

Sources for research on vectorizing clean line-art raster images using an encoder-decoder model trained with vector and raster-based losses (loosely based on Im2Vec [0]).
The decoder could potentially be coupled with a different encoder in order to generate line-art vector images from input images of different domains.

## Quick Usage

An ONNX version of the PyTorch model for vectorization is provided for efficient and easy inference. [Download it](https://huggingface.co/nopperl/marked-lineart-vectorizer/resolve/main/model.onnx?download=true) to the working directory. It can be used [in a various range of runtimes and languages](https://onnx.ai/supported-tools.html#deployModel). To get started, a standalone inference script in Python based on CUDA is also provided. To use it, [install CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), then install the required Python packages using:

```
pip install -r onnx-requirements.txt
```

Then, start the inference on any line-art raster image:

```
scripts/onnx_inference.py model.onnx -i $IMAGE_NAME
```

This script can also be run in a docker container. For this, install [Docker](https://docs.docker.com/engine/install/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Then, build the docker image:

```
docker compose build
```

Finally, run the inference script:

```
docker compose run onnx  -i data/$IMAGE_NAME
```

## Setup

The main environment can be setup in two ways: using Docker or Anaconda.

### Using Docker (recommended)
Install [Docker](https://docs.docker.com/engine/install/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Build the image using the nvidia runtime (the GPU needs to be available at build time).

`docker build -t marked_lineart_vec .`

Run a Docker container using the built image. All scripts in the repo can be run inside the container.

`docker run --rm --gpus all -w $(pwd) -v $(pwd):$(pwd) -it marked_lineart_vec bash`

### Using conda
Create a conda environment.

```
conda create -n marked_lineart_vec python=3.7 -y
conda activate marked_lineart_vec
```

Install the dependencies.

```
scripts/conda_install.sh
```

`LD_LIBRARY_PATH=$CONDA_PREFIX/lib` might have to be prepended to each command.

## Usage

### Data
#### Get datasets
There are three datasets which need to be retrieved: Tonari clean animation frames, SketchBench sketches and TU Berlin amateur sketches. The latter two can be retrieved and sanitized using:

```
scripts/get_open_data.sh
```

Unfortunately, the Tonari subset is proprietary. If it is available locally, it can be retrieved with:

```
scripts/get_tonari_data.sh
```

If the Tonari data is not available, the next steps might not work correctly. It makes sense to at least simulate the existance of Tonari data, e.g. using a part of the TU Berlin data:

```
cp -r data/clean/tuberlin/svg/panda/ data/clean/tonari
```

#### Get dataset statistics

Statistics about the subsets can be computed using:

```
scripts/get_data_statistics.py
```

Statistics will be cached as json file for the specific dataset in the `data/clean` directory. Summary tables and figures are stored in the `report` directory.

Further statistics about irregularities in the Tonari subset can be computed and stored in the `report` directory using:

```
scripts/get_tonari_statistics.py
```

#### Preprocess data
To preprocess the open subsets, run:

```
scripts/preprocess_clean_svg.py
```

To preprocess the Tonari subset, run:

```
scripts/preprocess_tonari.py
```

#### Prepare combined dataset
Up until now, the data subsets were processed independently. Since the model is trained on all subsets, they need to be combined into a single dataset. This dataset is then split into a train, validation and test dataset. This is done using:

```
scripts/combine_and_split_dataset.py
```

Note: If the Tonari subset is not available, it makes sense to use:

```
scripts/combine_and_split_dataset.py --test_subset sketchbench
```

This way, the test split is sampled from the sketchbench dataset, enabling evaluation on an openly available dataset.


### Training

A model can be trained using:

`python marked_lineart_vec/train.py -c $CONFIG_FILE`

The configuration needs to be specified by a config file. The `config` directory provides configs used during the research. The config used for the provided final model is `config/marked-clean.yaml`.

Training a model will result in a model file being placed in the `logs/MarkedReconstructionModel/$VERSION_NAME/checkpoints` directory.

To test a trained model on one or multiple images (e.g. of the images in the test dataset), run:

```
python inference.py -m $MODEL_FILE -d data/processed/sketchbench/tonari-black-tonari-blue-tonari-red-tonari-lime-sketchbench-black-tuberlin-black-512-0.512 
```

A model file (`*.ckpt`) needs to be specified, e.g. one from `logs/MarkedReconstructionModel/$VERSION_NAME/checkpoints/`. `test_marked_clean.py` can be used instead of `inference.py` to get more verbose output.

#### Convert trained models to ONNX

The trained model can be converted to the ONNX format for easier and more efficient deployment. To do this, run:

```
marked_lineart_vec/convert_to_onnx $MODEL_FILE
```

#### Extract information from logs

Training details are logged to the `logs/MarkedReconstructionModel/$VERSION_NAME/tensorboard` directory. To extract relevant information from these records, run:

```
python scripts/extract_tensorboard_data.py --log_dir logs/MarkedReconstructionModel/$VERSION_NAME/tensorboard/
```

The results are placed in the same directory. The `summary.json` file is used for the evaluations in the paper.

This can also be processed in parallel for all logs using:

```
scripts/extract_tensorboard_data.sh
```

To save disk space, the tensorboard records can then be removed using:

```
scripts/delete_tensorboard_records.sh
```

### Evaluation

#### Setup ground truth

The ground truth and predicted images need to be stored in the `outputs` directory for comparison. To extract the ground truth data and copy it to this directory, run:

```
scripts/copy_ground_truth.sh
```

Now, the output images for every vectorization method need to be stored with an identical structure.

#### Run vectorization on test dataset

To run the vectorization method on the test dataset, use:

```
scripts/vectorize_test_directories.sh
```

The results will be in `outputs/marked_lineart_vec`.

#### Setup prior work

##### Deep Vectorization of Technical Line Drawings

Clone the git repository:

```
git clone https://github.com/nopperl/Deep-Vectorization-of-Technical-Drawings
cd Deep-Vectorization-of-Technical-Drawings
```

Now, run the vectorization for every subdirectory of `outputs/ground-truth` separately. For this copy the test line-art images to the `data` directory, e.g.:

```
cp -r ../douga-vectorizer/outputs/ground-truth/512-0.512/*png data
```

Run the model on the lineart images using docker compose:

```
docker compose up
```

The vector images will be stored in `logs/outputs/vectorization/lines/merging_output`. Copy the output next to the ground truth, e.g. to `outputs/deepvectech/512-0.512`.

##### Virtual Sketching

Clone the git repository:

```
git clone https://github.com/nopperl/virtual_sketching
cd virtual_sketching
```

Now, run the vectorization for every subdirectory of `outputs/ground-truth` separately. For this, copy the test line-art images to the `sample_inputs` directory, e.g.:

```
rm -rf sample_inputs/clean_line_drawings/*
cp -r ../douga-vectorizer/outputs/ground-truth/512-0.512/*png sample_inputs/clean_line_drawings
```

Run the model on the lineart images using docker compose:

```
docker compose up
```

The vector images will be stored in `outputs/sampling/clean_line_drawings__pretrain_clean_line_drawings/svg`. Copy the output next to the ground truth, e.g. to `outputs/virtual_sketching/512-0.512`.

##### PolyVector Flow

Clone the git repository:

```
git clone https://github.com/nopperl/line-drawing-vectorization-polyvector-flow
cd line-drawing-vectorization-polyvector-flow
```

Acquire a license for the [Gurobi](https://www.gurobi.com/features/web-license-service/) library. Place the license file `gurobi.lic` in the repository directory.

Now, run the vectorization for every subdirectory of `outputs/ground-truth` separately. For this, copy the test line-art images to the `inputs` directory, e.g.:

```
rm -rf inputs/*
cp -r ../douga-vectorizer/outputs/ground-truth/512-0.512/*png inputs
```

Run the model on the lineart images using docker compose:

```
docker compose up
```

The vector images will be stored in `outputs`. Copy the output next to the ground truth, e.g. to `outputs/polyvector-flow/512-0.512`.


###### AutoTrace

Clone the git repository:

```
git clone https://github.com/nopperl/autotrace
cd autotrace
```

Now, run the vectorization for every subdirectory of `outputs/ground-truth` separately. For this copy the test line-art images to the `inputs` directory, e.g.:

```
rm -rf inputs/*
cp -r ../douga-vectorizer/outputs/ground-truth/512-0.512/*png inputs
```

Run the model on the lineart images using docker compose:

```
docker compose up
```

The vector images will be stored in `outputs`. Copy the output next to the ground truth, e.g. to `outputs/autotrace/512-0.512`.

#### Run evaluation

The disparate output images are consolidated in the `outputs` directory. First, sanitize them for the evaluation script:

```
scripts/postprocess_outputs_for_eval.sh outputs
```

Then, the evaluation can be reproduced by following the `notebooks/evaluation.ipynb` notebook.

#### Ablation

Likewise, the notebook for reproducing ablation is provided at `notebooks/ablation.ipynb`. However, the logs for the compared model versions are not provided in this repository, so they have to be reproduced before the notebook can be used.

### Further artifacts

This section gives instructions on how to reproduce artifacts for the report not covered in the above sections.

#### Visualize order
To visualize the vector topology of an SVG image, use:

```
scripts/visualize_order.py $SVG_FILE
```

#### Example figures

Some further example figures (e.g. showing data augmentations) can be generated using the scripts in the `marked_lineart_vec/examples` directory.

## References

[0]: http://geometry.cs.ucl.ac.uk/projects/2021/im2vec/

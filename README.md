# LLM transformer


## Configuration

### Dependencies

The dependencies for this project are managed by [Poetry](https://python-poetry.org). To install them, run

```bash
poetry install
```

Some of the dependencies are:
- Pytorch 2.0
- Python 3.10

### Docker

A Dockerfile is provided to run the code in a container. To build the image, run

```bash
./build_docker_image.sh
```

The image name is `$HOSTNAME/llm-transformer`. To run the container, run

```bash
./docker.sh python -m llmt.main --help
```

### Hardware

This code was developed and tested on the Nvidia 4090 GPU with 24GB of memory.

## Usage

### Setup

```sh
huggingface-cli login
cp ~/cache/huggingface/token ./data/
poetry install
```

Token is supposed to be under directory `./data`

### Dataset

In order to download the dataset, run

```bash
./docker.sh python -m llmt.main dataset download
```

and it will be downloaded under `-./data`.

### Training

In order to train a model, run

```bash
./docker.sh python -m llmt.main train
```

We use the tokenizer from https://huggingface.co/replit/replit-code-v1-3b

### Validation and metrics

_TODO_

Links:
- https://github.com/openai/human-eval-infilling
- https://github.com/nuprl/MultiPL-E

### Inference

_TODO_

# TODO

- [ ] Implement the test function to evaluate the generated code
- [ ] Use sintax trees for the languages to remove the spaces which add no information and may lead to slower learning
- [ ] Use sintax trees to change the variable names
# LLM transformer


## Setup

```sh
huggingface-cli login
cp ~/cache/huggingface/token ./data/
poetry install
```

Token is supposed to be under directory `./data`

## Download dataset

```
poetry run python -m llma.main dataset download
```

## Train

```sh
poetry run python -m llma.main train
```

We use the tokenizer from https://huggingface.co/replit/replit-code-v1-3b

## Evaluation:

_work in progress_

Links:
- https://github.com/openai/human-eval-infilling
- https://github.com/nuprl/MultiPL-E

# Docker

You can run the code in the docker containers. 

First you need to create the docker image:

```sh
./build_docker_image.sh
```

The name of the docker image is $HOSTNAME/llm-kotlin

Then you can run the docker container:

```sh
./run_docker.sh python -c llma.main --help
```

The previous command will display the help message.

# TODO

- [ ] Implement the test function to evaluate the generated code
- [ ] Use sintax trees for the languages to remove the spaces which add no information and may lead to slower learning
- [ ] Use sintax trees to change the variable names
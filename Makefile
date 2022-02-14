.DEFAULT_GOAL := help

NAME	:= 2021-10-21-resnet18-cifar10-baselines
TAG    	:= $(shell git log -1 --format=%h)
IMG    	:= ${NAME}:${TAG}
LATEST	:= ${NAME}:latest

help:			## Show this help dialog.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

create-venv:		## Create an isolated virtual environment for this experiment.
	virtualenv env -p python3.9
	. env/bin/activate && pip install -r requirements.txt

build:			## Build the Docker image.
	DOCKER_BUILDKIT=1 docker build --ssh default --build-arg WANDB_API_KEY=$(WANDB_API_KEY) --build-arg WANDB_TAGS=${IMG} -t ${IMG} .
	docker tag ${IMG} ${LATEST}

run:			## Run the experiment (using gpus=0,1 to specify GPU visibility).
	docker run -it --rm --runtime=nvidia --ipc=host \
		-e NVIDIA_VISIBLE_DEVICES=$(gpus) \
		-v /home-local2/adtup.extra.nobkp:/home-local2/adtup.extra.nobkp \
		${LATEST} \
		bash src/run.sh

run-lambda:		## Run the experiment on a LambdaStack server (using gpus=0,1 to specify GPU visibility).
	docker run -it --rm --ipc=host \
		--gpus '"device=$(gpus)"' \
		-v /home-local2/adtup.extra.nobkp:/home-local2/adtup.extra.nobkp \
		${LATEST} \
		bash src/run.sh

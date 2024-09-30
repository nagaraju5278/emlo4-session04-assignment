# To build the docker image

docker build -t dog_breed_classifier .

# To run the docker container

docker run -it --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs -v $(pwd)/checkpoints:/app/checkpoints dog_breed_classifier

#train the model:
./scripts/train.sh

#To evaluate the model:
./scripts/eval.sh

#To run inferen
./scripts/infer.sh

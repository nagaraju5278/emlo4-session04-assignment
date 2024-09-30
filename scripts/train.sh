   #!/bin/bash
   docker run --rm --shm-size=2g -v $(pwd)/src:/app/src -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs -v $(pwd)/checkpoints:/app/checkpoints dog_breed_classifier python src/train.py
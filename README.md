## Built Environment
* `python`: 3.6.12
* `tensorflow`: 1.14.0
* `CUDA`: 10.0
* `gym`: 0.17.2

## Train your own model
```
python train.py -w=[window size]
                -m=[model number]
                -e=[num. episode]
                -s=[num. steps in one episode]
                -v=[StockTradingEnv number]
                -r=[stock region 'us' or 'cn']
                --device=[cpu or gpu]
                --gpu=[which gpu to use]
```
## Use Trained model
Please check the notebook.

## Deploy model to Docker
Build Docker image from Dockerfile
```
docker build -t finrl:latest .
```
Run Docker Container
```
docker run -v .../results/:/home/results \
           -v .../weights/:/home/weights \
           -v .../reward_results/:/home/reward_results \
           -p 6006:6006 \
           --gpus all \
           -ti finrl:latest /bin/bash
```
Export Docker image
```
docker save finrl:latest | gzip > finrl.tar.gz
```
Loaded Saved Docker image
```
docker image load -i finrl.tar.gz
```

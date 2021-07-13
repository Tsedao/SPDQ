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
                --device=[cpu or gpu]
                --gpu=[which gpu to use]
```

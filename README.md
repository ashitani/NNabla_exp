# NNabla_exp

NNabla trial

Train exp(x) estimator and predict on c++

For datail, read [this Qiita Entry](https://qiita.com/ashitani/items/c853b3883bc62a52be97)

# requirements

NNabla (>=0.9.4)
NNabla C++ Library(>=0.9.4)

# Train on python

```
python train.py
```

Trained parameters are saved to exp_net.h5 and exp_net.nnp

# Replay on python

```
python replay.py
```
exp_net.h5 is used.

# Replay on C++

```
make
./exp_net
```
exp_net.nnp is used.

# License

Apache2.0 (following NNabla's license)


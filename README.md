# 2d-reconnection-data

![](reconnection_points.png)

## Requirements

Have analysator installed on path
```
cd $HOME
git clone https://github.com/fmihpc/analysator.git
export PYTHONPATH=$PYTHONPATH:$HOME/analysator
```

Python 3.6.8
```
pip install -r requirements.txt
```

## Run

Step 1.

Extract precalcualted reconnection points from the flux function
```
python flux_function.py -o x_points
```

Step 2.

Extract frames of correct size with chosen observables from bulk files
```
python extract_data.py -o frames
```

Step 3.

Combine states into series suitables for forecasting
```
python prepare_states.py -o series
```

# 2d-reconnection-data

![](reconnection_points.png)

## Requirements

```
cd $HOME
git clone https://github.com/fmihpc/analysator.git
pip install numpy scipy matplotlib Shapely
```

export PATH="$HOME/.local/bin:$PATH"

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

Combine states into datacubes suitables for forecasting
```
python prepare_states.py
```
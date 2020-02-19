# Assignment 3

- name: Alexander Cooper
- student ID: 20629774

## Dependencies

- json
- numpy
- matplotlib
- PyTorch

## Running `main.py`

To run `main.py`, use

```sh
python main.py /params/your_param_file.json
```

## Parameter file

You should include a json file of parameters for the program to use. Your parameter file should be of the following form. A default parameter file ```param.json``` is included.

```sh
{"epochs":10,
```
The number of training epochs

```sh
"display epochs":true,
```
Choice of displaying the current epoch when in high verbosity mode

```sh
"learning rate":0.05
```
The learning rate our model uses

```sh
"num test":500
```
The number of data points of set aside for testing


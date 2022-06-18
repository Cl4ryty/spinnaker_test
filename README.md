# Neurodynamics-Group5

## How to use
- clone this repository
- Install conda (e.g. https://www.anaconda.com/) if you don't have already
- create a new environment and install pip in it with `conda create -n env_name pip`, substituting env_name with a name of your liking
- activate the environment `conda activate env_name`
- navigate into the cloned repo
- Testing the SNN toolbox:
  - install the required libraries with `pip install -r requirements.txt`
  - run the `toolbox_test` to train the ann and convert it with the snn toolbox -> `python LibraryTests/toolbox_test.py`
- Running the ANN / adding DEs:
  -  no requirements.txt provided at the moment, just run it and install the missing packages it complains about

## Implementiing DEs for the ANN to solve
`run_ann_des.py` contains the class for simple implementation of DEs as well as some already implemented DEs as examples and the (not completely finished) code to train the ANN to solve the equations.
To add a DE create a new DE object and append it to the equations list. The code contains examples for creating DEs as well as comments about the parameters.
Things to note:
- only operations that can be applied to tensors should be used in the equation and solution functions (`+`, `-`, `*`, and `/` are fine to use, but for other operations use the tf version)
- the equation eq should have the parameters `df_dx`, (`df_dxx`, `df_d_xxx`, `df_d_xxxx`), `f`, `x`  – with df_dx, f, and x being required for all DEs and further derivatives only being required for the higher order DEs that use them. A third order DE would then have the parameters `df_dx, df_dxx, df_d_xxx, f, x`


## Useful links
- Nengo tutorial for converting ann to snn: https://www.nengo.ai/nengo-dl/v3.5.0/examples/keras-to-snn.html

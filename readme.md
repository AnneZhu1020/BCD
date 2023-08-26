First create an anaconda environment by
```bash
conda env create -f environment.yml
conda activate bcd
```

Then install the package that implements RL algorithms
```bash
cd minirl
pip install -e .
```

Run the code
```bash
python train_bcd.py
```

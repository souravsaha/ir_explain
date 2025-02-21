# ir_explain

## Installing requirements

```
git clone https://github.com/souravsaha/ir_explain.git
```
```
conda create -n ir_explain python=3.9
```
```
pip install .
```
## Install via PyPI:

```
pip3 install ir-explain
```

Note that for now we have tested ir_explain on Python 3.9 and Java 11. we will test on newer versions of Python soon and update.

Before running the ir_explain library, set the JAVA HOME as follows (replace the path appropriately):
```
export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64/"
```

## Usage

As of now, we have established a single pipeline with pointwise component. 

### To get started 

You can run the programs test_point_wise.py, test_pair_wise.py, and test_list_wise.py to get the pipeline. Run files for some neural ranking models are available in the `examples/runs folder`. 

## Contributing

IR Explain library is open-source, and we are open to all the contributions from IR and NLP community. If you find a bug please report to the issue tracker, even better to send us a pull-request on Github. 
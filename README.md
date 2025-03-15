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

## Usage

Before running the ir_explain library, set the JAVA HOME as follows (replace the path appropriately):
```
export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64/"
```
### To get started 

You can run the programs test_point_wise.py, test_pair_wise.py, and test_list_wise.py to get the pipeline. Run files for some neural ranking models are available in the `examples/runs folder`. 

## Roadmap
ir_explain is a open-source Python library for explaining IR methods. We will continute to update and add various explainable approaches. We also sincerely welcome contributions on this software. Our previous version of this toolkit can be found in https://github.com/souravsaha/ir_explain_old

- [x] Support for different Pointwise component
- [x] Support for different Pairwise axiomatic component
- [x] Support for different Listwise component. Currently four state-of-the-art listwise approaches are added. 
- [x] Support different evaluation component.
- [ ] Include an interpretable-by-design approach, SELECT-AND-RANK (https://dl.acm.org/doi/10.1145/3576924), which was published in TOIS 2023. 
- [ ] Include Probing for Dual Encoders (https://dl.acm.org/doi/abs/10.1145/3627673.3679556), published in CIKM 2024.
- [ ] Add more logging setup for better debugging.
- [ ] Enhance code adaptability and readability.

## Contributing

IR Explain library is open-source, and we are open to all the contributions from IR and NLP community. If you find a bug please report to the issue tracker, even better to send us a pull-request on Github. 
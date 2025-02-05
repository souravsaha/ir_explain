# irx

## Installing requirements

```
git clone https://github.com/souravsaha/irx.git
```
```
conda create -n irx python=3.9
```
```
pip install .
```
Note that for now we have tested irx on Python 3.9 and Java 11. we will test on newer versions of Python soon and update.

Before running the irx library, set the JAVA HOME as follows (replace the path appropriately):
```
JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64/"
```

## Usage

As of now, we have established a single pipeline with pointwise component. 

### To get started 

You can run the program test_point_wise.py to get the pipeline. 

## Contributing

IR Explain library is open-source, and we are open to all the contributions from IR and NLP community. If you find a bug please report to the issue tracker, even better to send us a pull-request on Github. 
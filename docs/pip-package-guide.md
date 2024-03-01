# PIP package Guide

## Building and installing

`pip install .` is mostly sufficient for most users. It will build `Release` version of the packages.

```bash
pip install .
```

To print with more information:

```bash
pip install -vv .
```

## Build only

To build the code only without installing:

```bash
python setup.py build_ext
```

## To build in Debug mode

Building Debug mode requires extra parameter:

```bash
python setup.py build_ext --debug
```

## Build wheel

To build the Release version of the wheel:

```bash
python setup.py bdist_wheel
```

Then you can inspect the wheel by:

```bash
unzip dist/*.whl
```

## Build Debug version of wheel

Debug version of the wheel requires extra steps:

```bash
python setup.py build_ext --debug
python setup.py bdist_wheel
```

## Clean build

Clean the build environment

```bash
python setup.py clean
```
### Workload File Format Description

!!! Note that there must be an empty line at the start and end of the file !!!

- The files in this directory are workload files for PNMulator, used to simulate NN
- The file format is csv, with each row representing a network layer, formatted as:
    - Name
    - Operator type
    - input size
    - shared dimension
    - weight size
    - batch size
- Example resnet18_32.csv is the cifar10 version of resnet18, with an input size of 32x32
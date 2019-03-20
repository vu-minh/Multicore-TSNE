# How to build:

+ In dev mode:
    - `cd Multicore-TSNE`

    - Build C++ extention with cmake
    ```
    cmake multicore_tsne/
    make .
    ```
    
    - If everything is ok, `libtsne_multicore_minh.so` will appear in the root dir.

    - place the python test code in the root dir to directly import python wrapper from `MulticoreTSNE` dir. E.g., see ![tsne-embedding.py](./tsne-embedding.py).


+ In production mode:
    - install the package system-wide: `pip install .` when everything is well compiled
    - or `python setup.py install`
    - if ok, everything will be in
    ```
    anaconda3/lib/python3.6/site-packages/MulticoreTSNE-0.2.dist-info/*
    anaconda3/lib/python3.6/site-packages/MulticoreTSNE/*
    ```
    - test the installed package with the script which imports the 'global' system-wide `MulticoreTSNE` package, see [./MulticoreTSNE/test/test_installed.py](./MulticoreTSNE/test/test_installed.py)
 



---

<div align="center">    
 
# HeavyBuilder2: Analysis of High-Throughput of Antibody Heavy Chain Repertoires in the Structural Space

</div>

---

## Abstract

Most antibody sequence data that’s publicly available comes from heavy chains, since they’re cheaper and easier to sequence than paired heavy–light chains. However, studying the 3D structure of these heavy chains at scale has been difficult — existing methods are often too slow or too resource-intensive. Fluft is a deep learning tool that makes this process fast and reliable. Built on the igfold framework, it can predict up to one million heavy chain structures in just over three days on a single GPU. It outperforms AlphaFold2 and IgFold, running much faster while maintaining accuracy. This makes it a practical solution for immune repertoire scale studies, with potential to speed up antibody discovery and analysis.


## Install

### Requirements

This package requires PyTorch. If you do not already have PyTorch installed, you can do so following these <a href="https://pytorch.org/get-started/locally/">instructions</a>.

It also requires OpenMM and pdbfixer for the refinement step. For details on how to install OpenMM please follow these <a href="http://docs.openmm.org/latest/userguide/application/01_getting_started.html#installing-openmm">instructions</a>.  
Alternatively, OpenMM and pdbfixer can be installed via conda using:

```bash
$ conda install -c conda-forge openmm pdbfixer
```

It also uses anarci for trimming and numbering sequences. We recommend installing ANARCI from <a href="https://github.com/oxpig/ANARCI/tree/master">here</a>, but it can also be installed using (maintained by a third party):

```bash
$ conda install -c bioconda anarci
```

### Install HeavyBUilder

Once you have all dependencies installed within one enviroment, you can install ImmuneBuilder via pip by doing:

```bash
$ git clone https://github.com/jao321/HeavyBuilder.git
$ pip install HeavyBuilder/
```

## Usage

### Antibody heavy chain structure prediction

To predict an antibody heavy chain structure using the python API you can do the following.

```python
from HeavyBuilder import HeavyBuilder2
predictor = HeavyBuilder2()

output_file = "my_antibody_heavy_chain.pdb"
sequences = {
  'H': 'EVQLVESGGGVVQPGGSLRLSCAASGFTFNSYGMHWVRQAPGKGLEWVAFIRYDGGNKYYADSVKGRFTISRDNSKNTLYLQMKSLRAEDTAVYYCANLKDSRYSGSYYDYWGQGTLVTVS'}

antibody = predictor.predict(sequences)
antibody.save(output_file)
```

HeavyBuilder2 can also be used via de command line. To do this you can use:

```bash
HeavyBuilder2 --fasta_file my_antibody_heavy_chain.fasta -v
```

You can get information about different options by using:

```bash
HeavyBuilder2 --help
```

I would recommend using the python API if you intend to predict many structures as you only have to load the models once.

If you want to speed up the prediction

```python
antibody.save_single_unrefined(output_file)
```

### Repertoire wide prediction and structure clustering

In case you want to predict heavy chain structures for sequences from a repertoire, here we have an example using the MiAIRR data format.

The toy example data is in the file example_data_HB_repo.tsv in the folder example_data/

This is mostly recommended for systems with CUDA ready graphics card.

In python:
```python
'''

Tutorial on how to predict and cluster heavy chains structure for a toy antibody repertoire

'''
import os
from HeavyBuilder import HeavyBuilder2
import pandas as pd

predictor = HeavyBuilder2()

#read the repertoire data
example = pd.read_csv('example_data_HB_repo.tsv',
                      sep='\t')

#create an output folder where the pdb files will be
output_folder = 'heavybuilder_example_pdb_files'
os.mkdir(output_folder)

#loop through the amino acid sequences of the heavy chain and predict their 3D structure
for i in range(len(example)):
    data = example.iloc[i][['sequence_id','sequence_aa']]
    output_file = os.path.join(output_folder,data.sequence_id+'.pdb')
    sequences= {
        'H':data.sequence_aa
    }
    try:
        antibody = predictor.predict(sequences)
        antibody.save_single_unrefined(output_file)
    except:
        continue
```

It should take some time to predict the ~2400 structures.

Now you can cluster them to check for structural similitude.

First you will need to install <a href="https://github.com/oxpig/SPACE2">SPACE2</a>

In a terminal you can clone SPACE2 repository and install in your python environment via pip

```bash
git clone https://github.com/fspoendlin/SPACE2.git
pip install SPACE2/
```

Back to python:
```python
import glob
import SPACE2

#list pdb files previously predicted
heavy_chain_models = glob.glob('heavybuilder_example_pdb_files/*pdb')

#cluster them together
clustered_dataframe = SPACE2.agglomerative_clustering(heavy_chain_models, cutoff=1.25, n_jobs=-1)

#save clustering results to a csv file
clustered_dataframe.to_csv('clustered_heavybuilder_example.csv',
                           index=False)
```

Happy antibodies!!

### Fasta formatting

If you wish to run the model on a sequence from a fasta file it must be formatted as follows:

```
>H
YOURHEAVYCHAINSEQUENCE
>L
YOURLIGHCHAINSEQUENCE
```

## Issues and Pull requests

Please submit issues and pull requests on this <a href="https://github.com/jao321/HeavyBuilder">repo</a>.

### Known issues

- Installing OpenMM from conda will automatically download the latest version of cudatoolkit which may not be compatible with your device. For more information on this please checkout the following <a href="https://github.com/brennanaba/ImmuneBuilder/issues/13">issue</a>.
- After following install instructions I get an ```Import Error: `GLIBCXX_3.4.30' not found```. This is an issue with OpenMM, and can be solved by doing ```conda install -c conda-forge libstdcxx-ng```. See issue <a href="https://github.com/openmm/openmm/issues/3943">here</a>.


## Citing this work

TBA

```tex
TBA
```


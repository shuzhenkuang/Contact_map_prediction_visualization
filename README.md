## Contact map prediction and visualization with Akita
The code here is for predicting and visualizing the contact map changes caused by sequence variants using Akita, which is a convolutional neural network based deep learning model to predict 3D genome folding from DNA sequence alone.

### Installation
The installation of basenji/Akita could be found at :[https://github.com/calico/basenji/tree/master].

### Instructions
#### Files required
- human genome fastq file (hg38)
- Pretrained Akita model: Two cell-type specific models have been trained on H1-hESC and HFFc6 Micro-C data using the Akita architecture, respectively. The pretrained model could be found under Data.
- Parameter file that was used to train Akita: The parameter file is available under Data.
- Bed file with the genomic coordinates of deletions or motifs to be mutated: The code is used to predict the genome folding changes caused by deletions or mutation of motifs. Example regions could be found under Data.

#### Example code for deletions
```
python Code/predictions_visualization.py -f hg38.fa -m Data/HFF_model.h5 -p Data/params.json -b Data/del_test_regions.bed -t del -o Data
```

### Example code for in silico mutation of transcription factor motifs 
```
python Code/predictions_visualization.py -f hg38.fa -m Data/HFF_model.h5 -p Data/params.json -b Data/mut_test_regions.bed -t mut -o Data
```

### Contact
If you have any questions, please feel free to contact shuzhen.kuang@gladstone.ucsf.edu.


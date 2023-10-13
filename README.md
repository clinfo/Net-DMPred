# Net-DMPred : Network-based driver mutation prediction


## Structure
The following tree describes the expected initial file sctructure:
```
.
├── data
│   ├── raw
│   └── ready
├── lib
│   ├── engine
│   ├── models
│   └── utils
└── scripts
    └── utils
````

## Datasets Used
__Pathway Commons__:
https://www.pathwaycommons.org/archives/PC2/v12/PathwayCommons12.All.hgnc.sif.gz  
Rodchenkov, I. et al. (2020) Pathway Commons 2019 Update: integration, analysis and exploration of pathway data. Nucleic Acids Res, 48, D489-D497.

__Gene symbol To HGNC__:
https://www.genenames.org/cgi-bin/download/custom?col=gd_hgnc_id&col=gd_app_sym&col=g[…]_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit

__Training dataset__:
http://karchinlab.org/data/CHASMplus/formatted_training_list.txt.gz  
__Benchmark dataset__:
http://karchinlab.org/data/CHASMplus/Tokheim_Cell_Systems_2019.tar.gz  
Tokheim, C. and Karchin, R. (2019) CHASMplus Reveals the Scope of Somatic Missense Mutations Driving Human Cancers. Cell Syst, 9, 9-23.e8.

__SNVBox__:
http://karchinlab.org/data/CHASMplus/SNVBox_chasmplus.sql.gz  
Wong, W.C. et al. (2011) CHASM and SNVBox: toolkit for detecting biologically important single nucleotide mutations in cancer. Bioinformatics, 27, 2147-2148.

__HotMAPS1D__:
https://github.com/KarchinLab/probabilistic2020  
Tokheim, C. et al. (2016) Exome-Scale Discovery of Hotspot Mutation Regions in Human Cancer Using 3D Protein Structure. Cancer Res, 76, 3719-3731.


## Reference
```
@article {Net-DMPred,
        author = {Hatano, Narumi and Kamada, Mayumi and Kojima, Ryosuke and Okuno, Yasushi},
        title = {Network-based prediction approach for cancer-specific driver missense mutations using a graph neural network},
        year = {2023},
        doi = {10.1186/s12859-023-05507-6},
        eprint = {https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05507-6},
        journal = {BMC Bioinformatics}
}
```

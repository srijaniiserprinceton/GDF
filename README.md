GDF
===
**Slepian based reconstruction of solar wind particle distributions**

 [![name](https://img.shields.io/badge/GitHub-srijaniiserprinceton%2FGDF-blue.svg?style=flat)]([https://github.com/srijaniiserprinceton/GDF])

**GDF** or Gyrotorpic Distribution Functions are a commonly used approximation for solar wind velocity distribution functions (VDFs).
This repository uses Slepian basis function based reconstructions to recover distribution functions from Parker Solar Probe's SPAN-Ai
instrument. This repository is an extension of [vdfit](https://github.com/srijaniiserprinceton/VDF_paper1_ESA) which was designed for 
reconstructing agyritropic *MMS* and *Solar Orbiter* VDFs to the PSP measurements.

>[!NOTE]
>This project is under active development. The next goal is to integrate Solar Probe Cup measurements to jointly constraint the GDF reconstructions.

Attribution
-----------

Please cite [Bharati Das & Terres (2025)](https://ui.adsabs.harvard.edu/abs/2025ApJ...982...96B/abstract) if you find this code useful in your
research. The BibTeX entry for the paper is:

```
@ARTICLE{2025ApJ...982...96B,
       author = {{Bharati Das}, Srijan and {Terres}, Michael},
        title = "{Recovering Ion Distribution Functions. I. Slepian Reconstruction of Velocity Distribution Functions from MMS and Solar Orbiter}",
      journal = {\apj},
     keywords = {Space plasmas, Solar wind, Regression, Solar instruments, 1544, 1534, 1914, 1499, Astrophysics - Solar and Stellar Astrophysics, Physics - Space Physics},
         year = 2025,
        month = apr,
       volume = {982},
       number = {2},
          eid = {96},
        pages = {96},
          doi = {10.3847/1538-4357/adb6a0},
archivePrefix = {arXiv},
       eprint = {2501.17294},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJ...982...96B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


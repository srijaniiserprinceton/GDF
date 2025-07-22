.. _install:

*************
Installation
*************
Clone the `gdf <https://github.com/srijaniiserprinceton/GDF>`_ github
repository to your local system directory using

.. code-block:: bash

   git clone https://github.com/srijaniiserprinceton/GDF.git

Installing a local Python environment
=====================================
The requirements can be found in ``gdf_env.yml`` file. 
If you already have a local Python environment containing all the
requirements, you can skip this step and move onto :ref:`gdf installation <gdf_installation>`.
If not, it is advisable to install the Python environment from the
``gdf_env.yml`` file. We have a default name ``gdf`` 
assigned to this environment but you can change the name in the
first line of the file.

.. code-block:: yaml

   name: <your-env>

Next, setup your Python environment using

.. code-block:: bash
   
   conda env create -f gdf_env.yml

The rest of the documentation is written assuming you are in this 
Python environment. In order to activate ``<your-env>`` execute in your
terminal

.. code-block:: bash

   conda activate <your-env>
   (your-env) gdf $

At this point, you are ready to install the ``gdf`` package in your 
Python environment.

.. _gdf_installation:

Installing the ``gdf`` repository
=================================
The cloned ``gdf`` directory can be installed as a Python package
in your local environment. Move into the cloned repository 
directory: ``cd GDF``. In order to install the ``gdf`` Python package simply

.. code-block:: bash

   pip install -e .

In order to test if the package is installed correctly, check if you can
import the ``gdf`` repository from inside a Python console or iPython
instance

.. code-block:: python
   
   import gdf

.. _matlab_slepian_installation:

Installing the Slepian packages (Matlab)
========================================

The final step involves installing the Slepian repositories developed in 
Matlab. A list of all the Slepian repositories can be found `here <https://geoweb.princeton.edu/people/simons/software.html>`_. 
For our purposes, we only require two packages `slepian_alpha <https://github.com/csdms-contrib/slepian_alpha>`_ and 
`slepian_foxtrot <https://github.com/csdms-contrib/slepian_foxtrot>`_. Since these packages are written in Matlab, it is 
expected that the user would have Matlab (version > Matlab_R2024a) installed on their system. It is advisable to keep the Slepian
repositories outside the `gdf` repository.

We assume that you have a directory ``Slepians`` outside the ``gdf`` directory. Once you are in the ``Slepians`` directory,
clone the repositories as follows

.. code-block::
   
   git clone https://github.com/csdms-contrib/slepian_alpha.git
   git clone https://github.com/csdms-contrib/slepian_foxtrot.git

Finally, since we run Matlab from inside our Python codebase by using `matlabengine <https://pypi.org/project/matlabengine/>`_,
we require a ``.config`` file which contains the absolute path to the ``Slepians`` directory. In order to access the proprietary data,
the user would need a ``config.json`` file in your ``gdf`` repository. This file should look like

.. code-block::

   {   
   "psp" : {
      "fields" : {
         "username": "<your-fields-username>", 
         "password" : "<your-fields-password>"
            },
      "sweap" : {
         "username" : "<your-sweap-username>", 
         "password" : "<your-sweap-password>"
            }
      }
   }


Building the repository structure and unit test
===============================================
After downloading the Matlab repositories, the last thing we need to do is to make the structure of the ``gdf`` repository. 
This can be done by executingthe following make file in the ``setup`` mode

.. code-block::

   make setup

Once this is run, you should have a directory structure as shown below (assuming you have downloaded a Slepians in a different 
directory than the ``gdf`` repository).

Directory structure::

    gdf/
    ├── main.py
    ├── init_gdf_default.py
    ├── src/
    │   └── ...
    ├── Outputs/
    │   └── ...
    └── Figures/
        ├── mcmc_dists_polcap/
        ├── span_rec_polcap/
        ├── super_res_polcap/
        ├── kneeL_polcap/
        ├── cartesian_slepians/
        ├── super_res_cartesian/
        └── super_res_hybrid/

    Slepians/
    ├── slepian_alpha/
    │   └── ...
    ├── slepian_foxtrot/
    │   └── ...
    └── IFILES/
        ├── LEGENDRE/
        └── SDWCAP/

This should setup the required directory structure required. Finally, in order to test the installation run the 
makefile in the ``testrun`` mode.

.. code-block::

   make testrun

If the installation is successful, this should go through without errors. Check the final generated figures in the directory ``Figures``. 

Setting up your own GDF run directory
=====================================

The section above laid out the installation and test run for a single gdf run on a template initialization file.
It is recommended to create a separate directory for each GDF time range you are interested in. Let's say that
you want to run GDF for the interval ``TRANGE = ['2020-01-26T14:28:00', '2020-01-26T20:30:59']``. So, you might
have a directory called ``gdfrun_2020_01_26T14_28_00__2020_01_26T20_30_59`` somewhere outside the actual ``gdf``
repository directory. Copy the following files into this directory from the ``gdf`` repository 

.. code-block::

   cp GDF/main.py gdfrun_2020_01_26T14_28_00__2020_01_26T20_30_59/.
   cp GDF/Makefile gdfrun_2020_01_26T14_28_00__2020_01_26T20_30_59/.
   cp GDF/init_gdf_default.py gdfrun_2020_01_26T14_28_00__2020_01_26T20_30_59/.
   cp GDF/.config gdfrun_2020_01_26T14_28_00__2020_01_26T20_30_59/.
   cp GDF/config.json gdfrun_2020_01_26T14_28_00__2020_01_26T20_30_59/.

Once you have copied these files, setup the directory structure for storing output pickle files and figures by running

.. code-block::

   cd gdfrun_2020_01_26T14_28_00__2020_01_26T20_30_59
   (your-env) gdfrun_2020_01_26T14_28_00__2020_01_26T20_30_59 $ make setup_new_gdfdir

At this point, the directory structure should look like this

.. code-block::

   gdf/

   Slepians/

   gdfrun_2020_01_26T14_28_00__2020_01_26T20_30_59/
   ├── main.py
   ├── init_gdf_default.py
   ├── Makefile
   ├── .config
   ├── config.json
   ├── Outputs/
   │   └── ...
   └── Figures/
       ├── mcmc_dists_polcap/
       ├── span_rec_polcap/
       ├── super_res_polcap/
       ├── kneeL_polcap/
       ├── cartesian_slepians/
       ├── super_res_cartesian/
       └── super_res_hybrid/

Now, you are setup to change ``init_gdf_default.py`` to your own requirements and simply run 
``python main.py init_gdf_default`` in your newly created gdf run directory.
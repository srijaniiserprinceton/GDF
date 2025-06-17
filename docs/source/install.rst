.. _install:

*************
Installation
*************

Installing a local Python environment
=====================================
The requirements can be found in ``pyspedas_env.yml`` file. 
If you already have a local Python environment containing all the
requirements, you can skip this step and move onto :ref:`gdf installation <gdf_installation>`.
If not, it is advisable to install the Python environment from the
``pyspedas_env.yml`` file. We have a default name ``pyspedas`` 
assigned to this environment but you can change the name in the
first line of the file.

.. code-block:: yaml

   name: <your-env>

Next, setup your Python environment using

.. code-block:: bash
   
   conda env create -f pyspedas_env.yml

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
Clone the `gdf <https://github.com/srijaniiserprinceton/GDF>`_ github
repository to your local system directory using

.. code-block:: bash

   git clone https://github.com/srijaniiserprinceton/GDF.git

The cloned ``GDF`` directory can be installed as a Python package
in your local environment. Move into the cloned repository 
directory: ``cd GDF``. In order to install the ``gdf`` Python package simply

.. code-block:: bash

   pip install -e .

In order to test if the package is installed correctly, check if you can
import the ``gdf`` repository from inside a Python console or iPython
instance

.. code-block:: python
   
   import gdf


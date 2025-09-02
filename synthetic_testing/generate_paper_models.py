import synthetic_models as synth
import numpy as np

model_params = synth.model_params()
model_grids  = synth.model_grids()

model = synth.synthetic_models(*model_params)

model.define_inst_grids(*model_grids)

bvecs = np.array([0.9807312 , -0.17285955, -0.09102581]) * np.ones(2)[:,None]
uvecs = np.array([-433.50574,  107.61503,   32.554  ]) * np.ones(2)[:,None]

model.define_model_vectors(bvecs, uvecs)

model.tag = 'paper_models'

model.evaluate_bimax_on_inst_grid()

model.save_cdf()



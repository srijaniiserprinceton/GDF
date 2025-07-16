# genrating a new suepr resolved grid for the polar cap to match the data points on the Cartesian
boundary_points = CartSlep_lr.XY * 1.0
eval_gridx = np.linspace(boundary_points[:,0].min(), boundary_points[:,0].max(), 49)
# eval_gridx = np.linspace(0.1, boundary_points[:,0].max(), 49)
eval_gridy = np.linspace(boundary_points[:,1].min(), boundary_points[:,1].max(), 49)
vdf_inv, _, vdf_super, data_misfit, model_misfit, knee_idx = gvdf_tstamp.inversion(gvdf_tstamp.ubulk, gvdf_tstamp.vdfdata, tidx,
                                                                                    SUPER=True, NPTS=None, grid_x=eval_gridy, grid_y=eval_gridx)

# trying the joint fitting
A = gvdf_tstamp.G_k_n.T
Af = gvdf_tstamp.super_G_k_n.T
B = CartSlep_lr.G
Bf = CartSlep_hr.G
ndata_A, nparams_A = A.shape
ndata_B, nparams_B = B.shape
nf = Af.shape[0]

# transposing Bf
BfT = np.reshape(Bf, (int(np.sqrt(nf)),int(np.sqrt(nf)),nparams_B))
BfT = np.transpose(BfT, [1,0,2])
Bf = np.reshape(BfT, (-1,nparams_B))

# creating the G matrix
G = np.zeros((ndata_A + ndata_B + nf, nparams_A + nparams_B))

lam = 1e-1
# filling in A
G[:ndata_A, :nparams_A] += A
# filling in B
G[ndata_A:ndata_A+ndata_B, nparams_A:nparams_A+nparams_B] = B
#filling in the (A(m1) - B(m2)) term
G[ndata_A+ndata_B:ndata_A+ndata_B+nf, :nparams_A] = np.sqrt(lam) * Af
G[ndata_A+ndata_B:ndata_A+ndata_B+nf, nparams_A:nparams_A+nparams_B] = -np.sqrt(lam) * Bf

# creating the augmented data matrix
d = np.zeros((ndata_A + ndata_B + nf))
d[:ndata_A] = gvdf_tstamp.vdfdata
d[ndata_A:ndata_A+ndata_B] = vdf_data

# calculating the coefficients
GT = G.T
GTG = GT @ G
GTd = GT @ d 

# coefficients (mA + mB)
m = np.linalg.inv(GTG) @ GTd

mA, mB = m[:nparams_A], m[nparams_A:]

# reconstructing the two models
vdfrec_A = Af @ mA
vdfrec_B = Bf @ mB

# converting the VDFs to SPAN-i consistent units
f_supres_A = np.power(10, vdfrec_A) * gvdf_tstamp.minval[tidx]
f_supres_B = np.power(10, vdfrec_B) * gvdf_tstamp.minval[tidx]
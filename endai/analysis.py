import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

from sklearn.decomposition import PCA
import umap

out = data_rec_dic["output_rec"]
hid = data_rec_dic["hidden_rec"]

pos_rec = data_rec_dic["pos_rec"]

Wi = ag.model.W_sens_hidden.detach().cpu().numpy()
Wo = ag.model.W_hidden_out.detach().cpu().numpy()
Wh = ag.model.W_hidden.detach().cpu().numpy()

lim1=0
lim2=1000

plt.plot(np.transpose(hid[:,lim1:lim2]))

fig = plt.figure(figsize=(5, 4), dpi=140)
font = {'family' : 'sans-serif',
        'weight' : 'light',
        'size'   : 10}
matplotlib.rc('font', **font)

lim1=100
lim2=1000

plt.matshow(out[:,lim1:lim2], aspect = 'auto')

plt.matshow(hid[:,lim1:lim2], aspect = 'auto')

# get UMAPs

reducer = umap.UMAP(
    #n_neighbors=350,
    n_neighbors=550,
    min_dist=0.7,
    n_components=2,
)

lim_1 = 0
lim_2 = 1000

#X = np.transpose(out[:,0:500])
Xh = np.transpose(hid[:,lim_1:lim_2])
Xo = np.transpose(out[:,lim_1:lim_2])

umap_emb_hid  = reducer.fit_transform(Xh)
umap_emb_out  = reducer.fit_transform(Xo)

ind=0
color = pos_rec[ind,:]

color = (color - np.min(color)) / (np.max(color) - np.min(color))

plt.plot(umap_emb_hid[:,0],umap_emb_hid[:,1],'k',alpha=0.2)

plt.scatter(umap_emb_hid[:,0], umap_emb_hid[:,1], s=15,marker='o',
           c= color, cmap='hot')

ind=0
color = pos_rec[ind,:]

color = (color - np.min(color)) / (np.max(color) - np.min(color))

plt.plot(umap_emb_out[:,0],umap_emb_out[:,1],'k',alpha=0.2)

plt.scatter(umap_emb_out[:,0], umap_emb_out[:,1], s=15,marker='o',
           c= color, cmap='hot')


# PCA of neural activity

pca = PCA(n_components=80)
projected_A = pca.fit_transform(Xh)  # Transpose to get time points as samples (1000x3)
projected_A = projected_A.T

# Get eigenvalues (explained variance)
eigenvalues = pca.explained_variance_

# Plot eigenvalues (scree plot)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue (Explained Variance)")
plt.title("Scree Plot of Eigenvalues")
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(5, 4), dpi=140)
font = {'family' : 'sans-serif',
        'weight' : 'light',
        'size'   : 10}
matplotlib.rc('font', **font)

lim1=100
lim2=600

plt.matshow(projected_A.T[0:8,lim1:lim2], aspect = 'auto')

projected_A = projected_A.T
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot(projected_A[:, 0], projected_A[:, 1], projected_A[:, 2], color='b', alpha=0.7)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Neural Trajectory in PCA Space")

ind=0
#color = hid[ind,:]

color = projected_A[:,ind]

color = (color - np.min(color)) / (np.max(color) - np.min(color))

plt.plot(pos_rec[:,0],pos_rec[:,1],'k',alpha=0.2)

plt.scatter(pos_rec[0,:], pos_rec[1,:], s=15,marker='o',
           c= color, cmap='hot')

eigvals = np.linalg.eigvals(Wh)

# Magnitudes (for stability analysis)
eig_mags = np.abs(eigvals)

# Print summary
print("Eigenvalues of W_hidden:", eigvals)
print("Max eigenvalue magnitude:", np.max(eig_mags))
print("Min eigenvalue magnitude:", np.min(eig_mags))

plt.figure(figsize=(6, 6))
plt.scatter(eigvals.real, eigvals.imag, color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('Eigenvalues of W_hidden on Complex Plane')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.show()


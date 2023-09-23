import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import kn
from sklearn.metrics import pairwise_distances
from cprint import cprint


def pdist2(x, y):
    return pairwise_distances(x, y)


def besselk(order, arg):
    return kn(order, arg)


def whittle(x1, x2, char_length, eps=np.spacing(1)):
    pdists = pdist2(x1, x2)
    b = besselk(1, eps + (char_length * pdists))
    return np.multiply(np.divide(pdists, 2.0 * char_length), b)


def matern_3_2(x1, x2, char_length, eps=np.spacing(1)):
    t0 = char_length * pdist2(x1, x2)
    t1 = 1.0 + t0
    t2 = np.exp(-1.0 * t0)
    return t1 * t2


# observations of a circle
circle_radius = 5.0
circle_res = 0.01
circle_center = np.array([0., 0.])
thetas = np.arange(-np.pi, np.pi, circle_res)
circle = np.add(circle_center.T, circle_radius *
                np.vstack((np.cos(thetas), np.sin(thetas))).T)
N_obs = circle.shape[0]

# noise in the observations
noise = 0.001

# query points for the distance field
[X, Y] = np.meshgrid(np.arange(-10.0, 10.0, 0.1),
                     np.arange(-10.0, 10.0, 0.1), indexing='xy')
Qpoint = np.zeros((X.shape[0] * X.shape[1], 2))
Qpoint[:, 0] = X.ravel()
Qpoint[:, 1] = Y.ravel()

# Log-GPIS hyperparameters
char_length = 40  # Lambda

# K
K1 = whittle(circle, circle, char_length)
K2 = matern_3_2(circle, circle, char_length)

# k_*
k1 = whittle(Qpoint, circle, char_length)
k2 = matern_3_2(Qpoint, circle, char_length)

# y
y = 1.0 + noise * np.random.randn(N_obs, 1)

# \bar{f}_*
mu1 = np.matmul(k1, np.matmul(np.linalg.inv(K1 + noise * np.eye(N_obs)), y))
mu2 = np.matmul(k2, np.matmul(np.linalg.inv(K2 + noise * np.eye(N_obs)), y))

# \bar{d}_*
pred_dist1 = (-1.0 / char_length) * np.log(np.abs(mu1))
pred_dist2 = (-1.0 / char_length) * np.log(np.abs(mu2))

# error in dist estimation
gt_dists_edf = np.abs(
    np.sqrt(np.power(X, 2.0) + np.power(Y, 2.0)) - circle_radius)
err1_edf = gt_dists_edf - pred_dist1.reshape(X.shape)
err2_edf = gt_dists_edf - pred_dist2.reshape(X.shape)
rmse1_edf = np.sqrt(
    np.mean(np.multiply(err1_edf.flatten(), err1_edf.flatten())))
rmse2_edf = np.sqrt(
    np.mean(np.multiply(err2_edf.flatten(), err2_edf.flatten())))

# printing some stats
cprint.info('RMSE EDF Whittle with λ = %d: %f' % (char_length, rmse1_edf))
cprint.info('RMSE EDF Matern 3/2 with λ = %d: %f' % (char_length, rmse2_edf))

# plotting everything
minmin = np.min([np.min(err1_edf), np.min(err2_edf)])
maxmax = np.max([np.max(err1_edf), np.max(err2_edf)])

edf_fig, edf_ax = plt.subplots(1, 2)
edf_ax[0].imshow(err1_edf, vmin=minmin, vmax=maxmax, cmap=cm.Greys, interpolation='none', extent=[-10, 10, -10, 10],
                 aspect='equal')
pos = edf_ax[1].imshow(err2_edf, vmin=minmin, vmax=maxmax, cmap=cm.Greys, interpolation='none', extent=[-10, 10, -10, 10],
                       aspect='equal')

edf_fig.subplots_adjust(right=0.8)
edf_cbar_ax = edf_fig.add_axes([0.85, 0.15, 0.05, 0.7])
edf_cbar = edf_fig.colorbar(pos, cax=edf_cbar_ax)

edf_ax[0].set_title('Whittle')
edf_ax[1].set_title('Matern 3/2')

plt.show()

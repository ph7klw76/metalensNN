import numpy as np
from fiber_metalens_nn.collimation_metrics import fit_angular_gaussian


def test_fitted_divergence_recovers_gaussian_with_weak_wide_angle_halo():
    theta=np.linspace(-1,1,1001)
    # A separated weak halo changes the global second moment, while the
    # explicitly defined Gaussian core retains its known 0.1-radian width.
    intensity=np.exp(-2*(theta/.1)**2)+1e-4*np.exp(-2*((abs(theta)-.6)/.05)**2)
    result=fit_angular_gaussian(theta,intensity)
    assert result["valid"]
    assert abs(result["half_angle_rad"]-.1)<1e-8


def test_divergence_fit_rejects_non_gaussian_top_hat_and_under_sampling():
    theta=np.linspace(-1,1,1001)
    intensity=(abs(theta)<.1).astype(float)
    assert not fit_angular_gaussian(theta,intensity)["valid"]
    theta=np.array([-.2,0,.2])
    assert not fit_angular_gaussian(theta,np.array([.01,1,.01]))["valid"]

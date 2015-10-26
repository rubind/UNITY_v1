import pystan
import cPickle as pickle
import numpy as np


def init_fn():
    n_sne = stan_data["n_sne"]
    n_samples = stan_data["n_samples"] 
            
    three_simplex = np.random.random(size = 3)
    three_simplex /= sum(three_simplex)

    obs_x1s = np.array([item[1] for item in stan_data["obs_mBx1c"]])
    obs_colors = np.array([item[2] for item in stan_data["obs_mBx1c"]])

    return {"MB": np.random.random()*0.2 - 19.1,
            "Om": np.random.random()*0.4 + 0.1,
            "alpha_angle_low": np.arctan(np.random.random()*0.2),
            "alpha_angle_high": np.arctan(np.random.random()*0.2),
            "beta_angle_blue": np.arctan(np.random.random()*0.5 + 2.5),
            "beta_angle_red": np.arctan(np.random.random()*0.5 + 2.5),
            "delta_0": np.random.random()*0.05,
            "delta_h": 0.5,
            "calibs": np.random.normal(size = stan_data["n_calib"])*0.5,
            
            "log10_sigma_int": np.log10(np.random.random(size = n_samples)*0.1 + 0.1),
            "mBx1c_int_variance": three_simplex, 
            "Lmat": [[1.0, 0.0, 0.0],
                     [np.random.random()*0.1 - 0.05, np.random.random()*0.1 + 0.7, 0.0],
                     [np.random.random()*0.1 - 0.05, np.random.random()*0.1 - 0.05, np.random.random()*0.1 + 0.7]],


            "true_c": np.random.random(size = n_sne)*0.02 - 0.01 + np.clip(obs_colors, -0.2, 1.0),
            "true_x1": np.random.random(size = n_sne)*0.2 - 0.1 + obs_x1s,
            
            "x1_star": np.random.random(size = [n_samples, stan_data["n_x1c_star"]])*0.05,
            "c_star": np.random.random(size = [n_samples, stan_data["n_x1c_star"]])*0.05,
            "delta_c": np.random.random(size = [n_samples, stan_data["n_x1c_star"]])*0.2 - 0.1,
            "log10_R_x1": np.random.random(size = n_samples)*0.5 - 0.25,
            "log10_R_c": np.random.random(size = n_samples)*0.4 - 1.2,

            "outl_frac": np.random.random()*0.02 + 0.01
        }
            
            

################################################# Main Program ###################################################


stan_data = pickle.load(open("stan_data.pickle", 'rb'))

fit = pystan.stan(file="stan_code.txt", data=stan_data,
                  iter=50, chains=1, n_jobs = 1, init = init_fn)

print fit


import numpy as np
import os


def generate_time_series_data(
    system_params: dict,
    timesteps: int,
    output_path: str,
    base_filename: str,
    noise_config: dict,
    seed: int = None
):
    """
    Generates time series data for a linear system x(k+1) = A*x(k) + B*u(k) + w(k).

    Args:
        system_params (dict): Dictionary with system parameters, e.g., {'a': 0.5, 'b': 0.5}.
        timesteps (int): The number of timesteps to generate.
        output_path (str): The file path to save the generated data.
        base_filename (str): The base name for the output files, e.g., 'simulation_run_1'.
        noise_config (dict): Configuration for the noise, 
                             e.g., {'distribution': 'uniform', 'level': 0.01} or
                             {'distribution': 'gaussian', 'std_dev': 0.01}.
        seed (int, optional): A seed for the random number generator for reproducibility.
    """
    # 1. Initialize the random number generator 
    rng = np.random.default_rng(seed)

    # 2. Unpack system parameters
    a = system_params.get('a', 0.5) #defaults to 0.5 if no input is given
    b = system_params.get('b', 0.5)
    initial_state = 0 

    A = np.array([[a]])
    B = np.array([[b]])
    
    # 3. Generate input and noise signals based on configuration
    input_signal = rng.standard_normal((1, timesteps))  # Input signal, normal distributed (mean=0, std=1)
    noise_dist = noise_config.get('distribution')
    if noise_dist == 'uniform':
        level = noise_config.get('level', 0.01) #defaults to 0.01
        noise_signal = rng.uniform(-level, level, (1, timesteps))
    elif noise_dist == 'gaussian':
        std_dev = noise_config.get('std_dev', 0.01) #defaults to 0.01
        noise_signal = rng.normal(0, std_dev, (1, timesteps))
    else:
        raise ValueError(f"Unknown noise distribution: {noise_dist}")

    # 4. Simulation loop
    state_vector = np.zeros((1, timesteps + 1))
    state_vector[:, 0] = initial_state

    for k in range(timesteps):
        state_vector[:, k+1] = A @ state_vector[:, k] + B @ input_signal[:, k] + noise_signal[:, k]

    # 5. Save data to files
    os.makedirs(output_path, exist_ok=True)

    np.save(os.path.join(output_path, f"{base_filename}_state.npy"), state_vector)
    np.save(os.path.join(output_path, f"{base_filename}_input.npy"), input_signal)
    np.save(os.path.join(output_path, f"{base_filename}_noise.npy"), noise_signal)
    

    # Optional: Return data in case it's needed immediately for further processing
    return state_vector, input_signal, noise_signal
    






#-------------------------------------------------------------------------------------------------



def generate_iid_samples(
    N: int,
    system_params: dict,
    params_config: dict,
    output_path: str,
    base_filename: str,
    seed: int = None
):
    """
    Generates i.i.d. (Independently and Identically Distributed) samples
    based on the static model y = a*x + b*u + w.

    Args:
        N (int): The number of independent samples (rollouts) to generate.
        system_params (dict): Dictionary with system coefficients, e.g., {'a': 0.5, 'b': 0.5}.
        params_config (dict): Configuration for the random variables' distributions.
                              e.g., {'x_std_dev': 1.0, 'u_std_dev': 1.0, 'w_std_dev': 0.005}
        output_path (str): The file path to save the generated data.
        base_filename (str): The base name for the output files.
        seed (int, optional): A seed for the random number generator.
    """
    # 1. Initialize the random number generator
    rng = np.random.default_rng(seed)

    # 2. Unpack system and distribution parameters
    a = system_params.get('a', 0.5)
    b = system_params.get('b', 0.5)

    x_std = params_config.get('x_std_dev', 1.0)
    u_std = params_config.get('u_std_dev', 1.0)
    w_std = params_config.get('w_std_dev', 0.01)

    # 3. Generate N independent samples for x, u, and w
    # We generate column vectors of size (N, 1)
    x = rng.normal(loc=0, scale=x_std, size=(N, 1))
    u = rng.normal(loc=0, scale=u_std, size=(N, 1))
    w = rng.normal(loc=0, scale=w_std, size=(N, 1))

    # 4. Calculate the output y 
    y = a * x + b * u + w

    # 5. Save the generated data arrays
    os.makedirs(output_path, exist_ok=True)

    np.save(os.path.join(output_path, f"{base_filename}_x_samples.npy"), x)
    np.save(os.path.join(output_path, f"{base_filename}_u_samples.npy"), u)
    np.save(os.path.join(output_path, f"{base_filename}_w_samples.npy"), w)
    np.save(os.path.join(output_path, f"{base_filename}_y_output.npy"), y)


    return x, u, w, y

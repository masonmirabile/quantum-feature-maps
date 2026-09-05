import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATASETS = ("blobs", "moons", "circles", "xor", "spiral", "wave", "sectors")

NOISE_VALUES = {
    "blobs":   {1:.30, 2:.53, 3:.77, 4:1.00, 5:1.35, 6:1.70, 7:2.20, 8:3.00, 9:3.80, 10:4.80, 11:6.00},
    "moons":   {1:.03, 2:.09, 3:.14, 4:.20, 5:.30, 6:.40, 7:.55, 8:.75, 9:.95, 10:1.20, 11:1.50},
    "circles": {1:.02, 2:.07, 3:.11, 4:.16, 5:.24, 6:.32, 7:.42, 8:.55, 9:.70, 10:.90, 11:1.15},
    "xor":     {1:.10, 2:.22, 3:.33, 4:.45, 5:.65, 6:.90, 7:1.20, 8:1.60, 9:2.00, 10:2.50, 11:3.20},
    "spiral":  {1:.02, 2:.05, 3:.08, 4:.12, 5:.18, 6:.24, 7:.32, 8:.42, 9:.55, 10:.72, 11:.95},
    "wave":    {1:.005, 2:.05, 3:.12, 4:.22, 5:.35, 6:.48, 7:.62, 8:.80, 9:1.00, 10:1.25, 11:1.55},
    "sectors": {1:.003, 2:.012, 3:.025, 4:.045, 5:.075, 6:.10, 7:.14, 8:.20, 9:.28, 10:.38, 11:.52}
}

def get_data(dataset, n=500, noise_level=1, seed=0):
    noise = NOISE_VALUES[dataset][noise_level]
    rng = np.random.default_rng(seed)

    # two gaussian clusters
    if dataset == "blobs":
        X, y = make_blobs(n_samples=n, centers=[(-1.5, 0), (1.5, 0)], cluster_std=noise, random_state=seed)

    # two interlocking semicircles
    elif dataset == "moons":
        X, y = make_moons(n_samples=n, noise=noise, random_state=seed)

    # two concentric circles
    elif dataset == "circles":
        X, y = make_circles(n_samples=n, factor=.45, noise=noise, random_state=seed)

    # four gaussian clusters with opposite diagonals sharing a class
    elif dataset == "xor":
        X, c = make_blobs(n_samples=n, centers=[(-1,-1), (1,1), (-1,1), (1,-1)], cluster_std=noise, random_state=seed)
        y = (c >= 2).astype(int)

    # two intertwined spirals
    elif dataset == "spiral":
        n0 = n // 2
        n1 = n - n0
        r0 = np.sqrt(rng.uniform(.02, 1, n0))
        r1 = np.sqrt(rng.uniform(.02, 1, n1))
        t0 = 2.5 * np.pi * r0
        t1 = 2.5 * np.pi * r1 + np.pi
        X0 = np.column_stack((r0*np.cos(t0), r0*np.sin(t0)))
        X1 = np.column_stack((r1*np.cos(t1), r1*np.sin(t1)))
        X = np.vstack((X0, X1)) + rng.normal(0, noise, (n, 2))
        y = np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]

    # points separated by a sinusoidal boundary
    elif dataset == "wave":
        X = np.column_stack((rng.uniform(-2, 2, n), rng.uniform(-1.5, 1.5, n)))
        y = (X[:,1] > .65*np.sin(2.5*X[:,0])).astype(int)
        X += rng.normal(0, noise, (n, 2))

    # disk split into eight alternating angular sectors
    elif dataset == "sectors":
        r = np.sqrt(rng.uniform(0, 1, n))
        t = rng.uniform(0, 2*np.pi, n)
        X = np.column_stack((r*np.cos(t), r*np.sin(t)))
        y = (np.sin(4*t) > 0).astype(int)
        X += rng.normal(0, noise, (n, 2))

    return X.astype(float), np.where(y == 0, -1, 1)

def get_train_eval_data(dataset, n=500, noise_level=1, seed=0, test_size=.2):
    X, y = get_data(dataset, n, noise_level, seed)

    # split first so the test data does not influence scaling
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # scale each feature to [-1, 1] using training data only
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)

    return X_train, X_eval, y_train, y_eval
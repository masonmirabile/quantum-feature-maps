import pennylane as qml
from pennylane import numpy as np

FEATURE_MAPS = ("angle", "z", "zz", "f1", "f2", "f4", "f6", "m1", "m2")

def feature_map(x, name):
    # simple y-rotation angle encoding
    if name == "angle":
        qml.AngleEmbedding(x, wires=[0, 1], rotation="Y")

    # first-order pauli-z encoding
    elif name == "z":
        for i in range(2):
            qml.Hadamard(wires=i)
            qml.RZ(2*x[i], wires=i)

    # second-order pauli-z encoding
    elif name == "zz":
        for i in range(2):
            qml.Hadamard(wires=i)
            qml.RZ(2*x[i], wires=i)
        qml.CNOT(wires=[0, 1])
        qml.RZ(2*(np.pi-x[0])*(np.pi-x[1]), wires=1)
        qml.CNOT(wires=[0, 1])

    # interaction-based feature maps from Jha et al. (2026)
    # use the same encoding gates as ZZ; only phi12 changes
    elif name in ("f1", "f2", "f4", "f6"):
        x1, x2 = x[0], x[1]

        if name == "f1":
            phi12 = np.pi / ((1 + np.cos(x1)) * (1 + np.cos(x2)))
        elif name == "f2":
            phi12 = np.pi * x1 * x2
        elif name == "f4":
            phi12 = np.exp((np.log(np.pi) / 8) * (x1-x2)**2)
        else:  # f6
            phi12 = np.pi * np.cos(x1) * np.cos(x2)

        for i in range(2):
            qml.Hadamard(wires=i)
            qml.RZ(2*x[i], wires=i)
        qml.CNOT(wires=[0, 1])
        qml.RZ(2*phi12, wires=1)
        qml.CNOT(wires=[0, 1])

    # student-designed interactions: same gate structure as ZZ
    elif name in ("m1", "m2"):
        x1, x2 = x[0], x[1]

        if name == "m1":
            phi12 = (np.pi / 2) * np.sin(x1) * np.sin(x2) * (x1**2 + x2**2)
        else:  # m2
            sign = np.where(x1*x2 >= 0, 1, -1)
            phi12 = x1*x2 + (np.pi / 4) * sign

        for i in range(2):
            qml.Hadamard(wires=i)
            qml.RZ(2*x[i], wires=i)
        qml.CNOT(wires=[0, 1])
        qml.RZ(2*phi12, wires=1)
        qml.CNOT(wires=[0, 1])

    else:
        raise ValueError(f"Unknown feature map: {name}")

def make_circuit(feature_map_name, layers):
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x, weights):
        feature_map(x, feature_map_name)
        qml.StronglyEntanglingLayers(weights, wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    return circuit

def train_vqc(X_train, y_train, X_eval, y_eval, feature_map_name="angle",
              layers=4, lr=.08, threshold=1e-4, max_steps=1000, seed=0):

    np.random.seed(seed)
    circuit = make_circuit(feature_map_name, layers)
    weights = .01*np.random.randn(layers, 2, 3, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=lr)

    def loss(weights):
        preds = np.array([circuit(x, weights) for x in X_train])
        return np.mean((preds-y_train)**2)

    old_loss = loss(weights)
    steps = 0
    converged = False

    while steps < max_steps:
        weights = opt.step(loss, weights)
        new_loss = loss(weights)
        steps += 1

        if abs(new_loss - old_loss) < threshold:
            converged = True
            break

        old_loss = new_loss

    train_preds = np.array([circuit(x, weights) for x in X_train])
    eval_preds = np.array([circuit(x, weights) for x in X_eval])

    train_acc = np.mean(np.sign(train_preds) == y_train)
    eval_acc = np.mean(np.sign(eval_preds) == y_eval)

    return {
        "train_acc": float(train_acc),
        "eval_acc": float(eval_acc),
        "steps": steps,
        "converged": converged,
        "weights": weights
    }
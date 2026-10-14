import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from vqc import make_circuit

circuit = make_circuit("angle", 4)

x = np.array([0.5, -0.3])
rng = np.random.default_rng(42)
weights = np.array(rng.normal(0, 0.01, (4, 2, 3)))

fig, ax = qml.draw_mpl(
    circuit,
    level="device",
    decimals=None,
    style="pennylane"
)(x, weights)

fig.savefig("vqc_circuit.png", dpi=300, bbox_inches="tight")
plt.show()
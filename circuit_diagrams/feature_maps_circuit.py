
import pennylane as qml
import matplotlib.pyplot as plt

def feature_map(x1, x2, phi12):
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)

    qml.RZ(2*x1, wires=0)
    qml.RZ(2*x2, wires=1)

    qml.CNOT(wires=[0, 1])
    qml.RZ(2*phi12, wires=1)
    qml.CNOT(wires=[0, 1])

# Draw the quantum function directly, not a QNode.
# No return statement means no measurement symbols.
fig, ax = qml.draw_mpl(
    feature_map,
    decimals=None,
    style="pennylane"
)(0.5, -0.3, 0.8)

# Replace the three RZ labels with symbolic expressions
symbols = [
    r"$R_Z(2x_1)$",
    r"$R_Z(2x_2)$",
    r"$R_Z(2\phi_{12})$"
]

rz_labels = [
    t for t in ax.texts
    if "RZ" in t.get_text() or "R_Z" in t.get_text()
]

if len(rz_labels) != 3:
    raise RuntimeError(
        f"Expected 3 RZ labels, found {len(rz_labels)}. "
        f"Actual labels: {[t.get_text() for t in ax.texts]}"
    )

for text, symbol in zip(rz_labels, symbols):
    text.set_text(symbol)
    text.set_fontsize(12)

fig.savefig(
    "feature_map_general.png",
    dpi=300,
    bbox_inches="tight"
)
fig.savefig(
    "feature_map_general.pdf",
    bbox_inches="tight"
)

plt.show()

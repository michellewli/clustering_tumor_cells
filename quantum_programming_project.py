import pennylane as qml
from pennylane import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Create a synthetic dataset for tumor cell clustering (features could represent tumor size, shape, etc.)
X, y = make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3)
X = StandardScaler().fit_transform(X)

# Reduce the dimensionality for plotting (we'll use PCA to project into 2D space)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Classical K-means clustering
def classical_kmeans(X):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_

# QAOA Parameters
n_qubits = 5
depth = 2

# Quantum Device
dev = qml.device("default.qubit", wires=n_qubits)

# Cost Hamiltonian for clustering
def cost_hamiltonian(z):
    cost = 0
    for i in range(len(z) - 1):
        cost += (1 - z[i] * z[i + 1])
    return cost / 2

# QAOA circuit
def qaoa_layer(gamma, beta):
    for wire in range(n_qubits):
        qml.Hadamard(wires=wire)
    
    for wire in range(n_qubits):
        qml.RZ(2 * gamma, wires=wire)
        
    for wire in range(n_qubits):
        qml.RX(2 * beta, wires=wire)

# QAOA Cost Function
def qaoa_cost(params):
    gamma, beta = params[0], params[1]
    qaoa_layer(gamma, beta)
    
    z = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    return cost_hamiltonian(z)

# QAOA Optimization
def run_qaoa(X):
    qaoa_opt = qml.GradientDescentOptimizer(stepsize=0.1)
    params = np.array([0.5, 0.5], requires_grad=True)  # initial params
    steps = 100

    for i in range(steps):
        params, cost = qaoa_opt.step_and_cost(qaoa_cost, params)
        if i % 10 == 0:
            print(f"Step {i} - Cost: {cost}")

    return params

# Plotting Function
def plot_clusters(X_pca, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()
    plt.show()

# Run Classical K-means
kmeans_labels, kmeans_centroids = classical_kmeans(X)

# Plot Classical K-means results
plot_clusters(X_pca, kmeans_labels, "Classical K-means Clustering")

# Run QAOA on tumor data
optimal_params = run_qaoa(X)

# For visualization, let's use the classical K-means labels as a proxy for the QAOA labels
# In your real case, you will define how to convert the QAOA output (quantum states) into labels
qaoa_labels = kmeans_labels  # Placeholder; replace with actual QAOA labeling method

# Plot QAOA results
plot_clusters(X_pca, qaoa_labels, "QAOA Clustering (Approximate)")

# Compare Results
print("Classical K-means clustering result:", kmeans_labels)
print("QAOA optimal parameters:", optimal_params)


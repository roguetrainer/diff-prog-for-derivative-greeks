# Introduction to Differentiable Programming

## What is Differentiable Programming?

**Differentiable programming** is a programming paradigm where programs are written to be automatically differentiable—that is, the derivatives (gradients) of program outputs with respect to inputs can be computed automatically. This enables optimization of program parameters through gradient-based methods, forming the mathematical foundation for modern machine learning, scientific computing, and beyond.

At its core, differentiable programming treats entire programs as mathematical functions that can be differentiated using automatic differentiation (AD). Unlike traditional programming where we explicitly code every step, differentiable programming allows us to specify *what* we want to optimize and let the system figure out *how* to compute the necessary gradients.

**Key principle**: Any computation expressible as a differentiable function can be optimized using gradient descent and its variants.

---

## The Three Pillars of Differentiable Programming

### 1. Automatic Differentiation (AD)

Automatic differentiation computes derivatives by applying the chain rule to elementary operations during program execution. There are two main modes:

- **Forward-mode AD**: Computes derivatives alongside the forward pass. Efficient when outputs >> inputs.
- **Reverse-mode AD**: Computes derivatives in a backward pass after the forward computation. Efficient when inputs >> outputs (e.g., neural networks with millions of parameters).

**Key distinction from other methods**:
- **Symbolic differentiation**: Manipulates mathematical expressions (can lead to expression swell)
- **Numerical differentiation**: Uses finite differences (introduces approximation errors)
- **Automatic differentiation**: Computes exact derivatives (to machine precision) efficiently

### 2. Computational Graphs

Programs are represented as directed acyclic graphs (DAGs) where:
- **Nodes** represent operations (addition, multiplication, activation functions)
- **Edges** represent data flow (tensors, scalars)
- **Gradients** flow backward through the graph via the chain rule

### 3. Gradient-Based Optimization

Once gradients are available, parameters can be optimized using:
- Stochastic gradient descent (SGD)
- Adam, RMSprop (adaptive learning rates)
- L-BFGS (second-order methods)
- Natural gradient descent

---

## A Brief History of Backpropagation

### Early Foundations (1960s-1970s)

The mathematical foundations of automatic differentiation and backpropagation emerged from several independent discoveries:

**1960s - Control Theory Origins**
- **Henry J. Kelley** (1960) and **Arthur E. Bryson** (1961) developed gradient methods for optimal control problems
- These "dynamic programming" approaches were early forms of backpropagation for continuous systems

**1970 - Linnainmaa's Thesis**
- **Seppo Linnainmaa** published his master's thesis introducing automatic differentiation in its modern form
- Described reverse-mode accumulation (the mathematical essence of backpropagation)
- Focused on general computational graphs, not specifically neural networks

### The Neural Network Revolution (1980s)

**1974 - Werbos's Dissertation**
- **Paul Werbos** independently discovered backpropagation in his PhD dissertation
- Applied it to neural networks but work remained largely unnoticed initially
- Later recognized as foundational contribution

**1986 - The Breakthrough Paper**
- **David Rumelhart, Geoffrey Hinton, and Ronald Williams** published "Learning representations by back-propagating errors" in *Nature*
- Made backpropagation accessible to the machine learning community
- Demonstrated practical success on several problems
- This paper catalyzed the neural network renaissance

**Key insight**: By treating a neural network as a composition of differentiable functions, gradients could be efficiently computed using the chain rule in reverse order—allowing networks with hidden layers to be trained end-to-end.

### The First AI Winter and Persistence (1990s)

Despite initial excitement, neural networks faced challenges:
- Vanishing/exploding gradients in deep networks
- Limited computational resources
- Success of support vector machines (SVMs) and other methods

**Persistent research**:
- **Yann LeCun** (1989): Backpropagation for convolutional neural networks (CNNs)
- **Sepp Hochreiter & Jürgen Schmidhuber** (1997): Long Short-Term Memory (LSTM) to address vanishing gradients
- **Yoshua Bengio**: Continued theoretical work on training deep networks

### The Deep Learning Era (2006-Present)

**2006 - Deep Belief Networks**
- **Geoffrey Hinton** showed that deep networks could be pre-trained layer-by-layer
- Reignited interest in deep learning

**2012 - ImageNet Breakthrough**
- **AlexNet** (Krizhevsky, Sutskever, Hinton) won ImageNet competition by large margin
- Used GPUs, ReLU activations, dropout
- Demonstrated that deep networks with backpropagation could scale

**2015-Present - Differentiable Programming**
- Frameworks like TensorFlow, PyTorch, JAX emerged
- Automatic differentiation became accessible to practitioners
- Backpropagation evolved into general differentiable programming
- Applications expanded far beyond neural networks

**Modern understanding**: Backpropagation is not just a neural network training algorithm—it's a special case of reverse-mode automatic differentiation, applicable to any differentiable computation.

---

## Application: Greeks in Derivative Pricing

### The Financial Context

In quantitative finance, **Greeks** are sensitivities of derivative prices to various parameters:
- **Delta (Δ)**: Sensitivity to underlying asset price (∂V/∂S)
- **Gamma (Γ)**: Convexity, second derivative (∂²V/∂S²)
- **Vega (ν)**: Sensitivity to volatility (∂V/∂σ)
- **Theta (Θ)**: Time decay (∂V/∂t)
- **Rho (ρ)**: Sensitivity to interest rate (∂V/∂r)

Computing Greeks is essential for risk management and hedging strategies.

### Traditional Methods

**1. Analytical formulas** (when available)
- Example: Black-Scholes Greeks for European options
- Pros: Exact and fast
- Cons: Only available for simple derivatives

**2. Finite difference methods** (bump-and-reprice)
- Compute V(S + ε) and V(S - ε), then approximate ∂V/∂S ≈ (V(S+ε) - V(S-ε))/(2ε)
- Pros: Simple to implement
- Cons: Requires N+1 pricings for N Greeks; approximation errors; expensive for Monte Carlo

**3. Pathwise derivatives**
- Differentiate through the Monte Carlo simulation
- Pros: Unbiased estimators for certain payoffs
- Cons: Requires smooth payoffs; complex to implement

### The "Smoking Adjoints" Revolution (2006)

**Giles & Glasserman's breakthrough**:

Michael Giles and Paul Glasserman's 2006 *Risk Magazine* article "Smoking Adjoints: Fast Monte Carlo Greeks" showed that **adjoint algorithmic differentiation (AAD)** could revolutionize Greeks computation:

**Key result**: All Greeks can be computed in roughly the same time as a single Monte Carlo pricing—regardless of the number of parameters.

**How it works**:
1. Run forward pass: Simulate Monte Carlo paths and compute option price
2. Run backward pass: Propagate "adjoint variables" backward through the computation
3. Accumulate gradients: Each operation contributes to parameter sensitivities via chain rule

**Efficiency**: 
- Forward-mode AD: O(N) cost for N Greeks
- Reverse-mode AD (adjoints): O(1) cost for all Greeks (constant factor ~2-4x single pricing)

**Mathematical foundation**: Reverse-mode AD is exactly backpropagation, but applied to financial computations rather than neural networks.

### Modern Implementation with Differentiable Programming

Today's frameworks (PyTorch, TensorFlow, JAX) make "smoking adjoints" accessible:

```python
import torch

def monte_carlo_call_price(S0, K, r, sigma, T, n_sims):
    """Price European call with automatic Greeks."""
    # Track gradients
    S0.requires_grad_(True)
    sigma.requires_grad_(True)
    
    # Monte Carlo simulation
    Z = torch.randn(n_sims)
    S_T = S0 * torch.exp((r - 0.5*sigma**2)*T + sigma*torch.sqrt(T)*Z)
    payoff = torch.maximum(S_T - K, torch.tensor(0.0))
    price = torch.exp(-r*T) * payoff.mean()
    
    # Compute Greeks automatically!
    price.backward()
    
    return {
        'price': price.item(),
        'delta': S0.grad.item(),      # ∂V/∂S
        'vega': sigma.grad.item()      # ∂V/∂σ
    }

# Usage
S0 = torch.tensor(100.0, requires_grad=True)
sigma = torch.tensor(0.2, requires_grad=True)
results = monte_carlo_call_price(S0, K=100, r=0.05, sigma=sigma, T=1.0, n_sims=100000)
```

**Result**: All Greeks computed in one backward pass—exactly as Giles & Glasserman envisioned, but with 10 lines of Python instead of custom C++ code.

**Impact**: 
- 10-100x faster than finite differences
- Exact derivatives (no approximation error)
- Works for any derivative structure (Asian, barrier, basket options, etc.)
- GPU acceleration available
- Production-ready implementations

### Why This Matters

The connection between backpropagation (1986) and adjoint methods in finance (2006) demonstrates that differentiable programming transcends any single domain. The same mathematical principles that train neural networks also compute financial sensitivities—and both are special cases of automatic differentiation.

---

## Application: Quantum Machine Learning with PennyLane

### Quantum Computing Meets Differentiable Programming

**PennyLane** is a quantum machine learning library that brings differentiable programming to quantum computing. It allows quantum circuits to be treated as differentiable functions, enabling:
- Training quantum circuits with gradient descent
- Hybrid quantum-classical optimization
- Quantum circuit learning

### Why Differentiate Quantum Circuits?

Quantum algorithms often have free parameters (rotation angles, gate parameters) that need optimization:
- **Variational Quantum Eigensolvers (VQE)**: Find ground state energies
- **Quantum Approximate Optimization Algorithm (QAOA)**: Solve combinatorial problems
- **Quantum Neural Networks (QNNs)**: Machine learning with quantum circuits

**Challenge**: How do we compute gradients of expectation values with respect to circuit parameters?

### Parameter-Shift Rule

Unlike classical neural networks where backpropagation naturally applies, quantum circuits require special treatment:

**The parameter-shift rule** enables gradient computation on quantum hardware:

For a gate with parameter θ, the gradient of an expectation value ⟨O⟩ is:

```
∂⟨O⟩/∂θ = (⟨O(θ + π/2)⟩ - ⟨O(θ - π/2)⟩) / 2
```

This works by evaluating the circuit at shifted parameter values—no backpropagation through quantum operations needed!

**Key insight**: The parameter-shift rule is exact (not approximate like finite differences) and can be computed using the quantum hardware itself.

### PennyLane's Differentiable Quantum Circuits

```python
import pennylane as qml
import numpy as np

# Create quantum device
dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev, diff_method='parameter-shift')
def quantum_circuit(params):
    """Differentiable quantum circuit."""
    # Apply parameterized gates
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[2], wires=1)
    
    # Return expectation value
    return qml.expval(qml.PauliZ(0))

# Define loss function
def loss(params):
    return (quantum_circuit(params) - 1.0)**2

# Compute gradients using parameter-shift rule
params = np.array([0.1, 0.2, 0.3], requires_grad=True)
gradient = qml.grad(loss)(params)

# Gradient descent optimization
for step in range(100):
    params = params - 0.1 * gradient
    gradient = qml.grad(loss)(params)
```

### Hybrid Quantum-Classical Models

PennyLane integrates seamlessly with PyTorch and TensorFlow, enabling hybrid architectures:

```python
import torch
import pennylane as qml

# Quantum device
dev = qml.device('default.qubit', wires=4)

@qml.qnode(dev, interface='torch', diff_method='parameter-shift')
def quantum_layer(inputs, weights):
    """Quantum layer as PyTorch operation."""
    # Encode classical data
    for i, x in enumerate(inputs):
        qml.RX(x, wires=i)
    
    # Variational circuit
    for i, w in enumerate(weights):
        qml.RY(w, wires=i)
    
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class HybridModel(torch.nn.Module):
    """Classical NN → Quantum Circuit → Classical NN"""
    
    def __init__(self):
        super().__init__()
        self.classical_in = torch.nn.Linear(10, 4)
        self.quantum_weights = torch.nn.Parameter(torch.randn(4))
        self.classical_out = torch.nn.Linear(4, 2)
    
    def forward(self, x):
        # Classical preprocessing
        x = torch.tanh(self.classical_in(x))
        
        # Quantum layer (differentiable!)
        x = quantum_layer(x, self.quantum_weights)
        x = torch.stack(x)
        
        # Classical postprocessing
        x = self.classical_out(x)
        return x

# Train end-to-end with standard PyTorch
model = HybridModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Gradients flow through quantum circuit!
loss = model(data).sum()
loss.backward()
optimizer.step()
```

### Differentiation Methods in PennyLane

PennyLane supports multiple gradient computation strategies:

1. **Parameter-shift rule**: Default for quantum hardware
   - Exact gradients
   - Hardware-compatible
   - Requires 2 circuit evaluations per parameter

2. **Finite differences**: Numerical approximation
   - Works for any circuit
   - Approximate gradients
   - Simple but less accurate

3. **Adjoint differentiation**: Efficient classical simulation
   - Fast for simulators
   - Exact gradients
   - Not hardware-compatible

4. **Backpropagation**: For classical simulators
   - Fastest for large circuits
   - Classical simulation only
   - Uses standard AD techniques

### Applications of Quantum Differentiable Programming

**1. Quantum Chemistry (VQE)**
```python
# Find ground state energy of H2 molecule
@qml.qnode(dev, diff_method='parameter-shift')
def vqe_circuit(params, hamiltonian):
    # Prepare ansatz state
    prepare_ansatz(params)
    # Return energy expectation
    return qml.expval(hamiltonian)

# Optimize to minimize energy
energy_gradient = qml.grad(vqe_circuit)
```

**2. Quantum Optimization (QAOA)**
```python
# Solve MaxCut problem
@qml.qnode(dev)
def qaoa_circuit(params, graph):
    # Cost Hamiltonian
    apply_cost_layer(params[::2], graph)
    # Mixer Hamiltonian
    apply_mixer_layer(params[1::2])
    return qml.expval(cost_hamiltonian(graph))
```

**3. Quantum Machine Learning**
```python
# Classify quantum data
def quantum_classifier(x, weights):
    encode_data(x)
    variational_circuit(weights)
    return qml.probs(wires=range(n_qubits))
```

### Why PennyLane Matters for Differentiable Programming

PennyLane demonstrates that differentiable programming extends beyond classical computing:

- **Unifies quantum and classical ML**: Same optimization techniques apply
- **Hardware-agnostic**: Works with simulators and real quantum processors
- **Framework integration**: Seamless with PyTorch, TensorFlow, JAX
- **Research enabler**: Accelerates quantum algorithm development

**The broader lesson**: Any computational paradigm with tunable parameters can benefit from differentiable programming—whether classical neural networks (1986), financial derivatives (2006), or quantum circuits (2018+).

---

## Framework Support: TensorFlow and PyTorch

### TensorFlow (Google, 2015)

**Original vision**: Static computational graphs with automatic differentiation

**TensorFlow 1.x** (2015-2019):
- Define-then-run paradigm
- Static computation graphs built before execution
- Powerful but less intuitive for research

**TensorFlow 2.x** (2019-present):
- Eager execution by default (like PyTorch)
- `tf.GradientTape` for automatic differentiation:

```python
import tensorflow as tf

# Automatic differentiation with GradientTape
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2 + 2*x + 1

dy_dx = tape.gradient(y, x)  # dy/dx = 2x + 2 = 8
```

**Strengths**:
- Production deployment (TensorFlow Serving, TensorFlow Lite)
- Distributed training (TensorFlow distributed strategies)
- Ecosystem (TensorBoard, TFX, TensorFlow Hub)
- Mobile and embedded devices
- Strong industry adoption

**Use cases**:
- Production ML systems
- Mobile/edge deployment
- Large-scale distributed training
- Google Cloud integration

### PyTorch (Facebook/Meta, 2016)

**Vision**: Dynamic computational graphs with Pythonic design

**PyTorch philosophy**:
- Define-by-run: Graphs built during execution
- Eager execution by default
- Feels like native Python

**Automatic differentiation with autograd**:

```python
import torch

# Automatic differentiation
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1

y.backward()  # Compute gradients
print(x.grad)  # dy/dx = 8
```

**Strengths**:
- Research-friendly (dynamic graphs, easy debugging)
- Intuitive API design
- Strong research community adoption
- Excellent for prototyping
- TorchScript for production deployment
- Growing production tools

**Use cases**:
- Research and experimentation
- Rapid prototyping
- Academic applications
- Custom gradient operations
- Computer vision and NLP research

### Feature Comparison

| Feature | TensorFlow 2.x | PyTorch |
|---------|----------------|---------|
| Graph type | Dynamic (eager) | Dynamic (eager) |
| API design | Keras-focused | Native tensors |
| Production | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Research | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Debugging | Good | Excellent |
| Mobile/Edge | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Learning curve | Medium | Easy |
| Community | Large | Very large |

### Convergence of Frameworks

Modern TensorFlow and PyTorch have converged significantly:

**Similarities**:
- Both support dynamic computation graphs
- Both have eager execution by default
- Both provide powerful AD engines
- Both support GPU/TPU acceleration
- Both integrate with common tools (ONNX, etc.)

**When to choose each**:

**Choose TensorFlow if**:
- Deploying to production at scale
- Targeting mobile/embedded devices
- Using Google Cloud Platform
- Need proven production tooling

**Choose PyTorch if**:
- Doing research or prototyping
- Need maximum flexibility
- Prefer Pythonic coding style
- Want easier debugging

**Reality**: Many practitioners know both, and the choice often depends on team preference and existing infrastructure.

### Beyond TensorFlow and PyTorch

The differentiable programming ecosystem continues to evolve:

- **JAX** (Google, 2018): Composable transformations, functional programming
- **Julia** (Zygote, Flux): High-performance scientific computing
- **Swift for TensorFlow**: Native Swift integration (development paused)
- **Enzyme**: AD for LLVM (works with C++, Rust, Julia, Swift)

---

## The Broader Impact of Differentiable Programming

### Beyond Machine Learning

Differentiable programming is expanding into diverse fields:

**1. Scientific Computing**
- Climate modeling with learnable parameters
- Protein folding (AlphaFold uses differentiable structure prediction)
- Computational fluid dynamics

**2. Computer Graphics**
- Differentiable rendering (inverse graphics)
- 3D shape optimization
- Neural radiance fields (NeRF)

**3. Robotics**
- Learning robot controllers
- Trajectory optimization
- Sim-to-real transfer

**4. Finance**
- Derivative pricing (as discussed)
- Portfolio optimization
- Risk management

**5. Physics Simulation**
- Differentiable physics engines
- Learning physical laws from data
- Inverse problems in physics

### The Unifying Principle

What connects backpropagation (1986), smoking adjoints (2006), and quantum circuit learning (2018+)?

**Answer**: They all use **reverse-mode automatic differentiation** to efficiently compute gradients of outputs with respect to inputs, enabling gradient-based optimization.

**The paradigm shift**: Instead of manually deriving and coding gradients, we:
1. Express our problem as a differentiable computation
2. Let the AD engine compute gradients automatically
3. Optimize using gradient descent

This works whether we're training neural networks, pricing derivatives, or tuning quantum circuits.

---

## Conclusion

Differentiable programming represents a fundamental paradigm shift in how we write and optimize software. By making entire programs differentiable, we can:

- **Optimize complex systems** end-to-end
- **Learn from data** rather than hand-crafting rules
- **Unify disparate fields** under common mathematical principles
- **Accelerate scientific discovery** through automated optimization

From its origins in control theory and neural networks through its application in finance and quantum computing, differentiable programming has proven to be one of the most powerful computational abstractions of the 21st century.

The frameworks (TensorFlow, PyTorch) and tools (PennyLane) continue to mature, making sophisticated automatic differentiation accessible to practitioners across domains. As we've seen, the same principles that Rumelhart, Hinton, and Williams used to train neural networks in 1986 now power:
- Financial risk management (Giles & Glasserman, 2006)
- Quantum algorithm development (PennyLane, 2018+)
- And countless other applications yet to be discovered

**The future of programming is differentiable.**

---

## Further Reading

### Foundational Papers

1. **Backpropagation**:
   - Rumelhart, Hinton & Williams (1986): "Learning representations by back-propagating errors", *Nature*
   - Linnainmaa (1970): "The representation of the cumulative rounding error of an algorithm as a Taylor expansion of the local rounding errors" (Master's thesis)

2. **Financial Applications**:
   - Giles & Glasserman (2006): "Smoking Adjoints: Fast Monte Carlo Greeks", *Risk Magazine*
   - Ferguson & Green (2018): "Deeply Learning Derivatives", arXiv:1809.02233

3. **Quantum Machine Learning**:
   - Schuld et al. (2019): "Evaluating analytic gradients on quantum hardware", *Physical Review A*
   - Mitarai et al. (2018): "Quantum circuit learning", *Physical Review A*

### Modern Frameworks

- **TensorFlow**: tensorflow.org
- **PyTorch**: pytorch.org
- **PennyLane**: pennylane.ai
- **JAX**: github.com/google/jax

### Books

- Goodfellow, Bengio & Courville (2016): *Deep Learning*, MIT Press
- Griewank & Walther (2008): *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, SIAM

---

*This introduction provides a foundation for understanding differentiable programming across domains. The principles are universal; the applications are limited only by imagination.*

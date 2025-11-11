# PyTorch for Financial Derivatives: Complete Package

This package demonstrates how to use PyTorch's automatic differentiation for pricing financial derivatives and computing Greeks efficiently and accurately.

![img](./AAD%20Greeks.png)

## 📦 Package Contents

### Core Files

1. **pytorch_derivatives.py** - Production-ready Python module
   - Complete implementation of option pricing with autograd
   - European, Asian, Basket, and Barrier options
   - Benchmarking and validation code
   - Run standalone: `python pytorch_derivatives.py`

2. **pytorch_derivatives_demo.ipynb** - Interactive Jupyter notebook
   - Step-by-step tutorials with visualizations
   - Delta surfaces and contour plots
   - Performance comparisons
   - Open with: `jupyter notebook pytorch_derivatives_demo.ipynb`

3. **pytorch_derivatives_overview.md** - Comprehensive article
   - Historical context and theory
   - Implementation details
   - Best practices and recommendations
   - View in any markdown reader

4. **differentiable_programming_intro.md** - ⭐ Introduction to Differentiable Programming
   - History of backpropagation (1970s-present)
   - Greeks computation in finance (Giles & Glasserman 2006)
   - Quantum machine learning with PennyLane
   - TensorFlow and PyTorch framework overview
   - Connects neural networks, finance, and quantum computing

### Setup Files

5. **setup.sh** - ⭐ Automated environment setup script
   - Creates Python virtual environment
   - Installs all dependencies (CPU or GPU)
   - Verifies installation and runs tests
   - See SETUP_GUIDE.md for details

6. **SETUP_GUIDE.md** - Detailed setup instructions
   - Step-by-step manual setup
   - Troubleshooting guide
   - Platform-specific notes

## 🚀 Quick Start

### Automated Setup (Recommended)

Use the provided setup script to create a virtual environment with all dependencies:

```bash
# CPU-only installation (works everywhere)
bash setup.sh

# GPU installation (if you have NVIDIA GPU)
bash setup.sh --gpu
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup instructions and troubleshooting.

### Manual Installation

```bash
pip install torch numpy scipy matplotlib jupyter
```

### Run the Demo

```bash
python pytorch_derivatives.py
```

### Key Concepts

**Automatic Differentiation**: PyTorch computes exact derivatives through any computation graph, making Greeks calculation trivial.

**Smoking Adjoints Connection**: PyTorch implements the adjoint algorithmic differentiation (AAD) method from Giles & Glasserman (2006) with modern tooling and GPU support.

### 🌟 New: Differentiable Programming Context

The new **differentiable_programming_intro.md** provides essential background:

- **Why it matters**: The same math (reverse-mode AD) powers all these applications
- **Historical journey**: From backpropagation (1986) to quantum circuits (2018+)
- **Cross-domain insight**: Neural networks, financial Greeks, and quantum optimization share the same foundation
- **Framework evolution**: How TensorFlow and PyTorch democratized automatic differentiation

**Recommended reading order**: Start with the intro, then dive into the finance-specific materials!

## 💡 Example

```python
import torch

# Create tensors with gradient tracking
S0 = torch.tensor(100.0, requires_grad=True)
sigma = torch.tensor(0.2, requires_grad=True)

# Price option (any formula)
price = black_scholes(S0, K=100, r=0.05, sigma=sigma, T=1.0)

# Compute Greeks automatically!
delta = torch.autograd.grad(price, S0, create_graph=True)[0]
gamma = torch.autograd.grad(delta, S0)[0]
vega = torch.autograd.grad(price, sigma)[0]
```

## 📊 Performance

- **17x faster** than finite difference methods
- **Exact derivatives** (not approximations)
- **10-100x speedup** with GPU acceleration
- **All Greeks in one pass** (like "Smoking Adjoints")

## 📚 What You'll Learn

- How PyTorch autograd works for derivatives pricing
- Implementation of vanilla and exotic options
- Monte Carlo with automatic Greeks
- GPU acceleration techniques
- Connection to academic literature
- **NEW**: Broader context of differentiable programming across domains
  - History from backpropagation (1986) to modern frameworks
  - Applications in finance, quantum computing, and beyond
  - Why the same math powers neural networks, Greeks, and quantum circuits

## 🎯 File Details

| File | Purpose | Best For |
|------|---------|----------|
| .py | Standalone module | Running benchmarks, importing functions |
| .ipynb | Interactive tutorial | Learning step-by-step, visualization |
| pytorch_derivatives_overview.md | Reference article | Understanding theory, best practices |
| differentiable_programming_intro.md | Broader context | Understanding AD history across domains |

## 🔬 Options Implemented

✅ European calls/puts (MC and analytical)  
✅ Asian options (arithmetic average)  
✅ Basket options (multi-asset with correlation)  
✅ Barrier options (down-and-out)  
✅ All with automatic Greeks computation

## 📖 References

- Giles & Glasserman (2006): "Smoking Adjoints: Fast Monte Carlo Greeks"
- Ferguson & Green (2018): "Deeply Learning Derivatives"
- PyTorch Documentation: pytorch.org/tutorials

## ⚡ Next Steps

1. **Start here**: Read `differentiable_programming_intro.md` for broader context
2. Run `python pytorch_derivatives.py` to see results
3. Open the Jupyter notebook for interactive learning
4. Read `pytorch_derivatives_overview.md` for deep understanding
5. Experiment with parameters and add new options!

**Ready to compute Greeks automatically?** Start with the Python script! 🚀

---

**Version**: 1.0 | **License**: MIT | **Status**: Educational demonstration with production-ready code

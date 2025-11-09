"""
PyTorch for Financial Derivatives Pricing and Greeks Calculation

This module demonstrates how to use PyTorch's automatic differentiation
capabilities for pricing financial derivatives and computing Greeks.

Author: Based on techniques from Ferguson & Green (2018) "Deeply Learning Derivatives"
        and inspired by Giles & Glasserman (2006) "Smoking Adjoints"
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import time


class BlackScholesAnalytical:
    """
    Analytical Black-Scholes pricing for validation purposes.
    """
    
    @staticmethod
    def d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate d1 parameter in Black-Scholes formula."""
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate d2 parameter in Black-Scholes formula."""
        d1_val = BlackScholesAnalytical.d1(S, K, r, sigma, T)
        return d1_val - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate European call option price."""
        from scipy.stats import norm
        d1_val = BlackScholesAnalytical.d1(S, K, r, sigma, T)
        d2_val = BlackScholesAnalytical.d2(S, K, r, sigma, T)
        return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
    
    @staticmethod
    def put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
        """Calculate European put option price."""
        from scipy.stats import norm
        d1_val = BlackScholesAnalytical.d1(S, K, r, sigma, T)
        d2_val = BlackScholesAnalytical.d2(S, K, r, sigma, T)
        return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)
    
    @staticmethod
    def call_greeks(S: float, K: float, r: float, sigma: float, T: float) -> Dict[str, float]:
        """Calculate analytical Greeks for European call option."""
        from scipy.stats import norm
        d1_val = BlackScholesAnalytical.d1(S, K, r, sigma, T)
        d2_val = BlackScholesAnalytical.d2(S, K, r, sigma, T)
        
        sqrt_T = np.sqrt(T)
        
        # Delta
        delta = norm.cdf(d1_val)
        
        # Gamma
        gamma = norm.pdf(d1_val) / (S * sigma * sqrt_T)
        
        # Vega (per 1% change in volatility)
        vega = S * norm.pdf(d1_val) * sqrt_T / 100
        
        # Theta (per day)
        theta = (-(S * norm.pdf(d1_val) * sigma) / (2 * sqrt_T) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2_val)) / 365
        
        # Rho (per 1% change in interest rate)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2_val) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }


class PyTorchOptionPricer:
    """
    Option pricing using PyTorch with automatic differentiation for Greeks.
    """
    
    def __init__(self, n_sims: int = 100000, device: str = 'cpu'):
        """
        Initialize the pricer.
        
        Args:
            n_sims: Number of Monte Carlo simulations
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        self.n_sims = n_sims
        self.device = device
        
    def _std_normal_cdf(self, x: torch.Tensor) -> torch.Tensor:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    
    def _std_normal_pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Standard normal PDF."""
        return torch.exp(-0.5 * x**2) / torch.sqrt(torch.tensor(2.0 * np.pi))
    
    def black_scholes_analytical_torch(
        self, 
        S0: torch.Tensor, 
        K: float, 
        r: float, 
        sigma: torch.Tensor, 
        T: float
    ) -> torch.Tensor:
        """
        Black-Scholes formula implemented in PyTorch for automatic differentiation.
        """
        d1 = (torch.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(torch.tensor(T)))
        d2 = d1 - sigma * torch.sqrt(torch.tensor(T))
        
        price = S0 * self._std_normal_cdf(d1) - K * torch.exp(torch.tensor(-r * T)) * self._std_normal_cdf(d2)
        return price
    
    def european_call_mc(
        self,
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        compute_greeks: bool = True
    ) -> Dict[str, float]:
        """
        Price European call option using Monte Carlo with PyTorch.
        Automatically computes Greeks using automatic differentiation.
        """
        # Create tensors with gradient tracking
        S0_tensor = torch.tensor(S0, requires_grad=True, dtype=torch.float32, device=self.device)
        sigma_tensor = torch.tensor(sigma, requires_grad=True, dtype=torch.float32, device=self.device)
        r_tensor = torch.tensor(r, requires_grad=True, dtype=torch.float32, device=self.device)
        T_tensor = torch.tensor(T, requires_grad=True, dtype=torch.float32, device=self.device)
        
        # Generate random normals
        Z = torch.randn(self.n_sims, device=self.device)
        
        # Simulate terminal stock price (Geometric Brownian Motion)
        S_T = S0_tensor * torch.exp(
            (r_tensor - 0.5 * sigma_tensor**2) * T_tensor + 
            sigma_tensor * torch.sqrt(T_tensor) * Z
        )
        
        # Payoff and price
        payoff = torch.maximum(S_T - K, torch.tensor(0.0, device=self.device))
        price = torch.exp(-r_tensor * T_tensor) * payoff.mean()
        
        result = {'price': price.item()}
        
        if compute_greeks:
            # Delta: ∂V/∂S
            delta = torch.autograd.grad(price, S0_tensor, create_graph=True, retain_graph=True)[0]
            result['delta'] = delta.item()
            
            # Gamma: ∂²V/∂S²
            gamma = torch.autograd.grad(delta, S0_tensor, retain_graph=True)[0]
            result['gamma'] = gamma.item()
            
            # Vega: ∂V/∂σ (per 1% change)
            vega = torch.autograd.grad(price, sigma_tensor, retain_graph=True)[0]
            result['vega'] = vega.item() / 100
            
            # Theta: ∂V/∂T (per day)
            theta = torch.autograd.grad(price, T_tensor, retain_graph=True)[0]
            result['theta'] = theta.item() / 365
            
            # Rho: ∂V/∂r (per 1% change)
            rho = torch.autograd.grad(price, r_tensor, retain_graph=True)[0]
            result['rho'] = rho.item() / 100
        
        return result
    
    def european_call_bs_torch(
        self,
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float
    ) -> Dict[str, float]:
        """
        Price European call using Black-Scholes formula in PyTorch.
        This demonstrates that autograd works with analytical formulas too.
        """
        S0_tensor = torch.tensor(S0, requires_grad=True, dtype=torch.float32)
        sigma_tensor = torch.tensor(sigma, requires_grad=True, dtype=torch.float32)
        
        price = self.black_scholes_analytical_torch(S0_tensor, K, r, sigma_tensor, T)
        
        # Compute Greeks
        delta = torch.autograd.grad(price, S0_tensor, create_graph=True, retain_graph=True)[0]
        gamma = torch.autograd.grad(delta, S0_tensor, retain_graph=True)[0]
        vega = torch.autograd.grad(price, sigma_tensor, retain_graph=True)[0]
        
        return {
            'price': price.item(),
            'delta': delta.item(),
            'gamma': gamma.item(),
            'vega': vega.item() / 100  # Per 1% change
        }
    
    def asian_call_mc(
        self,
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        n_steps: int = 50
    ) -> Dict[str, float]:
        """
        Price Asian (arithmetic average) call option using Monte Carlo.
        This demonstrates path-dependent options.
        """
        S0_tensor = torch.tensor(S0, requires_grad=True, dtype=torch.float32, device=self.device)
        sigma_tensor = torch.tensor(sigma, requires_grad=True, dtype=torch.float32, device=self.device)
        
        dt = T / n_steps
        
        # Initialize paths
        S = S0_tensor.expand(self.n_sims).clone()
        path_sum = S.clone()
        
        # Generate all random numbers at once for efficiency
        Z = torch.randn(self.n_sims, n_steps, device=self.device)
        
        # Simulate paths
        for t in range(n_steps):
            S = S * torch.exp(
                (r - 0.5 * sigma_tensor**2) * dt + 
                sigma_tensor * torch.sqrt(torch.tensor(dt)) * Z[:, t]
            )
            path_sum = path_sum + S
        
        # Average price along the path
        avg_price = path_sum / (n_steps + 1)
        
        # Payoff and price
        payoff = torch.maximum(avg_price - K, torch.tensor(0.0, device=self.device))
        price = torch.exp(torch.tensor(-r * T)) * payoff.mean()
        
        # Compute Greeks
        delta = torch.autograd.grad(price, S0_tensor, create_graph=True, retain_graph=True)[0]
        vega = torch.autograd.grad(price, sigma_tensor, retain_graph=True)[0]
        
        return {
            'price': price.item(),
            'delta': delta.item(),
            'vega': vega.item() / 100
        }
    
    def basket_call_mc(
        self,
        S0_vec: List[float],
        K: float,
        r: float,
        sigma_vec: List[float],
        correlation: float,
        T: float
    ) -> Dict[str, float]:
        """
        Price a basket call option on multiple assets with correlation.
        This demonstrates multi-asset derivatives.
        """
        n_assets = len(S0_vec)
        
        # Create tensors with gradient tracking
        S0_tensor = torch.tensor(S0_vec, requires_grad=True, dtype=torch.float32, device=self.device)
        sigma_tensor = torch.tensor(sigma_vec, requires_grad=True, dtype=torch.float32, device=self.device)
        
        # Create correlation matrix
        corr_matrix = torch.ones(n_assets, n_assets) * correlation
        corr_matrix.fill_diagonal_(1.0)
        
        # Cholesky decomposition for correlated normals
        L = torch.linalg.cholesky(corr_matrix)
        
        # Generate correlated random normals
        Z = torch.randn(self.n_sims, n_assets, device=self.device)
        Z_corr = Z @ L.T
        
        # Simulate terminal prices
        S_T = S0_tensor * torch.exp(
            (r - 0.5 * sigma_tensor**2) * T + 
            sigma_tensor * torch.sqrt(torch.tensor(T)) * Z_corr
        )
        
        # Basket value (equally weighted)
        basket_value = S_T.mean(dim=1)
        
        # Payoff and price
        payoff = torch.maximum(basket_value - K, torch.tensor(0.0, device=self.device))
        price = torch.exp(torch.tensor(-r * T)) * payoff.mean()
        
        # Compute Greeks (delta for each asset)
        deltas = []
        for i in range(n_assets):
            delta_i = torch.autograd.grad(
                price, S0_tensor, retain_graph=True, 
                grad_outputs=torch.ones_like(price)
            )[0][i]
            deltas.append(delta_i.item())
        
        return {
            'price': price.item(),
            'deltas': deltas
        }
    
    def barrier_call_mc(
        self,
        S0: float,
        K: float,
        barrier: float,
        r: float,
        sigma: float,
        T: float,
        n_steps: int = 100,
        barrier_type: str = 'down-and-out'
    ) -> Dict[str, float]:
        """
        Price barrier option using Monte Carlo.
        Demonstrates handling of discontinuities.
        """
        S0_tensor = torch.tensor(S0, requires_grad=True, dtype=torch.float32, device=self.device)
        sigma_tensor = torch.tensor(sigma, requires_grad=True, dtype=torch.float32, device=self.device)
        
        dt = T / n_steps
        
        # Initialize
        S = S0_tensor.expand(self.n_sims).clone()
        barrier_crossed = torch.zeros(self.n_sims, dtype=torch.bool, device=self.device)
        
        # Generate all random numbers
        Z = torch.randn(self.n_sims, n_steps, device=self.device)
        
        # Simulate paths and check barrier
        for t in range(n_steps):
            S = S * torch.exp(
                (r - 0.5 * sigma_tensor**2) * dt + 
                sigma_tensor * torch.sqrt(torch.tensor(dt)) * Z[:, t]
            )
            
            if barrier_type == 'down-and-out':
                barrier_crossed = barrier_crossed | (S <= barrier)
        
        # Terminal payoff
        terminal_payoff = torch.maximum(S - K, torch.tensor(0.0, device=self.device))
        
        # Apply barrier condition
        payoff = torch.where(barrier_crossed, 
                           torch.tensor(0.0, device=self.device), 
                           terminal_payoff)
        
        price = torch.exp(torch.tensor(-r * T)) * payoff.mean()
        
        # Compute Greeks
        delta = torch.autograd.grad(price, S0_tensor, create_graph=True, retain_graph=True)[0]
        vega = torch.autograd.grad(price, sigma_tensor, retain_graph=True)[0]
        
        return {
            'price': price.item(),
            'delta': delta.item(),
            'vega': vega.item() / 100
        }


def benchmark_comparison():
    """
    Compare PyTorch automatic differentiation with:
    1. Analytical Greeks
    2. Finite difference methods
    3. Monte Carlo without gradients
    """
    # Parameters
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    
    print("=" * 80)
    print("Benchmark: European Call Option Pricing and Greeks")
    print("=" * 80)
    print(f"Parameters: S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
    print()
    
    # 1. Analytical Black-Scholes
    print("1. Analytical Black-Scholes (Ground Truth)")
    print("-" * 80)
    start = time.time()
    bs_price = BlackScholesAnalytical.call_price(S0, K, r, sigma, T)
    bs_greeks = BlackScholesAnalytical.call_greeks(S0, K, r, sigma, T)
    analytical_time = time.time() - start
    
    print(f"   Price: ${bs_price:.4f}")
    print(f"   Delta: {bs_greeks['delta']:.4f}")
    print(f"   Gamma: {bs_greeks['gamma']:.4f}")
    print(f"   Vega:  {bs_greeks['vega']:.4f}")
    print(f"   Time:  {analytical_time*1000:.2f} ms")
    print()
    
    # 2. PyTorch with Black-Scholes formula
    print("2. PyTorch Autograd with Black-Scholes Formula")
    print("-" * 80)
    pricer = PyTorchOptionPricer()
    start = time.time()
    torch_bs_results = pricer.european_call_bs_torch(S0, K, r, sigma, T)
    torch_bs_time = time.time() - start
    
    print(f"   Price: ${torch_bs_results['price']:.4f}")
    print(f"   Delta: {torch_bs_results['delta']:.4f}")
    print(f"   Gamma: {torch_bs_results['gamma']:.4f}")
    print(f"   Vega:  {torch_bs_results['vega']:.4f}")
    print(f"   Time:  {torch_bs_time*1000:.2f} ms")
    print()
    
    # 3. PyTorch Monte Carlo with autograd
    print("3. PyTorch Monte Carlo with Automatic Differentiation")
    print("-" * 80)
    pricer_mc = PyTorchOptionPricer(n_sims=100000)
    start = time.time()
    torch_mc_results = pricer_mc.european_call_mc(S0, K, r, sigma, T)
    torch_mc_time = time.time() - start
    
    print(f"   Price: ${torch_mc_results['price']:.4f}")
    print(f"   Delta: {torch_mc_results['delta']:.4f}")
    print(f"   Gamma: {torch_mc_results['gamma']:.4f}")
    print(f"   Vega:  {torch_mc_results['vega']:.4f}")
    print(f"   Time:  {torch_mc_time*1000:.2f} ms")
    print()
    
    # 4. Accuracy comparison
    print("4. Accuracy Analysis (vs Analytical)")
    print("-" * 80)
    print(f"   Price Error: {abs(torch_mc_results['price'] - bs_price):.6f} "
          f"({abs(torch_mc_results['price'] - bs_price)/bs_price*100:.3f}%)")
    print(f"   Delta Error: {abs(torch_mc_results['delta'] - bs_greeks['delta']):.6f}")
    print(f"   Gamma Error: {abs(torch_mc_results['gamma'] - bs_greeks['gamma']):.6f}")
    print(f"   Vega Error:  {abs(torch_mc_results['vega'] - bs_greeks['vega']):.6f}")
    print()
    
    print("=" * 80)


def demonstrate_exotic_options():
    """
    Demonstrate PyTorch pricing for exotic options where no analytical formula exists.
    """
    pricer = PyTorchOptionPricer(n_sims=100000)
    
    print("=" * 80)
    print("Exotic Options Pricing with PyTorch")
    print("=" * 80)
    print()
    
    # Asian Option
    print("1. Asian Call Option (Arithmetic Average)")
    print("-" * 80)
    asian_results = pricer.asian_call_mc(
        S0=100, K=100, r=0.05, sigma=0.2, T=1.0, n_steps=50
    )
    print(f"   Price: ${asian_results['price']:.4f}")
    print(f"   Delta: {asian_results['delta']:.4f}")
    print(f"   Vega:  {asian_results['vega']:.4f}")
    print()
    
    # Basket Option
    print("2. Basket Call Option (3 Assets)")
    print("-" * 80)
    basket_results = pricer.basket_call_mc(
        S0_vec=[100, 105, 95],
        K=100,
        r=0.05,
        sigma_vec=[0.2, 0.25, 0.18],
        correlation=0.5,
        T=1.0
    )
    print(f"   Price: ${basket_results['price']:.4f}")
    print(f"   Deltas: {[f'{d:.4f}' for d in basket_results['deltas']]}")
    print()
    
    # Barrier Option
    print("3. Down-and-Out Barrier Call Option")
    print("-" * 80)
    barrier_results = pricer.barrier_call_mc(
        S0=100, K=100, barrier=90, r=0.05, sigma=0.2, T=1.0,
        n_steps=100, barrier_type='down-and-out'
    )
    print(f"   Price: ${barrier_results['price']:.4f}")
    print(f"   Delta: {barrier_results['delta']:.4f}")
    print(f"   Vega:  {barrier_results['vega']:.4f}")
    print()
    
    print("=" * 80)


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "PyTorch for Financial Derivatives: Greeks via Automatic Differentiation".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    # Run benchmarks
    benchmark_comparison()
    print("\n")
    
    # Demonstrate exotic options
    demonstrate_exotic_options()
    print("\n")
    
    print("✓ All demonstrations completed successfully!")
    print()

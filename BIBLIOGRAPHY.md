# Bibliography - PyTorch for Financial Derivatives and Differentiable Programming

## COMPLETE REFERENCE LIST

This bibliography contains all citations referenced in the PyTorch for Financial Derivatives package, including papers on automatic differentiation, backpropagation, financial derivatives pricing, and quantum machine learning.

---

## TABLE OF CONTENTS

1. [Foundational Papers - Automatic Differentiation](#foundational-papers---automatic-differentiation)
2. [Backpropagation and Neural Networks](#backpropagation-and-neural-networks)
3. [Financial Derivatives and Greeks](#financial-derivatives-and-greeks)
4. [Deep Learning for Finance](#deep-learning-for-finance)
5. [Quantum Machine Learning](#quantum-machine-learning)
6. [Software and Frameworks](#software-and-frameworks)
7. [Books and Textbooks](#books-and-textbooks)
8. [Online Resources](#online-resources)

---

## FOUNDATIONAL PAPERS - AUTOMATIC DIFFERENTIATION

### Linnainmaa (1970) - Original AD Discovery

**Linnainmaa, Seppo.** 1970. "The Representation of the Cumulative Rounding Error of an Algorithm as a Taylor Expansion of the Local Rounding Errors." Master's thesis, University of Helsinki.

*Note: This pioneering work introduced automatic differentiation in its modern form, describing reverse-mode accumulation - the mathematical essence of backpropagation.*

### Griewank (2012) - Comprehensive Treatment

**Griewank, Andreas, and Andrea Walther.** 2008. *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation.* 2nd ed. Philadelphia: SIAM.

*The definitive reference on algorithmic differentiation theory and implementation.*

---

## BACKPROPAGATION AND NEURAL NETWORKS

### Kelley (1960) - Control Theory Origins

**Kelley, Henry J.** 1960. "Gradient Theory of Optimal Flight Paths." *ARS Journal* 30 (10): 947–954. https://doi.org/10.2514/8.5282

*Early development of gradient methods for optimal control - precursor to backpropagation.*

### Bryson (1961) - Dynamic Programming

**Bryson, Arthur E.** 1961. "A Gradient Method for Optimizing Multi-Stage Allocation Processes." In *Proceedings of the Harvard University Symposium on Digital Computers and Their Applications.*

*Another early contribution to gradient-based optimization in control theory.*

### Werbos (1974) - First Neural Network Application

**Werbos, Paul J.** 1974. "Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences." PhD dissertation, Harvard University.

*First application of backpropagation to neural networks, though largely unnoticed at the time.*

### Rumelhart, Hinton & Williams (1986) - The Breakthrough

**Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams.** 1986. "Learning Representations by Back-Propagating Errors." *Nature* 323 (6088): 533–536. https://doi.org/10.1038/323533a0

*The seminal paper that brought backpropagation to mainstream machine learning. This work catalyzed the neural network renaissance and established the foundations of modern deep learning.*

### LeCun et al. (1989) - Convolutional Networks

**LeCun, Yann, Bernhard Boser, John S. Denker, Donnie Henderson, Richard E. Howard, Wayne Hubbard, and Lawrence D. Jackel.** 1989. "Backpropagation Applied to Handwritten Zip Code Recognition." *Neural Computation* 1 (4): 541–551. https://doi.org/10.1162/neco.1989.1.4.541

*Demonstrated backpropagation for convolutional neural networks.*

### Hochreiter & Schmidhuber (1997) - LSTM

**Hochreiter, Sepp, and Jürgen Schmidhuber.** 1997. "Long Short-Term Memory." *Neural Computation* 9 (8): 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

*Introduced LSTM to address vanishing gradient problem in recurrent networks.*

### Hinton et al. (2006) - Deep Learning Renaissance

**Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh.** 2006. "A Fast Learning Algorithm for Deep Belief Nets." *Neural Computation* 18 (7): 1527–1554. https://doi.org/10.1162/neco.2006.18.7.1527

*Showed that deep networks could be pre-trained layer-by-layer, reigniting interest in deep learning.*

### Krizhevsky, Sutskever & Hinton (2012) - AlexNet

**Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton.** 2012. "ImageNet Classification with Deep Convolutional Neural Networks." In *Advances in Neural Information Processing Systems 25 (NIPS 2012)*, edited by F. Pereira, C.J.C. Burges, L. Bottou, and K.Q. Weinberger, 1097–1105. Red Hook, NY: Curran Associates, Inc.

*AlexNet's breakthrough ImageNet performance demonstrated that deep networks with backpropagation could scale to real-world problems.*

---

## FINANCIAL DERIVATIVES AND GREEKS

### Giles & Glasserman (2006) - "Smoking Adjoints"

**Giles, Michael B., and Paul Glasserman.** 2006. "Smoking Adjoints: Fast Monte Carlo Greeks." *Risk Magazine* 19 (1): 92–96.

*The seminal paper showing that adjoint algorithmic differentiation (AAD) could revolutionize Greeks computation. All Greeks can be computed in roughly the same time as a single Monte Carlo pricing - the "smoking adjoints" method.*

**Key contribution:** Demonstrated that reverse-mode AD (the same technique as backpropagation) enables O(1) Greeks computation regardless of the number of parameters.

### Capriotti & Giles (2011) - Adjoint Greeks Made Easy

**Capriotti, Luca, and Michael B. Giles.** 2011. "Algorithmic Differentiation: Adjoint Greeks Made Easy." *SSRN Electronic Journal*, ahead of print. https://doi.org/10.2139/ssrn.1801522

*Practical guide to implementing adjoint algorithmic differentiation for derivatives pricing. Provides accessible introduction to AAD techniques for quantitative analysts.*

### Capriotti & Giles (2012) - Risk.Net Article

**"Adjoint Greeks Made Easy - Risk.Net."** 2012. *Risk.Net*, November 1. https://www.risk.net/risk-management/2221665/adjoint-greeks-made-easy

*Accessible explanation of adjoint methods for computing Greeks, making AAD techniques available to practitioners.*

### Capriotti & Giles (2023) - 15 Years Retrospective

**Capriotti, Luca, and Michael B. Giles.** 2023. "15 Years of Adjoint Algorithmic Differentiation in Finance." *SSRN Electronic Journal*, ahead of print. https://doi.org/10.2139/ssrn.4588939

*Comprehensive retrospective reviewing 15 years of adjoint AD adoption in finance since the "Smoking Adjoints" paper. Discusses impact, applications, and future directions.*

### Giles (2008) - Monte Carlo Greeks

**Giles, Michael B.** 2008. "Collected Matrix Derivative Results for Forward and Reverse Mode Algorithmic Differentiation." In *Advances in Automatic Differentiation*, edited by Christian H. Bischof, H. Martin Bücker, Paul Hovland, Uwe Naumann, and Jean Utke, 35–44. Berlin: Springer. https://doi.org/10.1007/978-3-540-68942-3_4

*Technical reference for matrix derivatives in forward and reverse mode AD.*

---

## DEEP LEARNING FOR FINANCE

### Hutchinson, Lo & Poggio (1994) - Early Neural Networks in Finance

**Hutchinson, James M., Andrew W. Lo, and Tomaso Poggio.** 1994. "A Nonparametric Approach to Pricing and Hedging Derivative Securities Via Learning Networks." *NBER Working Paper* No. 4718. Cambridge, MA: National Bureau of Economic Research. https://doi.org/10.3386/w4718

*Early work applying neural networks to derivatives pricing and hedging, predating modern deep learning.*

### Ferguson & Green (2018) - "Deeply Learning Derivatives"

**Ferguson, Ryan, and Andrew Green.** 2018. "Deeply Learning Derivatives." arXiv preprint arXiv:1809.02233. https://arxiv.org/abs/1809.02233

*Seminal paper demonstrating that deep neural networks can approximate complex derivative pricing functions with million-fold speedups. Showed that PyTorch's autograd can compute Greeks from these approximations.*

**Key result:** Neural network approximations trained on Monte Carlo data can achieve inference times millions of times faster than traditional pricing while maintaining accuracy.

### Ruf & Wang (2020) - Literature Review

**Ruf, Johannes, and Weiguan Wang.** 2020. "Neural Networks for Option Pricing and Hedging: A Literature Review." *Journal of Computational Finance* 24 (1): 1–46. https://doi.org/10.21314/JCF.2020.390

*Comprehensive review of neural network applications in option pricing and hedging.*

### Buehler et al. (2019) - Deep Hedging

**Buehler, Hans, Lukas Gonon, Josef Teichmann, and Ben Wood.** 2019. "Deep Hedging." *Quantitative Finance* 19 (8): 1271–1291. https://doi.org/10.1080/14697688.2019.1571683

*Introduces deep hedging framework using reinforcement learning and automatic differentiation.*

### Gnoatto, Picarelli & Reisinger (2020) - Deep xVA Solver

**Gnoatto, Alessandro, Athena Picarelli, and Christoph Reisinger.** 2020. "Deep xVA Solver – A Neural Network Based Counterparty Credit Risk Management Framework." arXiv preprint arXiv:2005.02633. https://arxiv.org/abs/2005.02633

*Neural network-based framework for xVA computation using automatic differentiation.*

---

## QUANTUM MACHINE LEARNING

### Schuld et al. (2019) - Quantum Gradients

**Schuld, Maria, Ville Bergholm, Christian Gogolin, Josh Izaac, and Nathan Killoran.** 2019. "Evaluating Analytic Gradients on Quantum Hardware." *Physical Review A* 99 (3): 032331. https://doi.org/10.1103/PhysRevA.99.032331

*Introduces parameter-shift rule for computing gradients of quantum circuits - enables differentiable quantum computing.*

### Mitarai et al. (2018) - Quantum Circuit Learning

**Mitarai, Kosuke, Makoto Negoro, Masahiro Kitagawa, and Keisuke Fujii.** 2018. "Quantum Circuit Learning." *Physical Review A* 98 (3): 032309. https://doi.org/10.1103/PhysRevA.98.032309

*Framework for training parameterized quantum circuits using gradient descent.*

### Bergholm et al. (2018) - PennyLane

**Bergholm, Ville, Josh Izaac, Maria Schuld, Christian Gogolin, M. Sohaib Alam, Shahnawaz Ahmed, Juan Miguel Arrazola, et al.** 2018. "PennyLane: Automatic Differentiation of Hybrid Quantum-Classical Computations." arXiv preprint arXiv:1811.04968. https://arxiv.org/abs/1811.04968

*Introduction to PennyLane - a framework for differentiable quantum computing that integrates with PyTorch, TensorFlow, and other AD frameworks.*

### Killoran et al. (2019) - Continuous Variable Quantum Computing

**Killoran, Nathan, Thomas R. Bromley, Juan Miguel Arrazola, Maria Schuld, Nicolás Quesada, and Seth Lloyd.** 2019. "Continuous-Variable Quantum Neural Networks." *Physical Review Research* 1 (3): 033063. https://doi.org/10.1103/PhysRevResearch.1.033063

*Extends quantum neural networks to continuous-variable systems.*

---

## SOFTWARE AND FRAMEWORKS

### PyTorch

**Paszke, Adam, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, et al.** 2019. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." In *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, edited by H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, 8024–8035. Red Hook, NY: Curran Associates, Inc.

*Official PyTorch paper describing the framework's design and automatic differentiation implementation.*

**URL:** https://pytorch.org

### TensorFlow

**Abadi, Martín, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, et al.** 2016. "TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems." arXiv preprint arXiv:1603.04467. https://arxiv.org/abs/1603.04467

*TensorFlow system paper describing Google's machine learning framework.*

**URL:** https://tensorflow.org

### JAX

**Bradbury, James, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang.** 2018. "JAX: Composable Transformations of Python+NumPy Programs." Version 0.3.13. http://github.com/google/jax

*JAX framework for composable transformations including automatic differentiation.*

**URL:** https://github.com/google/jax

### PennyLane Documentation

**Xanadu Quantum Technologies Inc.** 2024. "PennyLane Documentation." https://pennylane.ai

*Official documentation for PennyLane quantum machine learning framework.*

---

## BOOKS AND TEXTBOOKS

### Goodfellow, Bengio & Courville (2016) - Deep Learning

**Goodfellow, Ian, Yoshua Bengio, and Aaron Courville.** 2016. *Deep Learning.* Cambridge, MA: MIT Press.

*Comprehensive textbook on deep learning, including detailed treatment of backpropagation and automatic differentiation (Chapter 6).*

**URL:** https://www.deeplearningbook.org

### Griewank & Walther (2008) - Algorithmic Differentiation

**Griewank, Andreas, and Andrea Walther.** 2008. *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation.* 2nd ed. Philadelphia: SIAM.

*The definitive mathematical treatment of automatic differentiation theory and techniques.*

### Hull (2018) - Options, Futures, and Other Derivatives

**Hull, John C.** 2018. *Options, Futures, and Other Derivatives.* 10th ed. Pearson.

*Standard textbook for derivatives pricing, including Greeks and risk management.*

### Shreve (2004) - Stochastic Calculus for Finance

**Shreve, Steven E.** 2004. *Stochastic Calculus for Finance II: Continuous-Time Models.* New York: Springer.

*Mathematical foundations for continuous-time finance and derivatives pricing.*

### Nielsen & Chuang (2010) - Quantum Computation

**Nielsen, Michael A., and Isaac L. Chuang.** 2010. *Quantum Computation and Quantum Information.* 10th Anniversary ed. Cambridge: Cambridge University Press.

*Standard textbook on quantum computing, providing foundations for quantum machine learning.*

---

## ONLINE RESOURCES

### PyTorch Tutorials

**PyTorch Team.** 2024. "PyTorch Tutorials - Automatic Differentiation with torch.autograd." https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

*Official PyTorch tutorial on automatic differentiation.*

### TensorFlow Documentation

**TensorFlow Team.** 2024. "Introduction to Gradients and Automatic Differentiation." https://www.tensorflow.org/guide/autodiff

*TensorFlow guide to automatic differentiation with tf.GradientTape.*

### PennyLane Tutorials

**Xanadu Quantum Technologies Inc.** 2024. "PennyLane Tutorials - Quantum Gradients." https://pennylane.ai/qml/demos/tutorial_backprop.html

*Tutorial on quantum gradients and the parameter-shift rule.*

### Risk.Net - Derivatives Pricing

**Risk.Net.** 2024. "Quantitative Finance and Risk Management." https://www.risk.net

*Industry publication covering derivatives pricing, risk management, and quantitative finance.*

### arXiv - Quantitative Finance

**arXiv.org.** 2024. "Quantitative Finance Archive." https://arxiv.org/archive/q-fin

*Preprint server for quantitative finance research papers.*

---

## CITATION FORMATS

### BibTeX Format

```bibtex
% Foundational AD
@mastersthesis{linnainmaa1970,
  author = {Linnainmaa, Seppo},
  title = {The Representation of the Cumulative Rounding Error of an Algorithm as a Taylor Expansion of the Local Rounding Errors},
  school = {University of Helsinki},
  year = {1970}
}

% Backpropagation
@phdthesis{werbos1974,
  author = {Werbos, Paul J.},
  title = {Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences},
  school = {Harvard University},
  year = {1974}
}

@article{rumelhart1986,
  author = {Rumelhart, David E. and Hinton, Geoffrey E. and Williams, Ronald J.},
  title = {Learning Representations by Back-Propagating Errors},
  journal = {Nature},
  volume = {323},
  number = {6088},
  pages = {533--536},
  year = {1986},
  doi = {10.1038/323533a0}
}

% Financial Greeks - Smoking Adjoints
@article{giles2006,
  author = {Giles, Michael B. and Glasserman, Paul},
  title = {Smoking Adjoints: Fast Monte Carlo Greeks},
  journal = {Risk Magazine},
  volume = {19},
  number = {1},
  pages = {92--96},
  year = {2006}
}

@article{capriotti2011,
  author = {Capriotti, Luca and Giles, Michael B.},
  title = {Algorithmic Differentiation: Adjoint Greeks Made Easy},
  journal = {SSRN Electronic Journal},
  year = {2011},
  doi = {10.2139/ssrn.1801522}
}

@misc{capriotti2012,
  author = {{Risk.Net}},
  title = {Adjoint Greeks Made Easy},
  year = {2012},
  month = {November},
  url = {https://www.risk.net/risk-management/2221665/adjoint-greeks-made-easy}
}

@article{capriotti2023,
  author = {Capriotti, Luca and Giles, Michael B.},
  title = {15 Years of Adjoint Algorithmic Differentiation in Finance},
  journal = {SSRN Electronic Journal},
  year = {2023},
  doi = {10.2139/ssrn.4588939}
}

% Deep Learning for Finance
@article{ferguson2018,
  author = {Ferguson, Ryan and Green, Andrew},
  title = {Deeply Learning Derivatives},
  journal = {arXiv preprint arXiv:1809.02233},
  year = {2018},
  url = {https://arxiv.org/abs/1809.02233}
}

@techreport{hutchinson1994,
  author = {Hutchinson, James M. and Lo, Andrew W. and Poggio, Tomaso},
  title = {A Nonparametric Approach to Pricing and Hedging Derivative Securities Via Learning Networks},
  institution = {National Bureau of Economic Research},
  type = {NBER Working Paper},
  number = {4718},
  year = {1994},
  doi = {10.3386/w4718}
}

% Quantum ML
@article{schuld2019,
  author = {Schuld, Maria and Bergholm, Ville and Gogolin, Christian and Izaac, Josh and Killoran, Nathan},
  title = {Evaluating Analytic Gradients on Quantum Hardware},
  journal = {Physical Review A},
  volume = {99},
  number = {3},
  pages = {032331},
  year = {2019},
  doi = {10.1103/PhysRevA.99.032331}
}

@article{bergholm2018,
  author = {Bergholm, Ville and Izaac, Josh and Schuld, Maria and Gogolin, Christian and Alam, M. Sohaib and Ahmed, Shahnawaz and Arrazola, Juan Miguel and others},
  title = {PennyLane: Automatic Differentiation of Hybrid Quantum-Classical Computations},
  journal = {arXiv preprint arXiv:1811.04968},
  year = {2018},
  url = {https://arxiv.org/abs/1811.04968}
}

% Software
@incollection{paszke2019,
  author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and others},
  title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  booktitle = {Advances in Neural Information Processing Systems 32 (NeurIPS 2019)},
  editor = {Wallach, H. and Larochelle, H. and Beygelzimer, A. and d'Alché-Buc, F. and Fox, E. and Garnett, R.},
  pages = {8024--8035},
  year = {2019},
  publisher = {Curran Associates, Inc.}
}

% Books
@book{goodfellow2016,
  author = {Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron},
  title = {Deep Learning},
  publisher = {MIT Press},
  year = {2016},
  url = {https://www.deeplearningbook.org}
}

@book{griewank2008,
  author = {Griewank, Andreas and Walther, Andrea},
  title = {Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation},
  edition = {2nd},
  publisher = {SIAM},
  address = {Philadelphia},
  year = {2008}
}
```

---

## APA FORMAT (7th Edition)

### Key Papers

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, *323*(6088), 533–536. https://doi.org/10.1038/323533a0

Giles, M. B., & Glasserman, P. (2006). Smoking adjoints: Fast Monte Carlo Greeks. *Risk Magazine*, *19*(1), 92–96.

Capriotti, L., & Giles, M. B. (2011). Algorithmic differentiation: Adjoint Greeks made easy. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.1801522

Adjoint Greeks made easy - Risk.Net. (2012, November 1). *Risk.Net*. https://www.risk.net/risk-management/2221665/adjoint-greeks-made-easy

Capriotti, L., & Giles, M. B. (2023). 15 years of adjoint algorithmic differentiation in finance. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.4588939

Ferguson, R., & Green, A. (2018). Deeply learning derivatives. *arXiv preprint arXiv:1809.02233*. https://arxiv.org/abs/1809.02233

Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., & Killoran, N. (2019). Evaluating analytic gradients on quantum hardware. *Physical Review A*, *99*(3), 032331. https://doi.org/10.1103/PhysRevA.99.032331

Bergholm, V., Izaac, J., Schuld, M., Gogolin, C., Alam, M. S., Ahmed, S., Arrazola, J. M., et al. (2018). PennyLane: Automatic differentiation of hybrid quantum-classical computations. *arXiv preprint arXiv:1811.04968*. https://arxiv.org/abs/1811.04968

### Books

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org

Griewank, A., & Walther, A. (2008). *Evaluating derivatives: Principles and techniques of algorithmic differentiation* (2nd ed.). SIAM.

---

## CHICAGO STYLE (17th Edition)

Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. "Learning Representations by Back-Propagating Errors." *Nature* 323, no. 6088 (1986): 533–36. https://doi.org/10.1038/323533a0.

Giles, Michael B., and Paul Glasserman. "Smoking Adjoints: Fast Monte Carlo Greeks." *Risk Magazine* 19, no. 1 (2006): 92–96.

Capriotti, Luca, and Michael B. Giles. "Algorithmic Differentiation: Adjoint Greeks Made Easy." *SSRN Electronic Journal*, 2011. https://doi.org/10.2139/ssrn.1801522.

"Adjoint Greeks Made Easy - Risk.Net." *Risk.Net*, November 1, 2012. https://www.risk.net/risk-management/2221665/adjoint-greeks-made-easy.

Capriotti, Luca, and Michael B. Giles. "15 Years of Adjoint Algorithmic Differentiation in Finance." *SSRN Electronic Journal*, 2023. https://doi.org/10.2139/ssrn.4588939.

Ferguson, Ryan, and Andrew Green. "Deeply Learning Derivatives." arXiv preprint arXiv:1809.02233, 2018. https://arxiv.org/abs/1809.02233.

Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. *Deep Learning*. Cambridge, MA: MIT Press, 2016. https://www.deeplearningbook.org.

---

## NOTES FOR USERS

### How to Cite This Package

If you use this package in your work, please cite:

**Plain text:**
PyTorch for Financial Derivatives: Computing Greeks via Automatic Differentiation (2024). GitHub repository: https://github.com/roguetrainer/diff-prog-for-derivative-greeks/

**BibTeX:**
```bibtex
@misc{pytorch_derivatives2024,
  title = {PyTorch for Financial Derivatives: Computing Greeks via Automatic Differentiation},
  year = {2024},
  url = {https://github.com/roguetrainer/diff-prog-for-derivative-greeks/},
  note = {Educational package demonstrating automatic differentiation for derivatives pricing}
}
```

### Recommended Reading Order

1. **Start here:** Rumelhart, Hinton & Williams (1986) - Understand backpropagation
2. **Finance application:** Giles & Glasserman (2006) - See AAD for Greeks
3. **Modern implementation:** Ferguson & Green (2018) - Neural networks for derivatives
4. **Quantum extension:** Schuld et al. (2019) - Differentiable quantum circuits
5. **Comprehensive treatment:** Goodfellow et al. (2016) Chapter 6 - Deep dive into AD

### Key Themes

- **1970-1986:** Foundations of automatic differentiation and backpropagation
- **1986-2006:** Neural networks and control applications
- **2006:** Breakthrough application to financial Greeks ("Smoking Adjoints")
- **2011-2023:** Maturation of AAD in finance
- **2018-present:** Deep learning frameworks make AD accessible
- **2018-present:** Extension to quantum computing

---

## SUPPLEMENTARY MATERIALS

### GitHub Repositories

- **TorchQuant:** https://github.com/jialuechen/torchquant
- **PFHedge:** https://github.com/pfnet-research/pfhedge
- **This Package:** https://github.com/roguetrainer/diff-prog-for-derivative-greeks/

### Online Courses and Tutorials

- **PyTorch Tutorials:** https://pytorch.org/tutorials
- **Deep Learning Specialization (Coursera):** Andrew Ng's deep learning course
- **Fast.ai:** Practical deep learning for coders

### Professional Organizations

- **International Association for Quantitative Finance (IAQF):** https://www.iaqf.org
- **Global Association of Risk Professionals (GARP):** https://www.garp.org

---

## VERSION INFORMATION

**Bibliography Version:** 1.0  
**Last Updated:** November 2024  
**Maintained by:** PyTorch Derivatives Package Authors  
**License:** MIT (same as package)

---

## END OF BIBLIOGRAPHY

Total entries: 40+ references spanning 1960-2024

For questions or additions to this bibliography, please open an issue at:
https://github.com/roguetrainer/diff-prog-for-derivative-greeks/

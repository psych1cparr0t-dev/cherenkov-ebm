# Reference Lists for Cherenkov EBM Positioning

Compiled from authoritative sources. Annotated for relevance to the Cherenkov EBM project where appropriate.

---

## 1. Machine Learning Algorithms

Sources: scikit-learn 1.7 documentation, Murphy *Probabilistic Machine Learning* (2022/2023), Coursera 2026 ML overview.

### Supervised — regression
- Linear regression (OLS)
- Ridge regression
- Lasso regression
- ElasticNet
- Polynomial regression
- Bayesian linear regression
- Gaussian Process regression
- Support Vector Regression (SVR)
- Kernel ridge regression
- Quantile regression
- Generalized Linear Models (GLM) — Poisson, Gamma, Tweedie
- Isotonic regression

### Supervised — classification
- Logistic regression
- Naive Bayes (Gaussian, Multinomial, Bernoulli, Complement)
- k-Nearest Neighbors (kNN)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Support Vector Machines (SVM, kernel SVM)
- Decision trees (CART, C4.5)
- Multinomial logistic / softmax regression

### Ensembles
- Random Forest
- Extra Trees
- Bagging
- AdaBoost
- Gradient Boosting Machines (GBM)
- XGBoost
- LightGBM
- CatBoost
- Histogram-based GBM (sklearn)
- Stacking / blending

### Neural networks
- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN, LSTM, GRU)
- Transformer
- Vision Transformer (ViT)
- Graph Neural Network (GNN, GCN, GAT)
- Autoencoder, Variational Autoencoder (VAE)
- Generative Adversarial Network (GAN)
- Diffusion model (DDPM, score-based)
- Normalizing flow (RealNVP, Glow)
- **Energy-Based Model (EBM)** — the family Cherenkov belongs to
- Joint Embedding Predictive Architecture (JEPA, I-JEPA, V-JEPA) — LeCun's framework, AMI Labs core
- Mixture Density Network (MDN)

### Unsupervised — clustering
- K-Means, K-Medoids, K-Means++
- Hierarchical / agglomerative clustering
- DBSCAN, HDBSCAN, OPTICS
- Gaussian Mixture Model (GMM, EM)
- Mean Shift
- Spectral clustering
- BIRCH
- Affinity propagation

### Unsupervised — dimensionality reduction & manifold
- PCA, Kernel PCA, Sparse PCA, Incremental PCA
- ICA
- Factor Analysis
- NMF
- t-SNE
- UMAP
- LLE (Locally Linear Embedding)
- Isomap
- MDS (Multidimensional Scaling)
- Laplacian Eigenmaps
- Diffusion Maps
- Autoencoder dim reduction

### Unsupervised — density estimation & anomaly
- Kernel Density Estimation (KDE)
- One-Class SVM
- Isolation Forest
- Local Outlier Factor (LOF)
- Elliptic Envelope

### Self-supervised / representation learning
- Contrastive learning (SimCLR, MoCo, BYOL, SimSiam)
- Masked autoencoders (MAE, BEiT)
- DINO, DINOv2
- VICReg, Barlow Twins
- JEPA family (see above)

### Reinforcement learning
- Q-Learning
- SARSA
- DQN, Double DQN, Dueling DQN
- Policy Gradient (REINFORCE)
- Actor-Critic, A2C, A3C
- PPO
- TRPO
- DDPG, TD3, SAC
- Model-based RL (MuZero, Dreamer, world models)

### Probabilistic graphical models
- Bayesian network
- Hidden Markov Model (HMM)
- Conditional Random Field (CRF)
- Markov Random Field (MRF)
- Latent Dirichlet Allocation (LDA)

### Bandits / online
- Multi-armed bandit (epsilon-greedy, UCB, Thompson sampling)
- Contextual bandits
- Online gradient descent

---

## 2. Quantum Algorithms

Source: **Quantum Algorithm Zoo** (Stephen Jordan et al, Microsoft Research, https://quantumalgorithmzoo.org), Wikipedia Quantum Algorithm catalogue, Bharti et al. "Noisy intermediate-scale quantum algorithms" *Rev. Mod. Phys.* 2022.

### Algebraic & number-theoretic (exponential / superpolynomial speedup)
- Deutsch's algorithm (1985) — first quantum algorithm
- Deutsch–Jozsa
- Bernstein–Vazirani
- Simon's algorithm
- Shor's algorithm — integer factoring, discrete log
- Hidden Subgroup Problem (HSP) — generalises Shor
- Quantum Fourier Transform (QFT)
- Quantum Phase Estimation (QPE / Kitaev)
- Pell's equation, principal ideal
- Order finding
- Group commutator estimation

### Search & optimisation (typically quadratic speedup)
- Grover's search
- Amplitude amplification
- Amplitude estimation
- Quantum counting
- Quantum walks (continuous-time, discrete-time, Szegedy)
- Element distinctness
- Triangle finding
- Matrix product verification
- Welded tree (exponential speedup, oracle problem)
- Quantum minimum finding

### Linear algebra & differential equations
- HHL (Harrow–Hassidim–Lloyd) — linear systems Ax = b
- Quantum Singular Value Transformation (QSVT)
- Block encoding
- Quantum Linear Systems via Linear Combination of Unitaries (LCU)
- Quantum gradient descent
- Quantum recommendation systems (de-quantised by Tang)
- Quantum PageRank
- Quantum differential equation solvers (linear ODEs, PDEs)

### Simulation
- Hamiltonian simulation (Trotter–Suzuki)
- Hamiltonian simulation via QSVT / qubitization
- Quantum walk simulation
- Open-system / Lindbladian simulation
- Real-time evolution
- Imaginary time evolution

### Variational / NISQ-era
- Variational Quantum Eigensolver (VQE)
- Quantum Approximate Optimization Algorithm (QAOA)
- Quantum Alternating Operator Ansatz (QAOA generalisation)
- Variational Quantum Linear Solver (VQLS)
- Quantum Natural Gradient
- Subspace expansion methods

### Quantum machine learning
- Quantum Support Vector Machine (qSVM)
- Quantum kernel methods
- Quantum Principal Component Analysis (qPCA)
- Quantum k-means
- Quantum neural networks (parameterised quantum circuits)
- Quantum Boltzmann machines
- Quantum GANs
- Born machines

### Cryptography & security
- BB84 (quantum key distribution)
- E91 (entanglement-based QKD)
- Quantum coin flipping
- Quantum digital signatures
- Quantum random number generation

### Error correction & fault tolerance (algorithmic primitives)
- Shor 9-qubit code
- Steane code
- Surface code decoding
- Magic state distillation
- Topological codes

### Other notable
- Adiabatic quantum computation
- Measurement-based quantum computation (one-way)
- Quantum teleportation (subroutine)
- Superdense coding

**Cherenkov relevance note**: VQE and QAOA are the closest quantum analogues to the Cherenkov synthesizer — both fit a parameterised ansatz to minimise an objective. The Cherenkov primitive library could in principle be re-expressed as a quantum ansatz, but the current implementation is purely classical.

---

## 3. Existing Open-Source EBM Implementations

Sources: `yataobian/awesome-ebm` (curated catalogue), `WilliamYi96/Awesome-Energy-Based-Models`, `jxzhangjhu/awesome-energy-based-model`, GitHub topic `energy-based-model`.

### Libraries / frameworks
- **TorchEBM** — PyTorch library, Contrastive Divergence + Score Matching + NCE losses, Langevin/HMC samplers, CUDA. https://github.com/soran-ghaderi/torchebm
- **mini-ebm** — Minimalist educational EBM in PyTorch (2025). https://github.com/yataobian/mini-ebm
- **Learnergy** — Energy-based machine learners (Roder, de Rosa, Papa, 2020). https://github.com/gugarosa/learnergy
- **Nieme** (2009, Maes) — Large-scale energy-based models. Original code page is missing.
- **EB-JEPA** — Facebook Research lightweight library for Energy-Based JEPA (image, video, action-conditioned world models). https://github.com/facebookresearch/eb_jepa — **directly relevant to AMI Labs**

### Image generation / unconditional EBMs
- **openai/ebm_code_release** — "Implicit Generation and Generalization with Energy-Based Models" (Du & Mordatch 2019). CIFAR-10, ImageNet 128, dSprites. https://github.com/openai/ebm_code_release
- **MichaelArbel/GeneralizedEBM** — "Generalized Energy Based Models" (ICLR 2021). https://github.com/MichaelArbel/GeneralizedEBM
- **sndnyang/mebm** — "M-EBM: Towards Understanding the Manifolds of Energy-Based Models" (CIFAR10, CelebA-HQ, ImageNet 32). https://github.com/sndnyang/mebm
- **JEM (Joint Energy Models)** — Grathwohl et al. "Your Classifier is Secretly an Energy-Based Model" (ICLR 2020). Multiple reference implementations.
- **VAEBM** — Xiao et al., NeurIPS 2021. Symbiosis of VAE and EBM.
- **Patchwise Generative ConvNet** — internal-image EBMs.
- **Flow Contrastive Estimation** — Gao et al., CVPR 2020.
- **Diffusion Recovery Likelihood** — Gao et al., ICLR 2021.

### Discriminative / regression / structured prediction
- **fregu856/ebms_proposals** — "Learning Proposals for Practical Energy-Based Regression". https://github.com/fregu856/ebms_proposals
- **Energy-Based Models for Deep Probabilistic Regression** — Gustafsson et al., ECCV 2020.
- **Energy-Based Learning for Scene Graph Generation** — Suhail et al.

### Continual / lifelong learning
- **ShuangLI59/ebm-continual-learning** — Li, Du, van de Ven, Mordatch. https://github.com/ShuangLI59/ebm-continual-learning

### Domain-specific applications
- **facebookresearch/protein-ebm** — "Energy-based models for atomic-resolution protein conformations" (Du, Meier, Ma, Fergus, Rives, ICLR 2020). Transformer architecture, rotamer library negative sampling. https://github.com/facebookresearch/protein-ebm
- **dfuchsgruber/graph-ebm** — "Energy-based Epistemic Uncertainty for Graph Neural Networks" (NeurIPS 2024). https://github.com/dfuchsgruber/graph-ebm
- **Generative PointNet** — Xie et al., 3D point clouds.
- **Trajectory Prediction with Latent Belief EBM** — Pang et al.
- **Latent Space EBM for Symbol-Vector Coupling** — Pang & Wu, text generation.
- **Active learning for domain adaptation** (energy-based) — AAAI 2022.

### Sampling / training methodology
- **Improved Contrastive Divergence Training of EBMs** — Du, Li, Tenenbaum, Mordatch.
- **No MCMC for me** — Grathwohl et al., ICLR 2021. Amortized sampling.
- **Energy Discrepancies** — Schröder et al., NeurIPS 2023. Score-independent loss.
- **Convex Potential Mirror Langevin** — Yang et al., NeurIPS 2025.

### Hopfield / associative memory variants
- **Dense Associative Memory with Epanechnikov energy** — Hoover, Balasubramanian, Krotov, Ram. NeurIPS 2025.

### Reasoning / iterative
- **Difference-of-Convex Functions Approach to Energy-Based Iterative Reasoning** — Tschernutter et al., NeurIPS 2025.
- **Riemannian Metrics from Energy-Based Models** — Béthune et al., NeurIPS 2025. "Follow the Energy, Find the Path".

### Test-time adaptation
- **Rethinking Entropy in Test-Time Adaptation** — Park et al., NeurIPS 2025. Energy duality.

### Catalogues / survey resources
- https://github.com/yataobian/awesome-ebm — most current, updated through NeurIPS 2025
- https://github.com/jxzhangjhu/awesome-energy-based-model
- https://github.com/WilliamYi96/Awesome-Energy-Based-Models
- https://energy-based-model.github.io/ — Yilun Du's group site
- IJCAI-17 Tutorial: Energy-based machine learning (Osogami, Dasgupta)
- LeCun et al. (2006) "A tutorial on energy-based learning" — foundational reference

---

## What's already covered vs what Cherenkov adds

**Already in the literature**:
- Neural-net-parameterised energy functions (every major EBM library)
- Contrastive Divergence, Score Matching, NCE (TorchEBM, JEM, EB-JEPA)
- Langevin/HMC sampling for generative tasks (openai/ebm_code_release, M-EBM, GeneralizedEBM)
- Joint energy + classifier (JEM)
- Domain-specific EBMs for proteins, graphs, point clouds, trajectories
- JEPA / world model EBMs (EB-JEPA — most relevant prior art for AMI)

**What Cherenkov does that none of the above do**:
- Energy function as a sum of *named, interpretable geometric primitives* — not a neural network
- Autonomous primitive synthesis from residual geometry — no MCMC, no Langevin
- The "fingerprint" — sequence of primitive families retained — as a language-agnostic geometric description of the domain
- Plug-in parsing layer that augments any downstream model

**What's not yet validated**:
- Real-world results beyond synthetic data (the open question)
- Whether the fingerprint is consistent across subjects on the same task (the AMI-shaped result)
- Whether the parsing layer transfers — fit on one dataset, evaluate on another

---

*Compiled April 27, 2026 — Cherenkov Inc. internal.*

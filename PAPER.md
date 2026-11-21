# Physics-based and MACE-assisted Hamiltonian Construction

Korea University AA Co-E Creator Club (Nov. – Dec. 2026)  
Department of Materials Science and Engineering

**Team Members:** 안서영, 연제우, 이지연, 정윤석, 최준영, 모진석

## 1. Introduction

### 1.1 Project Motivation

As members of an automation club, we sought to explore how machine learning, a key technology in modern automation, could be extended **beyond conventional automation tasks to computational materials science.**

In particular, we were interested in whether machine-learning interatomic potentials (MLIPs), such as MACE, could facilitate the automated construction of physically motivated electronic Hamiltonians without directly relying on DFT-derived electronic-structure data for parameter fitting.

Rather than aiming for a complete reproduction of DFT-level electronic structures, our primary objective was to develop a computational workflow that integrates MACE-derived energies and forces with Slater–Koster tight-binding theory. Through this approach, we sought to construct parameterized Hamiltonians and extract electronic properties, including projected densities of states (PDOS), with minimal manual intervention.

Ultimately, this project represents an exploratory attempt to connect machine-learning-assisted automation with physics-based electronic structure modeling, while investigating the capabilities and limitations of such an approach.

### 1.2 Background and Related Work

Recent advances in machine learning have enabled efficient prediction of electronic structures through data-driven Hamiltonian modeling. MACE-H (2026) employs equivariant many-body message passing to predict DFT Kohn–Sham Hamiltonian matrices directly from atomic configurations [1]. Similarly, SlaKoNet (2025) optimizes physically interpretable Slater–Koster tight-binding parameters using electronic-structure data [2], while Fried et al. (2025) demonstrated the inference of tight-binding parameters from orbital-projected densities of states [3].

Other studies have explored the integration of machine-learned interatomic potentials with electronic-structure models. HAMSTER (2026) combines physically motivated tight-binding Hamiltonians with machine-learned corrections and employs MACE force fields for atomistic simulations [4]. A related framework couples MACE with DeepH-E3 to investigate temperature-dependent electronic properties [5]. More recently, NN-xTB (2026) has incorporated a MACE-based encoder into an adaptive semi-empirical Hamiltonian, using electronic-structure calculations and energy–force training targets [6].

Collectively, these studies demonstrate the potential of combining machine learning with physically structured electronic Hamiltonians.

### 1.3 Objectives and Scope

This project explores **an alternative route to electronic-structure reconstruction** by using energies and forces predicted by a pretrained MACE interatomic potential as reference quantities for parameterizing a Slater–Koster tight-binding Hamiltonian.

Unlike direct Hamiltonian-learning approaches, the present implementation does not predict Hamiltonian matrix elements through an additional neural network. **Instead, a physically parameterized Hamiltonian is constructed from orbital onsite energies and geometry-dependent hopping integrals.** Its parameters are optimized through automatic differentiation against MACE-derived energetic and force constraints, with additional parameter regularization.

The resulting Hamiltonian is diagonalized to obtain electronic eigenstates and orbital-resolved projected densities of states (PDOS). Pt(111) surface adsorption is employed as a model system to investigate the electronic interactions between metal and adsorbate orbitals.

A central objective is to examine whether agreement in atomic energies and forces is sufficient to reproduce physically meaningful electronic structures. Because energy–force information does not uniquely determine the underlying electronic Hamiltonian, comparisons with independently calculated DFT-PDOS are used to assess the limitations of the reconstructed model.

Rather than proposing a universally transferable electronic-structure predictor, this project investigates the opportunities and limitations of energy–force-constrained tight-binding parameterization, with particular emphasis on interpretability, orbital hybridization, and the consistency between atomic energetics and electronic structure.

**Figure 1.** Schematic illustration of the proposed computational workflow. MACE-derived atomic energies and forces are employed as reference quantities for optimizing Slater–Koster tight-binding parameters. The resulting Hamiltonian is diagonalized to obtain orbital-resolved electronic structures, which are subsequently used to investigate metal–adsorbate interactions relevant to oxygen reduction reactions (ORR).

## 2. Theoretical Background

### 2.1 MACE Interatomic Potential

Machine learning interatomic potentials (MLIPs) approximate the potential-energy surface of an atomic system from its chemical composition and geometry. MACE (Message Passing Atomic Cluster Expansion) combines equivariant message passing with many-body atomic cluster features [7]. Local environments are represented using the positions and chemical species of neighboring atoms, while the architecture is designed to respect the symmetries of three-dimensional space. Consequently, the predicted scalar energy is invariant under global translations and rotations, and the derived forces transform consistently with the atomic coordinates.

Unlike an explicit electronic-structure calculation, a conventional MACE interatomic potential does not directly provide orbital eigenstates, Hamiltonian matrix elements, or a projected density of states (PDOS). Its principal outputs are potential energies and atomic forces. The present project uses these outputs as energetic and mechanical reference quantities for a separately parameterized Slater–Koster (S–K) tight-binding model; it does not train a neural-network head to predict a DFT Hamiltonian.

An important distinction is that the final analysis notebook reads previously generated MACE-labeled configurations from MACE_training.extxyz; it does not retrain MACE or repeat the reference relaxation. Therefore, 'without electronic-structure labels' refers specifically to the subsequent S–K fitting step, rather than to the original training history of the pretrained MACE potential. The reconstructed electronic spectra must be validated independently, since an accurate interatomic potential alone does not uniquely specify the electronic Hamiltonian.

### 2.2 Slater-Koster Tight-Binding Theory

In a localized atomic-orbital basis {\|iμ⟩}, the tight-binding Hamiltonian is represented by matrix elements $H_{i\mu,j\nu} = \left\langle i\mu|Ĥ|j\nu \right\rangle$. Diagonal blocks contain onsite orbital energies, whereas off-diagonal blocks describe hopping or hybridization between orbitals on different atoms. For the Pt–adsorbate system, the Hamiltonian can be partitioned into Pt, adsorbate, and Pt–adsorbate coupling blocks. Hermiticity requires $H_{j\nu,i\mu} = H_{i\mu,j\nu}^{\dagger}$ in the present real Γ-point construction, the opposite block is filled by the transpose.

$$H\  = \ \begin{pmatrix}
H_{Pt} & H_{Pt,ads} \\
H_{ads,Pt} & H_{ads}
\end{pmatrix}$$

The Slater–Koster formalism separates each interatomic hopping integral into a radial factor and an angular coefficient determined by the bond direction. For an interatomic vector $Rᵢⱼ\  = \ (\Delta x,\ \Delta y,\ \Delta z)$, the direction cosines are $l\  = \ \Delta x/Rᵢⱼ$, $m\  = \ \Delta y/Rᵢⱼ$, and $n\  = \ \Delta z/Rᵢⱼ$. These direction cosines rotate the σ, π, and δ bond-channel contributions into the global s, p, and d orbital basis. The angular coefficients follow the standard Slater–Koster tables.

$$H_{i\mu,j\nu}\left( \mathbf{R}_{\mathbf{ij}} \right) = \sum_{\lambda}{C_{\mu\nu\lambda}(l,m,n)V_{\lambda}\left( R_{ij} \right)}\quad\text{where}\quad\lambda \in \{\sigma,\pi,\delta\}$$

For instance, the pₓ–pₓ term is $H_{p_{x},p_{x}} = l^{2}V_{pp\sigma} + \left( 1 - l^{2} \right)V_{pp\pi}$. The same approach produces the Pt d–d and Pt d–O p couplings, as well as the additional s-containing couplings. The direction dependence is particularly relevant to the ontop, bridge, and hollow geometries because their metal–adsorbate bond orientations differ even for the same atomic species.

**Table 1.** Slater-Koster angular-coefficient table used to construct the orbital hopping blocks. Here, x, y, and z are the direction cosines of $\widehat{\mathbf{R}}$ with $x^{2} + y^{2} = 1$, and the notation $\alpha = x^{2} + y^{2}$ and $\beta = x^{2} - y^{2}$. Reproduced from Koskinen (2009) [8].

> **Table 1 note:** The original table is embedded as an image in the Word source and is not transcribed here. Refer to Koskinen and Mäkinen (2009) [8] for the full angular-coefficient table.

The original reduced orbital model considered Pt 5d, O 2p, and H 1s states. The final notebook extends this basis to Pt (5d + 6s), O (2s + 2p), and H (1s). This extension introduces new hopping channels, including Pt s–s/s–d, Pt–O d–s/s–p/s–s, and O–O s–p/s–s contributions. Each chemical pair retains a separate repulsive term in addition to its electronic hopping channels. The basis remains orthogonal and non-spin-polarized; thus, the overlap matrix is treated as the identity and explicit spin splitting is not included.

The Hamiltonian in this project is physically constructed from tabulated angular factors, neighbor geometry, onsite levels, and fitted radial parameters. This is distinct from directly predicting every Hamiltonian matrix element with an additional graph neural network. It also means that the resulting orbital interpretation is tied to the chosen basis set and the approximations of the S–K parametrization.

### 2.3 Hamiltonian Diagonalization and PDOS

Once the tight-binding Hamiltonian is assembled, the electronic eigenvalues and eigenvectors are determined by diagonalization. In the orthogonal basis adopted here, the stationary eigenvalue problem takes the usual form $H(k)cₙ(k)\  = \ \varepsilon ₙ(k)cₙ(k)$. The eigenvalues describe the model's allowed one-electron energies, while the squared orbital coefficients quantify each basis orbital's contribution to an eigenstate. (For a nonorthogonal basis, the more general equation would contain an overlap matrix $S(k)$.)

$$H(k)c_{n}(k) = \varepsilon_{n}(k)c_{n}(k);\quad\quad H(k)^{\dagger} = H(k)$$

The density of states counts electronic eigenstates in an energy interval. Orbital-projected DOS is obtained from a projection operator ${\widehat{P}}_{M}$ onto a selected atomic orbital or group M. In an orthonormal basis, ${\widehat{P}}_{M}\  = \ \Sigma\mu \in M\ |\mu\rangle\langle\mu|$; its expectation value on an eigenstate becomes the sum of $|cₙ\mu(k)|²$ over the selected orbitals, consistent with the projector-based definition used in QuantumATK documentation [9].

$$D_{M}(E) = \left( 1/N_k \right)\sum_{\mathbf{k}}\sum_n\left\langle \psi_{n\mathbf{k}}\left| {\widehat{P}}_{M} \right|\psi_{n\mathbf{k}} \right\rangle\delta(E - \varepsilon_{n\mathbf{k}})$$

In these expressions, E denotes the energy variable at which the spectrum is evaluated, whereas $\varepsilon ₙ(k)$ denotes an eigenvalue of the Hamiltonian; thus $\delta\left( E - \varepsilon ₙ(k) \right)$ contributes at each eigenenergy. For numerical plots, the Dirac delta functions are replaced by normalized Gaussian kernels. Orbital PDOS curves are aggregated into Pt 5d, Pt 6s, O 2p, O 2s, and H 1s contributions. The internally determined Fermi level $E_{F}$ sets the plotted energy origin, $E - E_{F}$. These spectra are outputs of the fitted Hamiltonian and are not targets of the energy–force fitting.

For metal–oxygen adsorption, PDOS and orbital hybridization can reveal qualitative changes in Pt–adsorbate coupling. The O\*, OH\*, OOH\*, and O₂-containing surface configurations were selected as model systems relevant to oxygen reduction reactions (ORR).

## 3. Computational Methodology

### 3.1 Pt(111) Surface and Adsorption Models

The model system is a Pt(111) surface slab with oxygen-containing adsorbates relevant to the oxygen reduction reaction (ORR). The notebook permits bare Pt(111) as well as O, O₂, OH, and OOH placed at ontop, bridge, fcc hollow, or hcp hollow sites (17 nominal selections including the bare slab). Each adsorbate/site selection requires a corresponding, separately generated MACE dataset and is fitted independently; these options are supported input configurations rather than evidence that every combination has been calculated or validated.

The reference dataset was generated from an initial Pt(111) adsorption model: a 3 × 3 × 4 slab with an fcc lattice parameter of 3.92 Å, 15.0 Å of vacuum, and an adsorbate initially placed at the selected adsorption site (a nominal initial height of 2.0 Å). The lower two Pt layers were constrained during MACE structural relaxation. The resulting trajectory—atomic positions, energies, and forces—was saved as MACE_training.extxyz. Accordingly, this file contains the MACE results obtained from the specified starting conditions, not an independent geometry source unrelated to the initial model. The final TB-fitting notebook reads these stored MACE configurations; its optional ideal-geometry preview is for visualization only. Since an adsorbate may migrate during relaxation, the realized adsorption site and orientation must be checked from the relaxed coordinates.

The input extxyz file contains multiple configurations of consistent atom ordering and composition, in-plane periodicity, and finite MACE energy and force labels. Its first configuration establishes the relative-energy reference; subsequent configurations provide relative-energy residuals, while force residuals are evaluated across the training configurations. The final TB analysis notebook reads this previously generated dataset rather than rerunning MACE relaxation or performing first-principles electronic-structure calculations.

**Figure 2.** Schematic starting configuration for the Pt(111) adsorption model used in MACE relaxation. MACE-derived trajectory is subsequently used for tight-binding fitting.

Surface configurations are considered because adsorption geometry changes the relative orientations and separations of Pt–O and Pt–H bonds, and thus the Slater–Koster angular factors and hopping strengths. O-, OH-, OOH-, and O₂-containing species provide ORR-relevant test systems; the present electronic-structure workflow does not calculate ORR volcano curves or electrochemical reaction free energies.

### 3.2 Automated Hamiltonian Construction

Hamiltonian construction is automated by mapping the atomic coordinates to an orbital-resolved basis and then assembling the onsite and interatomic matrix blocks. The final basis has six orbitals per Pt atom (five 5d and one 6s), four per O atom (one 2s and three 2p), and one 1s orbital per H atom. The total basis dimension is $N_{basis} = 6N_{Pt} + 4N_{O} + N_{H}$, with a corresponding valence-electron count of $10N_{Pt} + 6N_{O} + N_{H}$ in the notebook's spin-degenerate filling model. Pt 5d onsite energy is fixed to zero as the model energy origin; the Pt 6s, O 2s, O 2p, and H 1s onsite levels are adjustable when present.

For a Pt/O-only configuration, the initial Hamiltonian contains $5N_{Pt} + 3N_{O}$ basis functions, whereas the refined Hamiltonian contains $6N_{Pt} + 4N_{O}.$ When H is included, both models add one H 1s basis function per H atom. Both models are fitted to MACE-derived relative energies and atomic forces; the denser k-point meshes discussed later are post-fitting spectral diagnostics, not a separate change to the reference fitting objective.

The ASE neighbor-list machinery identifies species-dependent interacting pairs and the lattice translations associated with periodic neighbors. Pt–Pt nearest-neighbor cutoff and reference separation are inferred from the loaded surface geometry. The other configured cutoffs are Pt–O 3.2 Å, Pt–H 2.7 Å, O–O 2.1 Å, O–H 1.6 Å, and H–H 1.3 Å, with pair-specific reference distances. Duplicate reverse pairs are eliminated, whereas their conjugate contributions are restored during Hermitian assembly.

For each retained bond, the program computes the distance $Rᵢⱼ$ and direction cosines (l,m,n) and evaluates the relevant S–K angular tensors $C_{\mu\nu\lambda}$. Radial hopping integrals have an exponentially decaying dependence on interatomic separation, multiplied by a smooth finite-range cutoff. A quintic switching function decreases from unity to zero over the final 10% of the pair cutoff range; it and its first two derivatives are continuous at the switching endpoints.

Distance-dependent hopping is

$$V_{\lambda}(R_{ij}) = V_{\lambda,0}\exp\left( - \beta\left( R_{ij} - R_{0} \right) \right)f_{c}(R_{ij})$$

and the cutoff function

$$f_{c}(t) = 1 - 10t^{3} + 15t^{4} - 6t^{5},\quad t = \left( R - 0.9R_{c} \right)/\left( 0.1R_{c} \right)\quad\text{for}\ \quad 0.9R_{c} < R < R_{c}$$

Distance-independent basis mappings, bond-list indices, angular tensors, and spatial derivatives of the angular tensors are cached before fitting. For the latter, the notebook employs centered finite differences with a $10^{- 5}\,\mathrm{\AA}$ displacement. The Hamiltonian is assembled using the angular-channel contraction $H_{i\mu,iv} = \Sigma_{\lambda}C_{\mu v}^{\lambda}(l,m,n)V_{\lambda}(R_{ij})$, and reverse entries are added to maintain Hermiticity. For post-fit k-point calculations, periodic-image hopping blocks are multiplied by a Bloch phase $exp(2\pi i\ k \cdot T)$. Here $T$ is the integer lattice-translation index of the periodic image and $k$ is expressed in fractional reciprocal coordinates, consistent with the implementation; fitting itself remains Γ-point based.

$$H_{i\mu,j\nu}\left( \mathbf{k} \right) = \sum_{\mathbf{T}}^{}{H_{i\mu,j\nu}\left( \mathbf{T} \right)\exp\left( 2\pi ik \cdot \mathbf{T} \right)}$$

where $\mathbf{T}$ is lattice translation index and $\mathbf{k}$ is k-point represented by fractional reciprocal coordinates.

### 3.3 Parameter Optimization

The optimization variables determine the Slater–Koster hopping integrals, orbital onsite energies, and short-range pair-repulsive interactions. For each active chemical pair, each hopping channel λ has a reference amplitude $V_{\lambda,0}$ and an exponential decay coefficient β. A pair-repulsion term adds an amplitude $A_{rep}$ and decay coefficient $\alpha_{rep}$, while onsite energies $\varepsilon$ set the diagonal orbital levels. Reference distances $R_{0}$ and cutoff radii are specified by the pair settings and are not fitted here.

$$E_{rep}(R_{ij}) = A_{rep}\exp\left( - \alpha_{rep}\left( R_{ij} - R_{0} \right) \right)f_{c}(R_{ij})$$

These quantities are collected into a parameter vector $\mathbf{\theta.}$ For example, the Pt–Pt d–d portion contains $V_{dd\sigma,0}$, $V_{dd\pi,0}$, $V_{dd\delta,0}$ and the associated β values; Pt–O d–p channels contain $V_{dp\sigma,0}$, $V_{dp\pi,0}$ and their respective decay constants. The final extended orbital basis also adds the s-containing channels described in Section 2.2. A bounded parameter search is used, with the initial values optionally inherited from a compatible previously fitted TB_parameters.json file (warm start).

The TB model energy is evaluated as a finite-temperature, spin-degenerate one-electron band free energy plus the pair-repulsive contribution. Occupations are obtained from a Fermi–Dirac distribution at 300 K, with the chemical potential selected to reproduce the chosen valence-electron count. The repulsive term is a phenomenological component of this TB total-energy model; matching a MACE potential energy must not be interpreted as recovering a uniquely defined DFT Kohn–Sham Hamiltonian or a complete first-principles energy decomposition.

$$E_{TB} = \sum_n{f_{n}\epsilon_{n}} + 2k_{B}T\sum_n\left[ p_{n}\ln p_{n} + \left( 1 - p_{n} \right)\ln\left( 1 - p_{n} \right) \right] + \sum_{i < j}^{}E_{rep,ij}$$

where $p_{n}$ is spin occupation.

The total loss function consists of energy and force fitting terms, together with three additional constraints. Parameter regularization penalizes excessive deviations of inherited Tight-Binding parameters from their previously fitted values **(initial model)** inherited from a compatible reduced-basis TB parameter file, when supplied. An additional regularization term constrains the O 2s onsite energy, while a soft energy-gap constraint encourages the O 2s level to remain at least 6 eV below the O 2p level. These constraints are introduced to stabilize parameter optimization and maintain physically reasonable orbital-energy relationships rather than to fit DFT-derived electronic-structure data directly.

$$\mathcal{L}(\theta) = \frac{1}{2}\left| \left| \mathbf{r}_{\mathbf{E}}\left( \mathbf{\theta} \right) \right| \right|^{2} + \frac{1}{2}\left| \left| r_{F}(\theta) \right| \right|^{2} + \mathcal{L}_{Reg} + \mathcal{L}_{O_{2s},Reg} + \mathcal{L}_{Gap}$$

The three additional penalty terms are:

$$\mathcal{L}_{Reg} = 0.03\frac{1}{N_{inherited}}\sum_{j \in inherited}^{}(\frac{\theta_{j} - \theta_{j,0}}{s_{j}}\text{)}^{2}$$

Regularization of shared TB parameter values relative to a compatible reduced-basis fit (if a warm start is used).

$$\mathcal{L}_{O_{2s},Reg} = 0.20\left(\frac{\varepsilon_{O_{2s}} - \varepsilon_{O_{2s,0}}}{3.0}\right)^{2}$$

Regularization of the newly introduced O 2s onsite energy around its reference value.

$$\mathcal{L}_{Gap} = 10.0\left[ \text{max}\left( 0,0.60 - \left( \varepsilon_{O_{2p}} - \varepsilon_{O_{2s}} \right) \right) \right]^{2}$$

A soft constraint encouraging the O 2s onsite level to lie at least 6 eV below the O 2p onsite level.

Energy and force residual are

$$r_{E_{a}} = \sqrt{w_{E}/(M - 1)}\frac{\left( E_{TB,a} - E_{TB,0} \right) - \left( E_{MACE,a} - E_{MACE,0} \right)}{s_{E}}$$

$$r_{F,aik} = \sqrt{w_{F}/3MN}\frac{F_{TB,aik} - F_{MACE,aik}}{s_{F}}$$

respectively.

Here M denotes the number of configurations, N the atoms per configuration, and k the Cartesian component; the notebook sets $w_{E} = w_{F} = 1$. The parameter-prior weight is 0.03 for inherited parameters, with a separate weak O 2s onsite prior weight of 0.20. An O 2s–O 2p separation of at least 6 eV is encouraged using a one-sided quadratic penalty with weight 10. These onsite priors are heuristic physical constraints, not DFT-derived orbital-energy labels or measured electronic targets.

Automatic differentiation applies the chain rule to the parameterized tight-binding model: changes in hopping parameters modify the Hamiltonian and eigenvalue spectrum, which affect the energy residuals. Force-loss gradients additionally require the dependence of atomic forces on the TB parameters. The final notebook evaluates parameter gradients of the combined scalar loss using PyTorch automatic differentiation and a custom fixed-electron-number spectral operator. This avoids a full finite-difference Jacobian with respect to fitting parameters. However, spatial derivatives of the Slater–Koster angular factors were approximated by centered finite differences and cached; automatic differentiation therefore does not imply that every spatial derivative is computed analytically.

$$\frac{\partial L_{E}}{\partial V_{0}} = \ \frac{\partial L_{E}}{\partial E_{TB}}\frac{\partial E_{TB}}{\partial H}\frac{\partial H}{\partial V}\frac{\partial V}{\partial V_{0}}$$

$$\mathbf{F}_{i}^{TB} = - \nabla_{\mathbf{R}_{\mathbf{i}}}E_{TB},\quad\quad\frac{\partial L_{F}}{\partial V_{0}} = \sum_{i,k}\frac{\partial L_{F}}{\partial F_{i,k}}\left( \frac{\partial^{2}E_{TB}}{\partial V_{0}\partial R_{i,k}} \right)$$

where $L_{E}(\theta)$ and $L_{F}(\theta)$ are energy and force loss respectively.

Bound-constrained L-BFGS-B optimization is applied in two stages: an energy-only warm-up (up to 40 iterations) followed by the combined energy-and-force optimization (up to 100 iterations). In both stages the parameter-prior terms remain active. The fitted parameters, gradient information, optimizer convergence flags, and training-set energy/force RMSE are exported to reproducible files. Because these metrics describe the configurations used for fitting, they must not be presented as out-of-sample validation. The code does not compute a full residual Jacobian or establish uniqueness of the fitted Hamiltonian.

### 3.4 Electronic Structure Calculation

After optimization, the fitted Γ-point Hamiltonian is reconstructed and checked for symmetry. Its Pt-only, adsorbate-only, and Pt–adsorbate coupling subblocks are stored together with the complete matrix. The eigenvalue problem is solved using SciPy Hermitian eigensolvers; the Fermi energy is found from the fixed electron count and finite-temperature occupations. Orbital-projection weights of normalized eigenvectors provide the Γ-point PDOS, which is summed into Pt 5d, Pt 6s, O 2p, O 2s, and H 1s channels for the orbitals present in the selected configuration.

Each discrete level is broadened with a normalized Gaussian with a half-width at half maximum of 0.10 eV. Rather than imposing a fixed plotted energy window, the code spans the calculated eigenvalue range plus a 1 eV margin on each side. An optional post-fit sampling step constructs H(k) for 1×1×1, 3×3×1, 5×5×1, 9×9×1, and 12×12×1 meshes. In this step, the Fermi level and PDOS are recalculated for each mesh. Importantly, these k-sampled spectra use the fitted Γ-point parameters and do not amount to a self-consistent k-mesh total-energy/force refit.

$$D_{M} \approx \frac{1}{N_{k}}\sum_{\mathbf{k}}{\sum_n{\sum_{\mu \in M}\left| c_{n\mu}\left( \mathbf{k} \right) \right|^{2}}}g_{\sigma}\left( E - \varepsilon_{n}\left( \mathbf{k} \right) \right)$$

The notebook additionally provides Green-function and self-energy diagnostics for orbital hybridization. For each oxygen atom, its three O 2p orbitals define a projected subspace A, while all remaining orbitals form the environment B. The exact Schur-complement form of the retarded projected Green function and self-energy is used to compare the direct spectral projection with the Green-function reconstruction and to examine energy-dependent hybridization. For bare Pt, an analogous five-orbital projection is constructed on a top-layer Pt 5d subspace rather than inventing an O-derived spectrum.

$$\Sigma_A(z,\mathbf{k}) = H_{AB}\left( \mathbf{k} \right)\left[ zI - H_{BB}\left( \mathbf{k} \right) \right]^{- 1}H_{BA}\left( \mathbf{k} \right)$$

$$G_{A}\left( z,\mathbf{k} \right) = \left[ zI - H_{AA}\left( \mathbf{k} \right) - \Sigma_A(z,\mathbf{k}) \right]^{- 1}$$

The Green-function diagnostics use a finite positive imaginary energy broadening and are calculated after fitting; they do not impose additional electronic-structure fitting labels. For OH, OOH, and O₂, environment B can include other oxygen and hydrogen orbitals, so it must not automatically be interpreted as a Pt-only bath. The model is nonmagnetic and spin-degenerate, a particularly important limitation for triplet O₂. ORR-related bonding interpretation is an intended application; quantitative reaction barriers, free-energy diagrams, and catalytic activity predictions are outside the scope of this notebook.

The reproducible outputs include TB_parameters.json, fit_report.json, fitted_Hamiltonian.npy, Hamiltonian_blocks.npz, fitting comparisons, orbital PDOS files/figures, optional k-mesh PDOS comparisons, and Green-function diagnostics. Agreement with DFT-PDOS can be assessed externally, but DFT-PDOS is not incorporated into the fitting loss or used for post-fit calibration.

### 3.5 Model Refinement Strategy

To evaluate the effects of model refinement, two Slater–Koster Tight-Binding models were considered: an initial reduced-basis model and a refined expanded-basis model.

The initial model included Pt 5d, O 2p, and H 1s orbitals, whereas the refined model additionally incorporated Pt 6s and O 2s orbitals. Both models retained the same fundamental Hamiltonian construction framework and MACE energy–force-based parameter fitting strategy. **The principal methodological modification was therefore the expansion of the orbital basis and the corresponding set of hopping and onsite parameters.**

**Table 2.** Comparison of the Initial and Refined Slater-Koster Tight-Binding Model

|                   | Initial Model             | Refined Model                 |
|-------------------|---------------------------|-------------------------------|
| Pt orbital basis  | 5d (5)                    | 5d + 6s (6)                   |
| O orbital basis   | 2p (3)                    | 2s + 2p (4)                   |
| H orbital basis   | 1s (1)                    | 1s (1)                        |
| Hamiltonian       | Slater-Koster TB          | Same as initial model         |
| Fitting reference | MACE Energy/Force         | Same as initial model         |
| Optimization      | Automatic differentiation | Same approach                 |
| Parameter space   | Reduced                   | Expanded                      |
| PDOS sampling     | $\Gamma$-point (1x1x1)    | $\Gamma$-point + k-mesh sweep |
| Spin              | Non-spin-polarized        | Same as initial model         |

**The refined model employs an expanded orbital basis within the same MACE-assisted Slater–Koster fitting framework, while** **additional k-point sampling is performed for post-fit PDOS analysis.** This additional sampling was performed after parameter optimization to examine the convergence behavior of the calculated PDOS.

Because the orbital basis and k-point sampling conditions differed between the initial and refined calculations, their respective contributions to the observed spectral changes cannot be fully isolated from the present comparison.

## 4. Results and Discussion

### 4.1 Initial Model

The initial implementation adopted a reduced orthogonal orbital basis consisting of Pt 5d, O 2p, and H 1s states. Spin polarization was omitted, and the projected density of states (PDOS) was initially calculated at the Γ point (1 × 1 × 1 k-point sampling). The reduced Slater–Koster model provided a computational baseline for constructing an onsite-and-hopping Hamiltonian from MACE-constrained parameters, but its limited basis and reciprocal-space sampling restricted the subsequent spectral interpretation.

The baseline/refinement comparison **therefore distinguishes two orbital bases**, rather than two different fitting frameworks: **initial Pt 5d / O 2p versus refined Pt 5d + 6s / O 2s + 2p**, with H 1s unchanged when applicable. Slater–Koster construction, MACE energy/force supervision and the non-spin-polarized approximation are retained.

The initial Pt-derived spectrum contained several narrow features, while the O 2p contribution was concentrated in a comparatively sharp peak near the lower-energy region of the occupied states. These results established that the Hamiltonian could be diagonalized and orbital projections extracted; they did not, by themselves, demonstrate agreement with a first-principles electronic structure.

**Figure 3.** Initial $\Gamma$-point PDOS from the reduced orbital-basis model for bridge-site O adsorption; orbital-resolved contributions (left) and elemental contributions (right).

### 4.2 Comparison with DFT-PDOS

The initial PDOS was first compared qualitatively with the surface electronic-structure trends discussed by Shuttleworth (2018) for oxygenated and hydroxylated Pt(111) [10]. Spectral features in the vicinity of approximately −5 eV (relative to each plot’s own Fermi energy) were identified as a broad point of resemblance. However, these literature calculations used different structural and modeling conditions, and similar peak positions alone cannot establish quantitative validation.

**Figure 4.** Representative Pt(111) electronic-structure spectra reproduced in the project material from Shuttleworth (2018); shown for qualitative context, not a numerical benchmark.

A separate DFT calculation was subsequently performed by 정윤석 for O adsorbed on a 3 × 3 × 4 Pt(111) slab, starting from a bridge-site configuration and same initial calculation condition. During structural relaxation, the O atom moved toward an fcc hollow site. The DFT-PDOS plot places a prominent O 2p feature around −6.9 eV relative to its Fermi level. This relaxation is relevant to the interpretation of the comparison: the relaxed DFT spectrum should not be treated as a reference spectrum for a final bridge geometry.

**Figure 5.** DFT orbital-resolved PDOS for O/Pt(111) by 정윤석, initialized at a bridge site and subsequently relaxed toward an fcc hollow geometry. Spin-resolved curves are plotted above and below the horizontal axis.

The calculation notes report a DFT final total energy of −236.19802881 eV and a reported O–slab separation of 1.1795 Å. Separately, the project calculations list relaxed energies of −212.729402 eV (bridge) and −213.089127 eV (fcc hollow), with reported vertical adsorption heights of 1.4696 and 1.2996 Å, respectively. The latter pair is consistent with a lower reported energy for fcc than bridge within that calculation set (approximately 0.360 eV). The absolute DFT and MACE/TB energies, however, have different energy definitions and reference offsets; comparing them as a percentage error would be invalid. Adsorption-height definitions must also be matched before a direct geometrical comparison.

The clearest discrepancy at this stage was spectral shape: the initial model produced considerably sharper O-related peaks than the available DFT spectrum. This observation motivated two changes: expansion of the orbital basis by including Pt 6s and O 2s states, and systematic k-point sampling from 1x1×1 to 12×12×1 while keeping the Gaussian HWHM fixed. The subsequent comparisons examine these two changes without claiming that either alone has been proven to explain the discrepancy.

### 4.3 Orbital Basis Expansion and k-point Sampling

The refined model retains the original Slater–Koster Hamiltonian construction and MACE energy–force fitting scheme, while extending the orbital basis to Pt 5d + 6s, O 2s + 2p, and H 1s when relevant. Because the independent DFT relaxation moved the adsorbed oxygen away from the initial bridge position toward an fcc hollow site, bridge and fcc hollow configurations are evaluated separately. The k-point series presented below is calculated from parameters optimized at Γ and is used to examine the resulting PDOS.

For clarity**, the orbital-basis expansion is the defining modification to the TB model itself**. The change from a Γ-point PDOS to denser k-point PDOS concerns the subsequent evaluation of that model. Because the additional orbital parameters are refitted, the comparison does not hold the numerical hopping and onsite parameter values fixed; it holds the underlying construction and fitting methodology fixed.

#### 4.3.1 O Adsorption at a Bridge Site

For bridge-site O adsorption, increasing the mesh density from Γ only to 3 × 3 × 1 and approximately 5 × 5 × 1 made **the Pt-derived DOS** **less sensitive to discrete eigenvalue sampling**. In the displayed data, further increases up to 12 × 12 × 1 produced comparatively smaller visual changes in the broad Pt spectral envelope. In contrast, a narrow O 2p peak remained prominent at essentially the same Gaussian HWHM, demonstrating that the oxygen feature cannot be removed simply by increasing k-point sampling in this fitted model.

**Figure 6.** k-point mesh convergence check for bridge-site adsorption, comparing Pt and O orbital-group PDOS at fixed Gaussian broadening

**Figure 7.** Bridge-site orbital-group PDOS at a 9x9x1 mesh. O 2p contribution remains narrow relative to the extended Pt-derived states.

The orbital-resolved view of the prominent oxygen feature (around −4.7 eV in the plotted model spectrum) shows a large O 2p weight and much smaller Pt d weight at the same energy. This is consistent with a relatively localized adsorbate-dominated state and comparatively limited mixing into the Pt states represented by the model. It should not be interpreted as a direct measure of adsorption strength or catalytic activity.

**Figure 8.** Bridge-site PDOS orbital decomposition and orbital-weight analysis near the narrow oxygen-dominated feature.

To check the algebraic consistency of the projected spectrum, the same fitted Hamiltonian was also analyzed with a Green-function construction based on a Schur complement. The direct eigenvector projection and the corresponding Green-function spectral function agree when evaluated with a consistent Lorentzian broadening. The minimum in the plotted inverse-Green-function residual occurs near the selected spectral feature. Such agreement is an internal numerical consistency check for the Hamiltonian and projection procedure; it is not an independent confirmation of DFT-level electronic-structure accuracy.

**Figure 9.** Bridge-site oxygen 2p Green-function consistency check (left) and inverse-Green-function resonance diagnostic at the dominant k-point (0.0, 0.4, 0.0), in fractional reciprocal coordinates, on a 5x5x1 mesh (right).

#### 4.3.2 O Adsorption at an FCC Hollow Site

The fcc hollow calculation shows a different distribution of oxygen and platinum spectral weight. As in the bridge case, the displayed Pt-derived DOS becomes substantially more stable by approximately the 5x5x1 sampling level. The oxygen spectrum, however, is distributed across a broader low-energy range and overlaps more visibly with Pt-related features than in the bridge case. This difference is physically plausible because the bond orientations and local coordination differ between the sites.

**Figure 10.** k-point mesh comparison for O adsorption at an fcc hollow site, with fixed broadening HWHM = 0.10 eV.

**Figure 11.** Orbital-group PDOS for the fcc hollow case at 9x9x1, highlighting an O-related spectral distribution broader than the bridge-site feature.

For the fcc hollow configuration, the calculated PDOS exhibits qualitative similarities to the independently obtained DFT result in its overall spectral shape and distribution of prominent oxygen-related features. This observation suggests that the refined, physically constructed TB Hamiltonian can capture selected aspects of the electronic structure. The broader O-related spectral distribution is also consistent with a different orbital-hybridization pattern in the chosen TB model, although PDOS alone cannot quantify bond strength. Because the DFT reference relaxed from the initial bridge position toward an fcc hollow geometry, this comparison is more appropriate than treating the DFT result as a final bridge-site spectrum. Nevertheless, differences in relaxed geometries, energy alignment, orbital representations, and broadening prevent a claim of quantitative agreement from visual resemblance alone.

**Figure 12**. FCC hollow-site orbital decomposition and representative orbital contributions for an oxygen-related spectral window.

**Figure 13.** FCC hollow-site Green-function projection consistency (left) and inverse-Green-function resonance diagnostic (right).

### 4.4 Effects of Model Refinement

Comparisons between the saved reduced-basis result and the updated calculation show clear changes in the Pt-derived spectral envelope and in the location and width of adsorbate-related peaks. In particular, the expanded-basis and k-sampled calculation reduces the artificial appearance of multiple discrete Pt peaks. However, a sharp oxygen feature remains in the bridge case, while the fcc hollow case exhibits a more distributed O 2p contribution.

The initial/refined distinction is Pt 5d / O 2p versus Pt 5d + 6s / O 2s + 2p (H 1s unchanged when included). The displayed before-and-after plots additionally differ in k-point sampling, so the improvement cannot be attributed quantitatively to basis expansion alone without a same-mesh control.

**Figure 14.** Comparison of the saved reduced model and updated model at the bridge site: Pt 5d (left) and O 2p (right).

**Figure 15.** Comparison of the saved reduced model and updated model at the fcc hollow site: Pt 5d (left) and O 2p (right).

These plots should be interpreted as an overall before-and-after comparison rather than a controlled demonstration that a single modification caused the improvement: the orbital basis, fitted parameter space, and reciprocal-space sampling changed together. The apparent convergence near 5 × 5 × 1 is based on the displayed PDOS profiles, not a stated quantitative convergence tolerance or a systematic error metric against matched DFT spectra.

Several limitations remain. MACE-derived energies and forces do not uniquely determine an electronic Hamiltonian, and the fitted TB free-energy expression uses phenomenological pair repulsion rather than a full first-principles energy functional. The Γ-only fitting and post-fit k-sampled PDOS are not fully self-consistent. Differences in spin treatment, reference geometry, Fermi-level alignment, and spectral broadening complicate comparisons with DFT; the non-spin-polarized approximation is especially restrictive for O₂.

## 5. Conclusion

This project developed an automated workflow that connects MACE-labeled atomic energies and forces to a physically constructed Slater–Koster tight-binding Hamiltonian for Pt(111) adsorption systems. Without using DFT-derived Hamiltonians, eigenvalues, or PDOS as direct fitting labels, onsite, hopping, and repulsive parameters were optimized against the MLIP reference data. Diagonalization and orbital projection then enabled PDOS analysis, with Green-function and self-energy calculations available as additional internal diagnostics.

The initial reduced-basis, Γ-point model exhibited oxygen-related spectral peaks that were substantially sharper than the available DFT reference. After extending the basis to include Pt 6s and O 2s and examining successive k-point meshes, the Pt-derived DOS displayed a smoother, more stable spectral envelope, although a narrow O 2p peak persisted in the bridge configuration. The fcc hollow calculation produced a broader oxygen-related distribution with qualitative similarities to the DFT-PDOS in overall spectral shape and characteristic features. These observations support the usefulness of the refined workflow for exploratory electronic-structure analysis but do not isolate the individual contributions of orbital expansion and k-point sampling.

**The principal methodological comparison** is between an initial Pt 5d / O 2p orbital basis and an expanded Pt 5d + 6s / O 2s + 2p basis. Both retain the same physical Slater–Koster construction and MACE-constrained fitting strategy; changes in displayed k-point sampling are treated separately as post-fitting diagnostics.

The present results should be regarded as an exploratory demonstration of automated, physics-based Hamiltonian construction, not as a validated substitute for first-principles electronic-structure calculations. In particular, agreement in MACE energy–force fitting does not guarantee the correct orbital-resolved spectrum. The reported Green-function agreement verifies internal consistency of the chosen Hamiltonian rather than its external physical accuracy.

Future work should prioritize more systematic, quantitative comparisons with DFT calculations to evaluate the electronic-structure accuracy and physical validity of the constructed tight-binding Hamiltonian. One possible extension is to optimize the TB parameters against Brillouin-zone-integrated energies and forces, rather than using the Γ-point fitting approximation, and then assess whether this change improves agreement with DFT-derived spectra. Such comparisons would help clarify the limitations of the current model and guide further refinement.

## 6. References


[1] Qian, C., Vitartas, V., Kermode, J. R., & Maurer, R. J. (2026). Equivariant electronic Hamiltonian prediction with many-body message passing. *npj Computational Materials*, *12*(1), 169.

[2] Choudhary, K. (2025). Slakonet: A unified slater-koster tight-binding framework using neural network infrastructure for the periodic table. *The Journal of Physical Chemistry Letters*, *16*(43), 11109-11119.

[3] Fried, H. P., Barragan-Yani, D., Libisch, F., & Wirtz, L. (2025). A machine learning approach to predict tight-binding parameters for point defects via the projected density of states. *npj Computational Materials*, *11*(1), 176.

[4] Schwade, M., Zhang, S., Vonhoff, F., Delgado, F. P., & Egger, D. A. (2026). Physics-informed Hamiltonian learning for large-scale optoelectronic property prediction. *Nature Communications*, *17*(1), 2652.

[5] Zhu, X., Rinke, P., & Egger, D. A. (2026). Predicting temperature-dependent optoelectronic properties of semiconductor defects with equivariant neural networks. *npj Computational Materials*, *12*(1), 176.

[6] Xia, Y., Thie, A., Soon, J., & Barca, G. M. (2026). NN-xTB: density functional accuracy at semi empirical speed with neural network extended tight binding. *Nature Communications*.

[7] Batatia, I., Kovacs, D. P., Simm, G., Ortner, C., & Csányi, G. (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *Advances in neural information processing systems*, *35*, 11423-11436.

[8] Koskinen, P., & Mäkinen, V. (2009). Density-functional tight-binding for beginners. *Computational Materials Science*, *47*(1), 237-253.

[9] Synopsys QuantumATK, “ProjectedDensityOfStates,” documentation. https://docs.quantumatk.com/manual/Types/ProjectedDensityOfStates/ProjectedDensityOfStates.html

10. Shuttleworth, I. G. (2018). Binding site transitions across strained oxygenated and hydroxylated Pt (111). *ChemistryOpen*, *7*(5), 356-369.

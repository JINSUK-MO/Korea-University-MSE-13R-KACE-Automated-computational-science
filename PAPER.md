# Physics-based and MACE-assisted Hamiltonian Construction

Korea University AA Co-E Creator Club (Sep. – Dec. 2026)  
Department of Materials Science and Engineering

**Team Members:** [@Seoyeong Ahn](https://github.com/seo00224), 연제우, [@Jiyeon Lee](https://github.com/jiyeonie322), [@Yunseok Jung](https://github.com/Sukee03), [@Junyoung Choi](https://github.com/chlwnsdud27-cloud), and [@Jinsuk Mo](https://github.com/JINSUK-MO)


## 1. Introduction

### 1.1 Project Motivation

As members of an automation club, we sought to explore how machine learning, a key technology in modern automation, could be extended **beyond conventional automation tasks to computational materials science.**

In particular, we were interested in whether machine-learning interatomic potentials (MLIPs), such as MACE, could facilitate the automated construction of physically motivated electronic Hamiltonians without directly relying on DFT-derived electronic-structure data for parameter fitting.

Rather than aiming for a complete reproduction of DFT-level electronic structures, our primary objective was to develop a computational workflow that integrates MACE-derived energies and forces with Slater–Koster tight-binding theory. Through this approach, we sought to construct parameterized Hamiltonians and extract electronic properties, including projected densities of states (PDOS), with minimal manual intervention.

Ultimately, this project represents an exploratory attempt to connect machine-learning-assisted automation with physics-based electronic structure modeling, while investigating the capabilities and limitations of such an approach.

### 1.2 Background and Related Work

Recent advances in machine learning have enabled efficient prediction of electronic structures through data-driven Hamiltonian modeling. MACE-H (2026) employs equivariant many-body message passing to predict DFT Kohn–Sham Hamiltonian matrices directly from atomic configurations [[1]](#ref1). Similarly, SlaKoNet (2025) optimizes physically interpretable Slater–Koster tight-binding parameters using electronic-structure data [[2]](#ref2), while Fried et al. (2025) demonstrated the inference of tight-binding parameters from orbital-projected densities of states [[3]](#ref3).

Other studies have explored the integration of machine-learned interatomic potentials with electronic-structure models. HAMSTER (2026) combines physically motivated tight-binding Hamiltonians with machine-learned corrections and employs MACE force fields for atomistic simulations [[4]](#ref4). A related framework couples MACE with DeepH-E3 to investigate temperature-dependent electronic properties [[5]](#ref5). More recently, NN-xTB (2026) has incorporated a MACE-based encoder into an adaptive semi-empirical Hamiltonian, using electronic-structure calculations and energy–force training targets [[6]](#ref6).

Collectively, these studies demonstrate the potential of combining machine learning with physically structured electronic Hamiltonians.

### 1.3 Objectives and Scope

This project explores **an alternative route to electronic-structure reconstruction** by using energies and forces predicted by a pretrained MACE interatomic potential as reference quantities for parameterizing a Slater–Koster tight-binding Hamiltonian.

Unlike direct Hamiltonian-learning approaches, the present implementation does not predict Hamiltonian matrix elements through an additional neural network. **Instead, a physically parameterized Hamiltonian is constructed from orbital onsite energies and geometry-dependent hopping integrals.** Its parameters are optimized through automatic differentiation against MACE-derived energetic and force constraints, with additional parameter regularization.

The resulting Hamiltonian is diagonalized to obtain electronic eigenstates and orbital-resolved projected densities of states (PDOS). Pt(111) surface adsorption is employed as a model system to investigate the electronic interactions between metal and adsorbate orbitals.

A central objective is to examine whether agreement in atomic energies and forces is sufficient to reproduce physically meaningful electronic structures. Because energy–force information does not uniquely determine the underlying electronic Hamiltonian, comparisons with independently calculated DFT-PDOS are used to assess the limitations of the reconstructed model.

Rather than proposing a universally transferable electronic-structure predictor, this project investigates the opportunities and limitations of energy–force-constrained tight-binding parameterization, with particular emphasis on interpretability, orbital hybridization, and the consistency between atomic energetics and electronic structure.

<p align="center">
<img width="30%" alt="abstract" src="https://github.com/user-attachments/assets/e873157b-ce44-4d2c-ad2f-cfd3ef1dd78c" />

**Figure 1.** Schematic illustration of the proposed computational workflow. MACE-derived atomic energies and forces are employed as reference quantities for optimizing Slater–Koster tight-binding parameters. The resulting Hamiltonian is diagonalized to obtain orbital-resolved electronic structures, which are subsequently used to investigate metal–adsorbate interactions relevant to oxygen reduction reactions (ORR).

## 2. Theoretical Background

### 2.1 MACE Interatomic Potential

Machine learning interatomic potentials (MLIPs) approximate the potential-energy surface of an atomic system from its chemical composition and geometry. MACE (Message Passing Atomic Cluster Expansion) combines equivariant message passing with many-body atomic cluster features [[7]](#ref7). Local environments are represented using the positions and chemical species of neighboring atoms, while the architecture is designed to respect the symmetries of three-dimensional space. Consequently, the predicted scalar energy is invariant under global translations and rotations, and the derived forces transform consistently with the atomic coordinates.

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

**Table 1.** Slater-Koster angular-coefficient table used to construct the orbital hopping blocks. Here, x, y, and z are the direction cosines of $\widehat{\mathbf{R}}$ with $x^{2} + y^{2} = 1$, and the notation $\alpha = x^{2} + y^{2}$ and $\beta = x^{2} - y^{2}$. Reproduced from Koskinen (2009) [[8]](#ref8).

<p align="center">
<img width="60%" alt="Slater_Koster_Table" src="https://github.com/user-attachments/assets/d6c0d876-5cf0-4814-9366-97d474534212" />

> **Table 1 note:** The original table is embedded as an image in the Word source and is not transcribed here. Refer to Koskinen and Mäkinen (2009) [[8]](#ref8) for the full angular-coefficient table.

The original reduced orbital model considered Pt 5d, O 2p, and H 1s states. The final notebook extends this basis to Pt (5d + 6s), O (2s + 2p), and H (1s). This extension introduces new hopping channels, including Pt s–s/s–d, Pt–O d–s/s–p/s–s, and O–O s–p/s–s contributions. Each chemical pair retains a separate repulsive term in addition to its electronic hopping channels. The basis remains orthogonal and non-spin-polarized; thus, the overlap matrix is treated as the identity and explicit spin splitting is not included.

The Hamiltonian in this project is physically constructed from tabulated angular factors, neighbor geometry, onsite levels, and fitted radial parameters. This is distinct from directly predicting every Hamiltonian matrix element with an additional graph neural network. It also means that the resulting orbital interpretation is tied to the chosen basis set and the approximations of the S–K parametrization.

### 2.3 Hamiltonian Diagonalization and PDOS

Once the tight-binding Hamiltonian is assembled, the electronic eigenvalues and eigenvectors are determined by diagonalization. In the orthogonal basis adopted here, the stationary eigenvalue problem takes the usual form $H(k)cₙ(k)\  = \ \varepsilon ₙ(k)cₙ(k)$. The eigenvalues describe the model's allowed one-electron energies, while the squared orbital coefficients quantify each basis orbital's contribution to an eigenstate. (For a nonorthogonal basis, the more general equation would contain an overlap matrix $S(k)$.)

$$H(k)c_{n}(k) = \varepsilon_{n}(k)c_{n}(k);\quad\quad H(k)^{\dagger} = H(k)$$

The density of states counts electronic eigenstates in an energy interval. Orbital-projected DOS is obtained from a projection operator $\widehat{P}_M$ onto a selected atomic orbital or group $M$. In an orthonormal basis,

$$
\widehat{P}_M = \sum_{\mu \in M} |\mu\rangle\langle\mu|
$$

Its expectation value on an eigenstate becomes the sum of $|c_{n\mu}(\mathbf{k})|^2$ over the selected orbitals, consistent with the projector-based definition used in QuantumATK documentation [[9]](#ref9).

$$D_{M}(E) = \left( 1/N_k \right)\sum_{\mathbf{k}}\sum_n\left\langle \psi_{n\mathbf{k}}\left| {\widehat{P}}_{M} \right|\psi_{n\mathbf{k}} \right\rangle\delta(E - \varepsilon_{n\mathbf{k}})$$

In these expressions, E denotes the energy variable at which the spectrum is evaluated, whereas $\varepsilon ₙ(k)$ denotes an eigenvalue of the Hamiltonian; thus $\delta\left( E - \varepsilon ₙ(k) \right)$ contributes at each eigenenergy. For numerical plots, the Dirac delta functions are replaced by normalized Gaussian kernels. Orbital PDOS curves are aggregated into Pt 5d, Pt 6s, O 2p, O 2s, and H 1s contributions. The internally determined Fermi level $E_{F}$ sets the plotted energy origin, $E - E_{F}$. These spectra are outputs of the fitted Hamiltonian and are not targets of the energy–force fitting.

For metal–oxygen adsorption, PDOS and orbital hybridization can reveal qualitative changes in Pt–adsorbate coupling. The O\*, OH\*, OOH\*, and O₂-containing surface configurations were selected as model systems relevant to oxygen reduction reactions (ORR).

## 3. Computational Methodology

### 3.1 Pt(111) Surface and Adsorption Models

The model system is a Pt(111) surface slab with oxygen-containing adsorbates relevant to the oxygen reduction reaction (ORR). The notebook permits bare Pt(111) as well as O, O₂, OH, and OOH placed at ontop, bridge, fcc hollow, or hcp hollow sites (17 nominal selections including the bare slab). Each adsorbate/site selection requires a corresponding, separately generated MACE dataset and is fitted independently; these options are supported input configurations rather than evidence that every combination has been calculated or validated.

The reference dataset was generated from an initial Pt(111) adsorption model: a 3 × 3 × 4 slab with an fcc lattice parameter of 3.92 Å, 15.0 Å of vacuum, and an adsorbate initially placed at the selected adsorption site (a nominal initial height of 2.0 Å). The lower two Pt layers were constrained during MACE structural relaxation. The resulting trajectory—atomic positions, energies, and forces—was saved as MACE_training.extxyz. Accordingly, this file contains the MACE results obtained from the specified starting conditions, not an independent geometry source unrelated to the initial model. The final TB-fitting notebook reads these stored MACE configurations; its optional ideal-geometry preview is for visualization only. Since an adsorbate may migrate during relaxation, the realized adsorption site and orientation must be checked from the relaxed coordinates.

The input extxyz file contains multiple configurations of consistent atom ordering and composition, in-plane periodicity, and finite MACE energy and force labels. Its first configuration establishes the relative-energy reference; subsequent configurations provide relative-energy residuals, while force residuals are evaluated across the training configurations. The final TB analysis notebook reads this previously generated dataset rather than rerunning MACE relaxation or performing first-principles electronic-structure calculations.

<p align="center">
<img width="30% alt="schema" src="https://github.com/user-attachments/assets/7b98ae3f-9a58-42b9-af0b-7e4c70c790a2" />

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

### 3.6 Quantitative Spectral Analysis

To quantitatively evaluate changes in the electronic structure following model refinement, several spectral descriptors were introduced for the comparison of projected densities of states (PDOS). This analysis was motivated by the discrepancies observed between the calculated TB-PDOS and the available DFT reference, particularly in the positions and widths of characteristic O 2p peaks for O adsorption on Pt(111).

Although visual comparisons provide qualitative information about spectral similarities, they are insufficient for systematically distinguishing peak shifts, spectral broadening, and redistribution of orbital spectral weight. Accordingly, five complementary descriptors were employed: Wasserstein-1 distance, main peak position, spectral centroid, full width at half maximum (FWHM), and spectral standard deviation.

#### 3.6.1 Wasserstein-1 Distance

The Wasserstein-1 distance was introduced to quantify differences between the overall energy distributions of two PDOS spectra. It is defined as

$$
W_1(P,Q)=\int_{-\infty}^{\infty}|F_P(E)-F_Q(E)|,dE
$$

where (F_P(E)) and (F_Q(E)) denote the cumulative distribution functions of the normalized initial and refined PDOS, respectively.

Unlike direct peak-height comparisons, the Wasserstein distance characterizes the average energy displacement associated with redistribution of spectral weight. It therefore captures differences arising from peak shifts, broadening, and changes in the distribution of electronic states.

The PDOS curves were interpolated onto a common energy grid and normalized to unit integrated spectral weight before evaluation. The full spectral ranges of the two models were included in the comparison.

#### 3.6.2 Main Peak Position

To quantify shifts in the most prominent spectral feature, the main peak position was defined as

$$
E_{\mathrm{peak}}=\arg\max_E D(E)
$$

The corresponding difference between the initial and refined models was calculated as

$$
\Delta E_{\mathrm{peak}}
=E_{\mathrm{peak}}^{\mathrm{Refined}}
-E_{\mathrm{peak}}^{\mathrm{Initial}}
$$

This descriptor directly measures changes in the position of the highest-intensity peak. However, in spectra containing multiple peaks, changes in relative peak intensities may cause the identity of the dominant peak to change between models.

#### 3.6.3 Spectral Centroid

The spectral centroid represents the energy-weighted average position of the PDOS and is defined as

$$
\mu=\frac{\int E D(E),dE}{\int D(E),dE}
$$

Unlike the main peak position, the centroid incorporates contributions from the entire spectral distribution. It was therefore introduced to distinguish changes in the overall energy distribution from shifts in an individual peak.

The centroid shift was defined as

$$
\Delta\mu=\mu_{\mathrm{Refined}}-\mu_{\mathrm{Initial}}
$$

#### 3.6.4 Full Width at Half Maximum (FWHM)

The FWHM was employed to characterize the width of the dominant spectral peak. It was determined from the energy interval between the two half-maximum crossings surrounding the highest-intensity peak.

This descriptor is particularly relevant to the present project because the initial TB model exhibited excessively sharp oxygen-related spectral features.

However, FWHM characterizes a local peak rather than the overall spectral distribution. Therefore, it was evaluated together with the spectral standard deviation.

#### 3.6.5 Spectral Standard Deviation

The spectral standard deviation describes the overall energy dispersion of the PDOS:

$$
\sigma_E=
\sqrt{
\frac{\int(E-\mu)^2D(E),dE}
{\int D(E),dE}
}
$$

Unlike FWHM, this descriptor accounts for additional peaks and spectral tails. Consequently, it can identify redistribution or broadening of spectral weight even when the dominant peak width remains nearly unchanged.

#### 3.6.6 Comparison Conditions

The descriptors were evaluated for 16 configurations consisting of four adsorbates (O, O₂, OH, and OOH) and four initial adsorption-site labels (ontop, bridge, fcc hollow, and hcp hollow).

The initial model employed Pt 5d, O 2p, and H 1s orbitals with Γ-point PDOS sampling. The refined model incorporated Pt 5d + 6s, O 2s + 2p, and H 1s orbitals, with the reported PDOS evaluated using a 9×9×1 k-point mesh.

Both models retained the same fundamental Slater–Koster Hamiltonian construction and MACE Energy/Force fitting framework, although the refined model involved an expanded parameter space and subsequent parameter optimization.

For both models, energies were expressed relative to their respective Fermi levels. The Wasserstein distances were calculated from normalized spectral distributions, whereas the other descriptors were evaluated directly from the corresponding PDOS curves.

All signed descriptor changes were defined as Refined − Initial. The Wasserstein distance, by contrast, is nonnegative and represents the magnitude of the spectral distribution difference.

Importantly, the present analysis compares two TB models rather than calculating discrepancies relative to DFT. Furthermore, because the orbital basis, fitted parameters, and PDOS sampling conditions differ between the models, their individual contributions cannot be isolated from this comparison alone.

To examine the independent numerical effect of reciprocal-space sampling, additional k-point sweeps were considered using the same refined Hamiltonian and its fixed fitted parameters.

## 4. Results and Discussion

### 4.1 Initial Model

The initial implementation adopted a reduced orthogonal orbital basis consisting of Pt 5d, O 2p, and H 1s states. Spin polarization was omitted, and the projected density of states (PDOS) was initially calculated at the Γ point (1 × 1 × 1 k-point sampling). The reduced Slater–Koster model provided a computational baseline for constructing an onsite-and-hopping Hamiltonian from MACE-constrained parameters, but its limited basis and reciprocal-space sampling restricted the subsequent spectral interpretation.

The baseline/refinement comparison **therefore distinguishes two orbital bases**, rather than two different fitting frameworks: **initial Pt 5d / O 2p versus refined Pt 5d + 6s / O 2s + 2p**, with H 1s unchanged when applicable. Slater–Koster construction, MACE energy/force supervision and the non-spin-polarized approximation are retained.

The initial Pt-derived spectrum contained several narrow features, while the O 2p contribution was concentrated in a comparatively sharp peak near the lower-energy region of the occupied states. These results established that the Hamiltonian could be diagonalized and orbital projections extracted; they did not, by themselves, demonstrate agreement with a first-principles electronic structure.

<p align="center">
<img width="2416" height="826" alt="O_bridge_pdos_initial" src="https://github.com/user-attachments/assets/cea9b544-9a20-49e5-afd9-6ed20ce3af29" />

**Figure 3.** Initial $\Gamma$-point PDOS from the reduced orbital-basis model for bridge-site O adsorption; orbital-resolved contributions (left) and elemental contributions (right).

### 4.2 Comparison with DFT-PDOS

The initial PDOS was first compared qualitatively with the surface electronic-structure trends discussed by Shuttleworth (2018) for oxygenated and hydroxylated Pt(111) [[10]](#ref10). Spectral features in the vicinity of approximately −5 eV (relative to each plot’s own Fermi energy) were identified as a broad point of resemblance. However, these literature calculations used different structural and modeling conditions, and similar peak positions alone cannot establish quantitative validation.

<p align="center">
<img width="456" height="423" alt="Shuttleworth_O_bridge" src="https://github.com/user-attachments/assets/c8349a0d-a2c3-4efd-9e4e-42d7bce879f9" />

**Figure 4.** Representative Pt(111) electronic-structure spectra reproduced in the project material from Shuttleworth (2018) [[10]](#ref10); shown for qualitative context, not a numerical benchmark.

A separate DFT calculation was subsequently performed by [@Yunseok Jung](https://github.com/Sukee03) for O adsorbed on a 3 × 3 × 4 Pt(111) slab, starting from a bridge-site configuration and same initial calculation condition. During structural relaxation, the O atom moved toward an fcc hollow site. The DFT-PDOS plot places a prominent O 2p feature around −6.9 eV relative to its Fermi level. This relaxation is relevant to the interpretation of the comparison: the relaxed DFT spectrum should not be treated as a reference spectrum for a final bridge geometry.

<p align="center">
<img width="652" height="407" alt="DFTpdos" src="https://github.com/user-attachments/assets/16a4dd74-9c7e-4e27-abec-1b187edad496" />

**Figure 5.** DFT orbital-resolved PDOS for O/Pt(111) by [@Yunseok Jung](https://github.com/Sukee03), initialized at a bridge site and subsequently relaxed toward an fcc hollow geometry. Spin-resolved curves are plotted above and below the horizontal axis.

https://github.com/user-attachments/assets/c33340f8-4785-4cc7-949b-06297ebdefa7

**Supplementary Video 1.** Structural relaxation of oxygen adsorbed on Pt(111), initially positioned at a bridge site and subsequently moving toward an fcc hollow site.

The calculation notes report a DFT final total energy of −236.19802881 eV and a reported O–slab separation of 1.1795 Å. Separately, the project calculations list relaxed energies of −212.729402 eV (bridge) and −213.089127 eV (fcc hollow), with reported vertical adsorption heights of 1.4696 and 1.2996 Å, respectively. The latter pair is consistent with a lower reported energy for fcc than bridge within that calculation set (approximately 0.360 eV). The absolute DFT and MACE/TB energies, however, have different energy definitions and reference offsets; comparing them as a percentage error would be invalid. Adsorption-height definitions must also be matched before a direct geometrical comparison.

The clearest discrepancy at this stage was spectral shape: the initial model produced considerably sharper O-related peaks than the available DFT spectrum. This observation motivated two changes: expansion of the orbital basis by including Pt 6s and O 2s states, and systematic k-point sampling from 1x1×1 to 12×12×1 while keeping the Gaussian HWHM fixed. The subsequent comparisons examine these two changes without claiming that either alone has been proven to explain the discrepancy.

### 4.3 Orbital Basis Expansion and k-point Sampling

The refined model retains the original Slater–Koster Hamiltonian construction and MACE energy–force fitting scheme, while extending the orbital basis to Pt 5d + 6s, O 2s + 2p, and H 1s when relevant. Because the independent DFT relaxation moved the adsorbed oxygen away from the initial bridge position toward an fcc hollow site, bridge and fcc hollow configurations are evaluated separately. The k-point series presented below is calculated from parameters optimized at Γ and is used to examine the resulting PDOS.

For clarity**, the orbital-basis expansion is the defining modification to the TB model itself**. The change from a Γ-point PDOS to denser k-point PDOS concerns the subsequent evaluation of that model. Because the additional orbital parameters are refitted, the comparison does not hold the numerical hopping and onsite parameter values fixed; it holds the underlying construction and fitting methodology fixed.

#### 4.3.1 O Adsorption at a Bridge Site

For bridge-site O adsorption, increasing the mesh density from Γ only to 3 × 3 × 1 and approximately 5 × 5 × 1 made **the Pt-derived DOS** **less sensitive to discrete eigenvalue sampling**. In the displayed data, further increases up to 12 × 12 × 1 produced comparatively smaller visual changes in the broad Pt spectral envelope. In contrast, a narrow O 2p peak remained prominent at essentially the same Gaussian HWHM, demonstrating that the oxygen feature cannot be removed simply by increasing k-point sampling in this fitted model.

<p align="center">
<img width="3600" height="860" alt="O_bridge_k_point_sweep" src="https://github.com/user-attachments/assets/d4d3fc7d-5f13-42fa-8280-f0f596f4fd7d" />

**Figure 6.** k-point mesh convergence check for bridge-site adsorption, comparing Pt and O orbital-group PDOS at fixed Gaussian broadening

<p align="center">
<img width="2967" height="1609" alt="O_bridge_9x9x1_pdos" src="https://github.com/user-attachments/assets/a109b92c-6ab9-45da-9a2e-7c7b8863b38f" />

**Figure 7.** Bridge-site orbital-group PDOS at a 9x9x1 mesh. O 2p contribution remains narrow relative to the extended Pt-derived states.

The orbital-resolved view of the prominent oxygen feature (around −4.7 eV in the plotted model spectrum) shows a large O 2p weight and much smaller Pt d weight at the same energy. This is consistent with a relatively localized adsorbate-dominated state and comparatively limited mixing into the Pt states represented by the model. It should not be interpreted as a direct measure of adsorption strength or catalytic activity.

<p align="center">
<img width="85%" alt="O_bridge_2p_peak_eigenvector_analysis" src="https://github.com/user-attachments/assets/71045455-a7d2-4bc7-90de-7db74b029310" />

**Figure 8.** Bridge-site PDOS orbital decomposition and orbital-weight analysis near the narrow oxygen-dominated feature.

To check the algebraic consistency of the projected spectrum, the same fitted Hamiltonian was also analyzed with a Green-function construction based on a Schur complement. The direct eigenvector projection and the corresponding Green-function spectral function agree when evaluated with a consistent Lorentzian broadening. The minimum in the plotted inverse-Green-function residual occurs near the selected spectral feature. Such agreement is an internal numerical consistency check for the Hamiltonian and projection procedure; it is not an independent confirmation of DFT-level electronic-structure accuracy.

<p align="center">
<img width="85%" alt="O_bridge_Green_resonance_O_atom36" src="https://github.com/user-attachments/assets/c73ba855-9331-4d63-bebd-a6959586225c" />

**Figure 9.** Bridge-site oxygen 2p Green-function consistency check (left) and inverse-Green-function resonance diagnostic at the dominant k-point (0.0, 0.4, 0.0), in fractional reciprocal coordinates, on a 5x5x1 mesh (right).

#### 4.3.2 O Adsorption at an FCC Hollow Site

The fcc hollow calculation shows a different distribution of oxygen and platinum spectral weight. As in the bridge case, the displayed Pt-derived DOS becomes substantially more stable by approximately the 5x5x1 sampling level. The oxygen spectrum, however, is distributed across a broader low-energy range and overlaps more visibly with Pt-related features than in the bridge case. This difference is physically plausible because the bond orientations and local coordination differ between the sites.

<p align="center">
<img width="3600" height="860" alt="O_fcc_k_point_sweep" src="https://github.com/user-attachments/assets/fbacabd7-b253-4b92-9d4e-572f8a21e936" />

**Figure 10.** k-point mesh comparison for O adsorption at an fcc hollow site, with fixed broadening HWHM = 0.10 eV.

<p align="center">
<img width="2967" height="1609" alt="O_fcc_9x9x1_pdos" src="https://github.com/user-attachments/assets/2a7cfa11-1486-4aa0-bdb2-6d98e40ac95c" />

**Figure 11.** Orbital-group PDOS for the fcc hollow case at 9x9x1, highlighting an O-related spectral distribution broader than the bridge-site feature.

For the fcc hollow configuration, the calculated PDOS exhibits qualitative similarities to the independently obtained DFT result in its overall spectral shape and distribution of prominent oxygen-related features. This observation suggests that the refined, physically constructed TB Hamiltonian can capture selected aspects of the electronic structure. The broader O-related spectral distribution is also consistent with a different orbital-hybridization pattern in the chosen TB model, although PDOS alone cannot quantify bond strength. Because the DFT reference relaxed from the initial bridge position toward an fcc hollow geometry, this comparison is more appropriate than treating the DFT result as a final bridge-site spectrum. Nevertheless, differences in relaxed geometries, energy alignment, orbital representations, and broadening prevent a claim of quantitative agreement from visual resemblance alone.

<p align="center">
<img width="85%" alt="O_fcc_2p_peak_eigenvector_analysis" src="https://github.com/user-attachments/assets/776fe59f-f619-4bb3-8756-4b3588568bcc" />

**Figure 12**. FCC hollow-site orbital decomposition and representative orbital contributions for an oxygen-related spectral window.

<p align="center">
<img width="85%" alt="O_fcc_Green_resonance_O_atom36" src="https://github.com/user-attachments/assets/3ea5b020-b909-40a7-914f-807629162bf9" />

**Figure 13.** FCC hollow-site Green-function projection consistency (left) and inverse-Green-function resonance diagnostic (right).

### 4.4 Effects of Model Refinement

Comparisons between the saved reduced-basis result and the updated calculation show clear changes in the Pt-derived spectral envelope and in the location and width of adsorbate-related peaks. In particular, the expanded-basis and k-sampled calculation reduces the artificial appearance of multiple discrete Pt peaks. However, a sharp oxygen feature remains in the bridge case, while the fcc hollow case exhibits a more distributed O 2p contribution.

The initial/refined distinction is Pt 5d / O 2p versus Pt 5d + 6s / O 2s + 2p (H 1s unchanged when included). The displayed before-and-after plots additionally differ in k-point sampling, so the improvement cannot be attributed quantitatively to basis expansion alone without a same-mesh control.

<p align="center">
<img width="3567" height="1159" alt="O_bridge_comparison" src="https://github.com/user-attachments/assets/525720b9-1001-4b1a-80d6-e8b36fc282c1" />

**Figure 14.** Comparison of the saved reduced model and updated model at the bridge site: Pt 5d (left) and O 2p (right).

<p align="center">
<img width="3567" height="1159" alt="O_fcc_comparison" src="https://github.com/user-attachments/assets/9d2f5c1d-0caa-40ed-9ae7-a4141ed4840c" />

**Figure 15.** Comparison of the saved reduced model and updated model at the fcc hollow site: Pt 5d (left) and O 2p (right).

These plots should be interpreted as an overall before-and-after comparison rather than a controlled demonstration that a single modification caused the improvement: the orbital basis, fitted parameter space, and reciprocal-space sampling changed together. The apparent convergence near 5 × 5 × 1 is based on the displayed PDOS profiles, not a stated quantitative convergence tolerance or a systematic error metric against matched DFT spectra.

Several limitations remain. MACE-derived energies and forces do not uniquely determine an electronic Hamiltonian, and the fitted TB free-energy expression uses phenomenological pair repulsion rather than a full first-principles energy functional. The Γ-only fitting and post-fit k-sampled PDOS are not fully self-consistent. Differences in spin treatment, reference geometry, Fermi-level alignment, and spectral broadening complicate comparisons with DFT; the non-spin-polarized approximation is especially restrictive for O₂.

### 4.5. Configuration-Dependent Spectral Changes and k-Point Convergence

#### 4.5.1. Motivation and Quantitative Comparison Across 16 Adsorption Configurations

As discussed in Section 4.2, the comparison between the calculated Tight-Binding (TB) projected density of states (PDOS) and the independently obtained DFT reference for O adsorption at the fcc hollow site revealed discrepancies in characteristic O 2p peak positions and spectral distributions.

Although the refined model exhibited qualitative similarities to the DFT reference, the remaining differences highlighted the limitations of visual comparisons and motivated a quantitative analysis of spectral changes.

Accordingly, the spectral descriptors defined in Section 3.6 were employed to systematically compare the initial and refined TB models across 16 adsorption configurations, consisting of four adsorbates (O, O₂, OH, and OOH) and four initial adsorption-site labels (ontop, bridge, fcc hollow, and hcp hollow).

The Wasserstein-1 distance was first evaluated for the O 2p and Pt 5d orbital groups to characterize differences in their normalized spectral distributions.

<table>
  <tr>
    <td width="50%" align="center">
      <img width="100%" alt="Wasserstein_O2p_16case_heatmap" src="https://github.com/user-attachments/assets/f45031e1-38d0-48d8-bd17-8cb4fe52d192" />
      <br>
      (a) O 2p
    </td>
    <td width="50%" align="center">
      <img width="100%" alt="Wasserstein_Pt5d_16case_heatmap" src="https://github.com/user-attachments/assets/f32abd51-c7a7-4f3e-94f4-8e818d75e5bf" />
      <br>
      (b) Pt 5d
    </td>
  </tr>
</table>

**Figure 16.** Wasserstein-1 distances for (a) O 2p and (b) Pt 5d PDOS across 16 adsorption configurations. Each value represents the difference between the initial Γ-point and refined 9×9×1 spectra, normalized to unit spectral weight and expressed relative to their respective Fermi levels.

The results revealed substantial configuration-dependent variations in the electronic spectra.

Among the investigated configurations, O–hcp exhibited the largest O 2p Wasserstein distance (2.355 eV), followed by OOH–ontop (2.083 eV). O₂–hcp also showed a notable difference of 1.000 eV.

However, the Pt 5d results exhibited a different trend. OOH–ontop showed the largest Pt 5d Wasserstein distance (2.294 eV), whereas O–hcp exhibited a substantially smaller value (0.698 eV). The corresponding Pt 5d distance for O₂–hcp was 1.099 eV.

These observations indicate that the spectral responses of the adsorbate-derived and metal-derived orbital groups are not necessarily identical.

To further characterize these differences, the changes in spectral centroid, main peak position, full width at half maximum (FWHM), and spectral standard deviation were evaluated.

<table>
  <tr>
    <td width="50%" align="center">
      <img width="100%" alt="O2p_centroid_shift_heatmap" src="https://github.com/user-attachments/assets/7a9d9bbb-6810-479e-980e-7fa067af20f5" />
      <br>
      (a) Spectral centroid
    </td>
    <td width="50%" align="center">
      <img width="100%" alt="O2p_peak_position_shift_heatmap" src="https://github.com/user-attachments/assets/f2acb055-aa0e-414c-a5b4-ca263d9f8c34" />
      <br>
      (b) Main peak position
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <<img width="100%" alt="O2p_peak_FWHM_change_heatmap" src="https://github.com/user-attachments/assets/f98a68ae-06bf-4610-ae87-24e8b7b76677" />
      <br>
      (c) FWHM
    </td>
    <td width="50%" align="center">
      <img width="100%" alt="O2p_spectral_sigma_change_heatmap" src="https://github.com/user-attachments/assets/8ed02aab-13f2-4ef2-858b-4c617d3315cf" />
      <br>
      (d) Spectral standard deviation
    </td>
  </tr>
</table>

**Figure 17.** Changes in the O 2p spectral descriptors between the initial and refined models: (a) spectral centroid, (b) main peak position, (c) FWHM, and (d) spectral standard deviation. All signed values are calculated as Refined − Initial.

The descriptor analysis revealed that similar Wasserstein distances may originate from different types of spectral changes. In particular, substantial differences in peak position do not necessarily correspond to comparable changes in the overall spectral distribution.

Based on these observations, O–hcp, O₂–hcp, and OOH–ontop were selected for detailed investigation. These configurations exhibit distinct combinations of overall spectral redistribution, energy displacement, and changes in the Pt-derived electronic spectra.

#### 4.5.2. Detailed Analysis of Selected Adsorption Configurations

**O adsorption at the hcp hollow site**

O–hcp exhibited the largest O 2p Wasserstein distance (2.355 eV) among the investigated configurations.

Interestingly, the spectral centroid shifted by only −0.464 eV, whereas the main peak position shifted by −1.948 eV. The FWHM remained nearly unchanged (−0.016 eV), while the spectral standard deviation increased substantially by +2.074 eV.

These observations suggest that the spectral difference cannot be explained solely by a uniform energy displacement or broadening of the dominant peak.

Instead, the substantial increase in spectral standard deviation is consistent with redistribution of spectral weight across different energy regions, potentially involving additional spectral features or changes in the relative intensities of existing peaks.

Furthermore, the comparatively modest centroid shift indicates that the redistribution does not produce an equally large displacement of the average spectral energy.

Thus, O–hcp represents a configuration in which changes in the overall spectral distribution are considerably more pronounced than those captured by the dominant peak width alone.

**O₂ adsorption at the hcp hollow site**

In contrast, O₂–hcp exhibited an O 2p Wasserstein distance of 1.000 eV and a spectral centroid shift of +0.988 eV.

The nearly identical magnitudes of these quantities suggest that an overall energy displacement accounts for much of the difference between the two normalized spectra.

The main peak position shifted by +1.086 eV, while the spectral standard deviation increased by only +0.104 eV. The FWHM decreased slightly by −0.064 eV.

Together, these observations indicate that the O₂–hcp spectrum is characterized primarily by an energy displacement rather than a pronounced increase in spectral dispersion.

The Pt 5d Wasserstein distance was also relatively large (1.099 eV), indicating noticeable differences in both metal- and adsorbate-projected spectra.

Nevertheless, the present non-spin-polarized TB approximation limits the physical interpretation of the O₂ electronic structure.

**OOH adsorption at the ontop site**

The OOH–ontop configuration exhibited substantial Wasserstein distances for both O 2p (2.083 eV) and Pt 5d (2.294 eV).

The O 2p centroid shifted by −2.069 eV, nearly matching the magnitude of its Wasserstein distance. The main peak position exhibited a larger shift of −3.178 eV.

In contrast, the FWHM and spectral standard deviation increased by only +0.034 eV and +0.205 eV, respectively.

These results indicate that the spectral difference is predominantly associated with a displacement of the O 2p energy distribution rather than substantial overall broadening. The larger main peak shift may additionally reflect changes in relative peak intensities or a change in the identity of the dominant spectral feature.

The comparatively large Pt 5d Wasserstein distance distinguishes OOH–ontop from O–hcp, suggesting that model refinement produces substantial differences in both orbital groups.

However, these observations do not independently demonstrate stronger Pt–O hybridization or changes in adsorption stability.

Overall, the three configurations demonstrate that large Wasserstein distances can arise from distinct spectral responses. Consequently, distribution-based descriptors and peak-specific metrics provide complementary information rather than interchangeable measures of electronic-structure change.

#### 4.5.3. Influence of k-Point Sampling and Fermi-Level Alignment

The preceding 16-configuration analysis compared the initial Γ-point and refined 9×9×1 TB-PDOS. However, the spectra were referenced to the respective Fermi levels of the two models.

Consequently, the calculated Wasserstein distances include differences associated with the orbital basis, fitted Hamiltonian parameters, reciprocal-space sampling, and energy references. Therefore, these results cannot independently identify the contribution of k-point sampling.

To investigate the effect of reciprocal-space sampling under controlled conditions, additional calculations were performed using the same refined Hamiltonian with fixed fitted parameters.

Five k-point meshes were considered: 1×1×1, 3×3×1, 5×5×1, 9×9×1, and 12×12×1. The 12×12×1 result was selected as the finest available reference.

<table>
  <tr>
    <td align="center">
      <img width="85%" alt="O_hcp_k_point_sweep" src="https://github.com/user-attachments/assets/aa6c77d0-a677-44a7-ae99-13d2da44c993" />
      <br>(a) O–hcp
    </td>
  </tr>
  <tr>
    <td align="center">
      <img width="85%" alt="O2_hcp_k_point_sweep" src="https://github.com/user-attachments/assets/fe54e087-d564-40e3-bfea-dbecf8d7f9e3" />
      <br>(b) O₂–hcp
    </td>
  </tr>
  <tr>
    <td align="center">
      <img width="85%" alt="OOH_ontop_k_point_sweep" src="https://github.com/user-attachments/assets/7b560522-0e06-49cd-950b-6bd3dd389471" />
      <br>(c) OOH–ontop
    </td>
  </tr>
</table>

**Figure 20.** k-point dependence of Pt 5d, 6s, O 2p and O 2s PDOS for (a) O–hcp, (b) O₂–hcp, and (c) OOH–ontop. All spectra were calculated using the same refined TB Hamiltonian and fixed Gaussian broadening, with only the k-point sampling density varied.

To quantitatively evaluate k-point convergence, the Wasserstein-1 distance, spectral centroid, and spectral standard deviation were compared with those obtained using the 12×12×1 mesh.

<p align="center">
<img width="85%" alt="kmesh_convergence_3cases" src="https://github.com/user-attachments/assets/9201e7aa-b6c4-4020-a5a8-e455af8f011c" />

**Figure 21.** k-point convergence of the refined TB-PDOS for O–hcp, O₂–hcp, and OOH–ontop relative to the 12×12×1 reference, showing (a) O 2p Wasserstein-1 distance, (b) Pt 5d Wasserstein-1 distance, (c) O 2p centroid difference, and (d) O 2p spectral standard-deviation difference. In this figure, each spectrum is expressed relative to its own calculated Fermi level.

The initial convergence analysis revealed a rapid decrease in Wasserstein distances with increasing k-point mesh density.

However, the spectral centroid differences were found to be almost exactly equal in magnitude and opposite in sign to the corresponding Fermi-level differences.

For example, the Γ-point OOH–ontop spectrum exhibited a centroid difference of −0.202587 eV relative to the 12×12×1 result, accompanied by a Fermi-level difference of +0.202587 eV.

Similar relationships were observed for O–hcp and O₂–hcp.

Because the spectral energies were expressed relative to the Fermi level, the centroid differences satisfy

$$
\Delta\mu_{\mathrm{absolute}}-\Delta E_F
]
$$

The observed correspondence therefore indicates that the apparent centroid shifts arise almost entirely from differences in the calculated Fermi levels, rather than shifts in the absolute mean energies of the orbital-projected spectra.

Furthermore, the O 2p spectral standard deviations remained nearly invariant across the investigated meshes, with differences generally on the order of 10⁻⁷ eV or smaller.

These observations are consistent with the stability of the low-order spectral moments of the same orthogonal TB Hamiltonian under the investigated reciprocal-space sampling schemes.

Absolute-energy Wasserstein analysis

To distinguish differences caused by k-dependent Fermi-level references from changes in the spectral distributions themselves, the Wasserstein analysis was repeated using a common absolute-energy reference for all k-point meshes of each refined Hamiltonian.

The energy axes were reconstructed from the saved Fermi-relative spectra according to

$$
E_{\mathrm{absolute}}=(E-E_F)+E_F
$$

Because the Hamiltonian and fitted parameters were unchanged within each configuration, this procedure enabled direct comparison of the PDOS distributions without introducing an independent Fermi-level reference shift for each mesh.

The resulting Wasserstein distances are summarized in Table 4.

**Table 4.** k-point convergence of the refined TB-PDOS relative to the 12×12×1 mesh. The Fermi-relative O 2p distances were calculated using each mesh's own Fermi level, while the absolute-energy distances were evaluated on a common energy scale within the same refined Hamiltonian. All values are in eV.

| adsorbate   | site   | kmesh   |   W1_relative_O2p_eV |   W1_absolute_O2p_eV |   W1_absolute_Pt5d_eV |   delta_centroid_absolute_eV |   delta_sigma_absolute_eV |
|:------------|:-------|:--------|---------------------:|---------------------:|----------------------:|-----------------------------:|--------------------------:|
| O           | hcp    | 1x1x1   |             0.033924 |             0.028754 |              0.059389 |                     0.000001 |                  0.000000 |
| O           | hcp    | 3x3x1   |             0.008296 |             0.001468 |              0.007016 |                    -0.000000 |                 -0.000000 |
| O           | hcp    | 5x5x1   |             0.000990 |             0.000284 |              0.001376 |                     0.000001 |                  0.000000 |
| O           | hcp    | 9x9x1   |             0.000766 |             0.000022 |              0.000176 |                     0.000000 |                 -0.000000 |
| O           | hcp    | 12x12x1 |             0.000000 |             0.000000 |              0.000000 |                     0.000000 |                  0.000000 |
| O2          | hcp    | 1x1x1   |             0.051691 |             0.007406 |              0.038731 |                     0.000000 |                 -0.000000 |
| O2          | hcp    | 3x3x1   |             0.004626 |             0.000639 |              0.002745 |                     0.000000 |                  0.000000 |
| O2          | hcp    | 5x5x1   |             0.001309 |             0.000237 |              0.000809 |                     0.000000 |                  0.000000 |
| O2          | hcp    | 9x9x1   |             0.000676 |             0.000066 |              0.000236 |                    -0.000000 |                 -0.000000 |
| O2          | hcp    | 12x12x1 |             0.000000 |             0.000000 |              0.000000 |                     0.000000 |                  0.000000 |
| OOH         | ontop  | 1x1x1   |             0.203826 |             0.022382 |              0.174894 |                    -0.000000 |                 -0.000000 |
| OOH         | ontop  | 3x3x1   |             0.038349 |             0.005169 |              0.020352 |                    -0.000000 |                 -0.000000 |
| OOH         | ontop  | 5x5x1   |             0.005372 |             0.002395 |              0.006833 |                    -0.000000 |                 -0.000000 |
| OOH         | ontop  | 9x9x1   |             0.002800 |             0.000609 |              0.001766 |                    -0.000000 |                 -0.000000 |
| OOH         | ontop  | 12x12x1 |             0.000000 |             0.000000 |              0.000000 |                     0.000000 |                  0.000000 |

The absolute-energy analysis revealed substantially smaller Wasserstein distances than those obtained using independently Fermi-referenced spectra, particularly for O₂–hcp and OOH–ontop.

For example, the Γ-point OOH–ontop O 2p Wasserstein distance decreased from 0.203826 eV to 0.022382 eV after aligning the spectra to a common absolute-energy scale.

Similarly, the corresponding distance for O₂–hcp decreased from 0.051691 eV to 0.007406 eV.

These differences indicate that Fermi-level reference shifts made substantial contributions to the apparent k-point dependence of the O 2p spectra.

Nevertheless, nonzero absolute-energy Wasserstein distances remained, reflecting changes in the detailed spectral distributions associated with reciprocal-space sampling.

For OOH–ontop, the absolute-energy Pt 5d Wasserstein distance at the Γ-point remained relatively large (0.174894 eV), suggesting that its metal-projected spectral distribution is more sensitive to sparse k-point sampling than its O 2p distribution.

With increasing k-point mesh density, the absolute-energy Wasserstein distances decreased substantially.

At 9×9×1, all three configurations exhibited O 2p Wasserstein distances below 0.001 eV relative to the 12×12×1 reference. The corresponding Pt 5d distances were also small, with the largest value being 0.001766 eV for OOH–ontop.

These results support the numerical adequacy of the 9×9×1 mesh for the investigated PDOS calculations under the present broadening conditions.

However, because the 12×12×1 mesh was used as the numerical reference rather than a demonstrably fully converged result, convergence with respect to still finer k-point meshes remains to be established.

## 5. Conclusion

This project developed an automated computational workflow integrating MACE-derived atomic energies and forces with a physically constructed Slater–Koster Tight-Binding (TB) Hamiltonian for Pt(111) adsorption systems. Rather than directly employing DFT-derived Hamiltonians, eigenvalues, or projected densities of states (PDOS) as fitting targets, the proposed approach optimized onsite energies, hopping parameters, and repulsive interactions using reference energies and forces obtained from a pretrained machine-learning interatomic potential. The resulting Hamiltonians enabled orbital-resolved electronic-structure analysis, while Green-function and self-energy calculations provided additional internal consistency checks.

The initial reduced-basis model exhibited excessively sharp oxygen-related spectral features compared with the available DFT reference. To address this limitation, the orbital basis was expanded to include Pt 6s and O 2s states, followed by systematic k-point sampling. The refined model produced smoother Pt-derived spectral distributions, while the O–fcc configuration exhibited qualitative similarities to the DFT-PDOS. Nevertheless, discrepancies in characteristic oxygen-related peaks remained, indicating the need for further evaluation of the reconstructed electronic structures.

To quantitatively investigate these differences, Wasserstein-1 distance and complementary spectral descriptors were evaluated across 16 adsorption configurations. The analysis revealed distinct configuration-dependent responses to model refinement. In particular, O–hcp exhibited substantial spectral-weight redistribution, O₂–hcp showed a pronounced overall energy displacement, and OOH–ontop exhibited substantial changes in both metal- and adsorbate-projected spectra. These observations demonstrate that electronic-structure differences cannot be adequately characterized by individual peak positions alone.

Further analysis of k-point convergence revealed the importance of consistent energy-reference alignment. Apparent changes in the spectral centroid were largely attributable to differences in the calculated Fermi levels. When the PDOS was compared on a common absolute-energy scale within the same refined Hamiltonian, the Wasserstein distances decreased substantially and the low-order spectral moments remained nearly invariant. The results also supported the numerical adequacy of the 9×9×1 mesh for the investigated PDOS calculations.

Despite these findings, the present approach remains an exploratory demonstration rather than a validated alternative to first-principles electronic-structure calculations. MACE-derived energies and forces do not uniquely determine an electronic Hamiltonian, and successful energy–force fitting does not necessarily guarantee accurate orbital-resolved spectra. Moreover, the initial-to-refined comparison involves simultaneous changes in orbital basis, fitted parameters, and reciprocal-space sampling. Differences in Fermi-level alignment, structural configurations, and spin treatment introduce additional limitations, particularly for O₂-containing systems. The internal consistency of Green-function and eigenvector-based PDOS calculations likewise does not establish agreement with DFT.

Future work should prioritize systematic comparisons with DFT-derived electronic structures using matched adsorption geometries, consistent energy references, and comparable computational settings. Separately evaluating orbital-basis expansion and parameter refitting would help clarify their individual contributions to the reconstructed spectra. Furthermore, incorporating Brillouin-zone-integrated energies and forces into TB parameter optimization may improve consistency between the fitted Hamiltonian and its k-resolved electronic properties.

Overall, this work demonstrates the feasibility of combining machine-learning-assisted automation with physically interpretable Hamiltonian construction, while highlighting the importance of independent electronic-structure validation and controlled numerical analysis.

## 6. References

<a id="ref1"></a>
[1] Qian, C., Vitartas, V., Kermode, J. R., & Maurer, R. J. (2026). Equivariant electronic Hamiltonian prediction with many-body message passing. *npj Computational Materials*, *12*(1), 169.

<a id="ref2"></a>
[2] Choudhary, K. (2025). Slakonet: A unified slater-koster tight-binding framework using neural network infrastructure for the periodic table. *The Journal of Physical Chemistry Letters*, *16*(43), 11109-11119.

<a id="ref3"></a>
[3] Fried, H. P., Barragan-Yani, D., Libisch, F., & Wirtz, L. (2025). A machine learning approach to predict tight-binding parameters for point defects via the projected density of states. *npj Computational Materials*, *11*(1), 176.

<a id="ref4"></a>
[4] Schwade, M., Zhang, S., Vonhoff, F., Delgado, F. P., & Egger, D. A. (2026). Physics-informed Hamiltonian learning for large-scale optoelectronic property prediction. *Nature Communications*, *17*(1), 2652.

<a id="ref5"></a>
[5] Zhu, X., Rinke, P., & Egger, D. A. (2026). Predicting temperature-dependent optoelectronic properties of semiconductor defects with equivariant neural networks. *npj Computational Materials*, *12*(1), 176.

<a id="ref6"></a>
[6] Xia, Y., Thie, A., Soon, J., & Barca, G. M. (2026). NN-xTB: density functional accuracy at semi empirical speed with neural network extended tight binding. *Nature Communications*.

<a id="ref7"></a>
[7] Batatia, I., Kovacs, D. P., Simm, G., Ortner, C., & Csányi, G. (2022). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *Advances in neural information processing systems*, *35*, 11423-11436.

<a id="ref8"></a>
[8] Koskinen, P., & Mäkinen, V. (2009). Density-functional tight-binding for beginners. *Computational Materials Science*, *47*(1), 237-253.

<a id="ref9"></a>
[9] Synopsys QuantumATK, “ProjectedDensityOfStates,” documentation. https://docs.quantumatk.com/manual/Types/ProjectedDensityOfStates/ProjectedDensityOfStates.html

<a id="ref10"></a>
[10] Shuttleworth, I. G. (2018). Binding site transitions across strained oxygenated and hydroxylated Pt (111). *ChemistryOpen*, *7*(5), 356-369.

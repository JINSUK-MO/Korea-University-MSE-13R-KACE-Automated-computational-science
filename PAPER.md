# Physics-based and MACE-assisted Hamiltonian Construction

Korea University AA Co-E Creator Club (Sep. – Dec. 2026)  
Department of Materials Science and Engineering

**Team Members:** [@Seoyeong Ahn](https://github.com/seo00224), [@Jewoo Yeon](https://github.com/jewoo1041304-coder), [@Jiyeon Lee](https://github.com/jiyeonie322), [@Yunseok Jung](https://github.com/Sukee03), [@Junyoung Choi](https://github.com/chlwnsdud27-cloud), and [@Jinsuk Mo](https://github.com/JINSUK-MO)


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

> **Table 1 note:** The original table is embedded as an image and is not transcribed here. Refer to Koskinen and Mäkinen (2009) [[8]](#ref8) for the full angular-coefficient table.

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

The FWHM was employed to characterize the width of the dominant spectral peak. It was determined from the energy interval between the two half-maximum crossings surrounding the highest-intensity peak. This descriptor is particularly relevant to the present project because the initial TB model exhibited excessively sharp oxygen-related spectral features.

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

The initial model employed Pt 5d, O 2p, and H 1s orbitals with Γ-point PDOS sampling. The refined model incorporated Pt 5d + 6s, O 2s + 2p, and H 1s orbitals, with the reported PDOS evaluated using a 9×9×1 k-point mesh. Both models retained the same fundamental Slater–Koster Hamiltonian construction and MACE Energy/Force fitting framework, although the refined model involved an expanded parameter space and subsequent parameter optimization. For both models, energies were expressed relative to their respective Fermi levels. The Wasserstein distances were calculated from normalized spectral distributions, whereas the other descriptors were evaluated directly from the corresponding PDOS curves. All signed descriptor changes were defined as Refined − Initial. The Wasserstein distance, by contrast, is nonnegative and represents the magnitude of the spectral distribution difference.

Importantly, the present analysis compares two TB models rather than calculating discrepancies relative to DFT. Furthermore, because the orbital basis, fitted parameters, and PDOS sampling conditions differ between the models, their individual contributions cannot be isolated from this comparison alone. To examine the independent numerical effect of reciprocal-space sampling, additional k-point sweeps were considered using the same refined Hamiltonian and its fixed fitted parameters.

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
<img width="70%" alt="O_bridge_9x9x1_pdos" src="https://github.com/user-attachments/assets/a109b92c-6ab9-45da-9a2e-7c7b8863b38f" />

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
<img width="3600" height="860" alt="O_fcc_k_point_sweep" src="https://github.com/user-attachments/assets/92292a0f-ba6b-4866-baf9-aeea9a07a5c3" />

**Figure 10.** k-point mesh comparison for O adsorption at an fcc hollow site, with fixed broadening HWHM = 0.10 eV.

<p align="center">
<img width="70%" alt="O_fcc_9x9x1_pdos" src="https://github.com/user-attachments/assets/9aec6770-21bf-4d5f-b2b3-a4485cdcdc52" />

**Figure 11.** Orbital-group PDOS for the fcc hollow case at 9x9x1, highlighting an O-related spectral distribution broader than the bridge-site feature.

For the fcc hollow configuration, the calculated PDOS exhibits qualitative similarities to the independently obtained DFT result in its overall spectral shape and distribution of prominent oxygen-related features. This observation suggests that the refined, physically constructed TB Hamiltonian can capture selected aspects of the electronic structure. The broader O-related spectral distribution is also consistent with a different orbital-hybridization pattern in the chosen TB model, although PDOS alone cannot quantify bond strength. Because the DFT reference relaxed from the initial bridge position toward an fcc hollow geometry, this comparison is more appropriate than treating the DFT result as a final bridge-site spectrum. Nevertheless, differences in relaxed geometries, energy alignment, orbital representations, and broadening prevent a claim of quantitative agreement from visual resemblance alone.

<p align="center">
<img width="3867" height="1309" alt="O_2p_peak_eigenvector_analysis" src="https://github.com/user-attachments/assets/1362b15b-0cfb-438a-b740-fdcb31245e2e" />

**Figure 12**. FCC hollow-site orbital decomposition and representative orbital contributions for an oxygen-related spectral window.

<p align="center">
<img width="2974" height="1018" alt="O_fcc_green_funtion" src="https://github.com/user-attachments/assets/cd055609-26ee-4fa9-ade6-d61c48f04b2b" />

**Figure 13.** FCC hollow-site Green-function projection consistency (left) and inverse-Green-function resonance diagnostic (right).

### 4.4 Effects of Model Refinement

Comparisons between the saved reduced-basis result and the updated calculation show clear changes in the Pt-derived spectral envelope and in the location and width of adsorbate-related peaks. In particular, the expanded-basis and k-sampled calculation reduces the artificial appearance of multiple discrete Pt peaks. However, a sharp oxygen feature remains in the bridge case, while the fcc hollow case exhibits a more distributed O 2p contribution.

The initial/refined distinction is Pt 5d / O 2p versus Pt 5d + 6s / O 2s + 2p (H 1s unchanged when included). The displayed before-and-after plots additionally differ in k-point sampling, so the improvement cannot be attributed quantitatively to basis expansion alone without a same-mesh control.

<p align="center">
<img width="3567" height="1159" alt="O_bridge_comparison" src="https://github.com/user-attachments/assets/525720b9-1001-4b1a-80d6-e8b36fc282c1" />

**Figure 14.** Comparison of the saved reduced model and updated model at the bridge site: Pt 5d (left) and O 2p (right).

<p align="center">
<img width="3567" height="1159" alt="O_fcc_comparison" src="https://github.com/user-attachments/assets/ddc1532d-ca67-4b16-8d2a-6655c26d2324" />

**Figure 15.** Comparison of the saved reduced model and updated model at the fcc hollow site: Pt 5d (left) and O 2p (right).

These plots should be interpreted as an overall before-and-after comparison rather than a controlled demonstration that a single modification caused the improvement: the orbital basis, fitted parameter space, and reciprocal-space sampling changed together. The apparent convergence near 5 × 5 × 1 is based on the displayed PDOS profiles, not a stated quantitative convergence tolerance or a systematic error metric against matched DFT spectra.

Post-fitting k-point sampling does not alter the optimized TB parameters or the eigenvalues at a fixed k-point. **Instead, the k-dependent Bloch Hamiltonian yields different eigenvalues and orbital-projection weights throughout the Brillouin zone.** Increasing the sampling density therefore changes the set of eigenstates contributing to the calculated PDOS, allowing discrete features in the Γ-point spectrum to develop into smoother spectral distributions. Changes in the calculated Fermi level may additionally affect the apparent peak positions on the Fermi-relative energy scale. However, because the initial and refined models in Fig. 15 also differ in their orbital bases and fitted parameters, the observed spectral changes cannot be attributed exclusively to the post-fitting k-point sampling.

Several limitations remain. MACE-derived energies and forces do not uniquely determine an electronic Hamiltonian, and the fitted TB free-energy expression uses phenomenological pair repulsion rather than a full first-principles energy functional. The Γ-only fitting and post-fit k-sampled PDOS are not fully self-consistent. Differences in spin treatment, reference geometry, Fermi-level alignment, and spectral broadening complicate comparisons with DFT; the non-spin-polarized approximation is especially restrictive for O₂.

### 4.5. Configuration-Dependent Spectral Changes and k-Point Convergence

#### 4.5.1. Motivation and Quantitative Comparison Across 16 Adsorption Configurations

As discussed in Section 4.2, the comparison between the calculated Tight-Binding (TB) projected density of states (PDOS) and the independently obtained DFT reference for O adsorption at the fcc hollow site revealed discrepancies in characteristic O 2p peak positions and spectral distributions.

Although the refined model exhibited qualitative similarities to the DFT reference, the remaining differences highlighted the limitations of visual comparisons and motivated a quantitative analysis of spectral changes. Accordingly, the spectral descriptors defined in Section 3.6 were employed to systematically compare the initial and refined TB models across 16 adsorption configurations, consisting of four adsorbates (O, O₂, OH, and OOH) and four initial adsorption-site labels (ontop, bridge, fcc hollow, and hcp hollow). The Wasserstein-1 distance was first evaluated for the O 2p and Pt 5d orbital groups to characterize differences in their normalized spectral distributions.

<table>
  <tr>
    <td width="50%" align="center">
      <img width="100%" alt="Wasserstein_O2p_16case_heatmap" src="https://github.com/user-attachments/assets/1d5fb5db-e5cc-467a-b774-659ecf4c1b23" />
      <br>
      (a) O 2p
    </td>
    <td width="50%" align="center">
      <img width="100%" alt="Wasserstein_Pt5d_16case_heatmap" src="https://github.com/user-attachments/assets/520c55b1-439d-4cf3-9aca-4e75a1d75192" />
      <br>
      (b) Pt 5d
    </td>
  </tr>
</table>

**Figure 16.** Wasserstein-1 distances for (a) O 2p and (b) Pt 5d PDOS across 16 adsorption configurations. Each value represents the difference between the initial Γ-point and refined 9×9×1 spectra, normalized to unit spectral weight and expressed relative to their respective Fermi levels.

The results revealed substantial configuration-dependent variations in the electronic spectra. Among the investigated configurations, O–hcp exhibited the largest O 2p Wasserstein distance (2.355 eV), followed by OOH–ontop (2.083 eV). O₂–hcp also showed a notable difference of 1.000 eV. However, the Pt 5d results exhibited a different trend. OOH–ontop showed the largest Pt 5d Wasserstein distance (2.294 eV), whereas O–hcp exhibited a substantially smaller value (0.698 eV). The corresponding Pt 5d distance for O₂–hcp was 1.099 eV.

These observations indicate that the spectral responses of the adsorbate-derived and metal-derived orbital groups are not necessarily identical. To further characterize these differences, the changes in spectral centroid, main peak position, full width at half maximum (FWHM), and spectral standard deviation were evaluated.

<table>
  <tr>
    <td width="50%" align="center">
      <img width="100%" alt="O2p_centroid_shift_heatmap" src="https://github.com/user-attachments/assets/b3185919-feea-4a24-b107-9550691e6171" />
      <br>
      (a) Spectral centroid
    </td>
    <td width="50%" align="center">
    <img width="100%" alt="O2p_peak_position_shift_heatmap" src="https://github.com/user-attachments/assets/9f8afb4e-b1ac-44e4-a917-269711098c16" />
      <br>
      (b) Main peak position
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img width="100%" alt="O2p_peak_FWHM_change_heatmap" src="https://github.com/user-attachments/assets/25c3d8e1-96c8-42b2-aec5-6ff7145dd588" />
      <br>
      (c) FWHM
    </td>
    <td width="50%" align="center">
      <img width="100%" alt="O2p_spectral_sigma_change_heatmap" src="https://github.com/user-attachments/assets/4a9c23e7-98f7-48e2-a2ec-e086c95b42f8" />
      <br>
      (d) Spectral standard deviation
    </td>
  </tr>
</table>

**Figure 17.** Changes in the O 2p spectral descriptors between the initial and refined models: (a) spectral centroid, (b) main peak position, (c) FWHM, and (d) spectral standard deviation. All signed values are calculated as Refined − Initial.

The descriptor analysis revealed that similar Wasserstein distances may originate from different types of spectral changes. In particular, substantial differences in peak position do not necessarily correspond to comparable changes in the overall spectral distribution.

Based on these observations, O–hcp, O₂–hcp, and OOH–ontop were selected for detailed investigation. These configurations exhibit distinct combinations of overall spectral redistribution, energy displacement, and changes in the Pt-derived electronic spectra.

#### 4.5.2. Detailed Analysis of Selected Adsorption Configurations

**O adsorption at the fcc hollow site**

The O–fcc configuration exhibited the largest O 2p Wasserstein-1 distance (2.391 eV) among the 16 investigated adsorption configurations, accompanied by a substantial Pt 5d distance (1.891 eV) (Figure 16). The O 2p spectral centroid shifted by −1.354 eV, whereas the dominant peak position shifted by −3.654 eV, indicating that the movement of the main peak does not fully represent the change in the overall spectral distribution (Figure 17(a)).

Furthermore, the spectral standard deviation increased by 2.223 eV, demonstrating a substantial redistribution of spectral weight across the energy range. The calculated FWHM also increased markedly, although this descriptor requires additional verification because of the potentially multimodal nature of the O 2p spectrum (Figure 17(c)). Together, these observations suggest that O–fcc undergoes pronounced configuration-dependent spectral changes following model refinement, involving both adsorbate- and metal-projected states.

Nevertheless, these differences should not be interpreted as direct evidence of improved agreement with DFT or stronger Pt–O hybridization, since the initial and refined calculations involve changes in orbital basis, fitted parameters, reciprocal-space sampling, and Fermi-level references.

**O adsorption at the hcp hollow site**

O–hcp exhibited the largest O 2p Wasserstein distance (2.355 eV) among the investigated configurations. Interestingly, the spectral centroid shifted by only −0.464 eV (Figure 16), whereas the main peak position shifted by −1.948 eV. The FWHM remained nearly unchanged (−0.016 eV), while the spectral standard deviation increased substantially by +2.074 eV (Figure 17(c)). These observations suggest that the spectral difference cannot be explained solely by a uniform energy displacement or broadening of the dominant peak.

Instead, the substantial increase in spectral standard deviation is consistent with redistribution of spectral weight across different energy regions, potentially involving additional spectral features or changes in the relative intensities of existing peaks. Furthermore, the comparatively modest centroid shift indicates that the redistribution does not produce an equally large displacement of the average spectral energy. Thus, O–hcp represents a configuration in which changes in the overall spectral distribution are considerably more pronounced than those captured by the dominant peak width alone.

**O₂ adsorption at the hcp hollow site**

In contrast, O₂–hcp exhibited an O 2p Wasserstein distance of 1.000 eV and a spectral centroid shift of +0.988 eV (Figure 16). The nearly identical magnitudes of these quantities suggest that an overall energy displacement accounts for much of the difference between the two normalized spectra. The main peak position shifted by +1.086 eV, while the spectral standard deviation increased by only +0.104 eV (Figure 17(b)). The FWHM decreased slightly by −0.064 eV (Figure 17(c)).

Together, these observations indicate that the O₂–hcp spectrum is characterized primarily by an energy displacement rather than a pronounced increase in spectral dispersion. The Pt 5d Wasserstein distance was also relatively large (1.099 eV), indicating noticeable differences in both metal- and adsorbate-projected spectra. Nevertheless, the present non-spin-polarized TB approximation limits the physical interpretation of the O₂ electronic structure.

**OOH adsorption at the ontop site**

The OOH–ontop configuration exhibited substantial Wasserstein distances for both O 2p (2.083 eV) and Pt 5d (2.294 eV) (Figure 16). The O 2p centroid shifted by −2.069 eV, nearly matching the magnitude of its Wasserstein distance. The main peak position exhibited a larger shift of −3.178 eV. In contrast, the FWHM and spectral standard deviation increased by only +0.034 eV and +0.205 eV, respectively. (Figure 17(c))

These results indicate that the spectral difference is predominantly associated with a displacement of the O 2p energy distribution rather than substantial overall broadening. The larger main peak shift may additionally reflect changes in relative peak intensities or a change in the identity of the dominant spectral feature. The comparatively large Pt 5d Wasserstein distance distinguishes OOH–ontop from O–hcp, suggesting that model refinement produces substantial differences in both orbital groups. However, these observations do not independently demonstrate stronger Pt–O hybridization or changes in adsorption stability.

Overall, the three configurations demonstrate that large Wasserstein distances can arise from distinct spectral responses. Consequently, distribution-based descriptors and peak-specific metrics provide complementary information rather than interchangeable measures of electronic-structure change.

#### 4.5.3. Influence of k-Point Sampling and Fermi-Level Alignment

The preceding 16-configuration analysis compared the initial Γ-point and refined 9×9×1 TB-PDOS. However, the spectra were referenced to the respective Fermi levels of the two models. Consequently, the calculated Wasserstein distances include differences associated with the orbital basis, fitted Hamiltonian parameters, reciprocal-space sampling, and energy references. Therefore, these results cannot independently identify the contribution of k-point sampling.

To investigate the effect of reciprocal-space sampling under controlled conditions, additional calculations were performed using the same refined Hamiltonian with fixed fitted parameters. Five k-point meshes were considered: 1×1×1, 3×3×1, 5×5×1, 9×9×1, and 12×12×1. The 12×12×1 result was selected as the finest available reference.

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

**Figure 18.** k-point dependence of Pt 5d, 6s, O 2p and O 2s PDOS for (a) O–hcp, (b) O₂–hcp, and (c) OOH–ontop. All spectra were calculated using the same refined TB Hamiltonian and fixed Gaussian broadening, with only the k-point sampling density varied.

To quantitatively evaluate k-point convergence, the Wasserstein-1 distance, spectral centroid, and spectral standard deviation were compared with those obtained using the 12×12×1 mesh.

<p align="center">
<img width="85%" alt="kmesh_convergence_4cases" src="https://github.com/user-attachments/assets/7384bc0b-ee9f-4153-b646-bda5a3869f27" />

**Figure 19.** k-point convergence of the refined TB-PDOS for O–hcp, O₂–hcp, and OOH–ontop relative to the 12×12×1 reference, showing (a) O 2p Wasserstein-1 distance, (b) Pt 5d Wasserstein-1 distance, (c) O 2p centroid difference, and (d) O 2p spectral standard-deviation difference. In this figure, each spectrum is expressed relative to its own calculated Fermi level.

The initial convergence analysis revealed a rapid decrease in Wasserstein distances with increasing k-point mesh density. However, the spectral centroid differences were found to be almost exactly equal in magnitude and opposite in sign to the corresponding Fermi-level differences. For example, the Γ-point OOH–ontop spectrum exhibited a centroid difference of −0.202587 eV relative to the 12×12×1 result, accompanied by a Fermi-level difference of +0.202587 eV.

Similar relationships were observed for O–hcp and O₂–hcp. Because the spectral energies were expressed relative to the Fermi level, the centroid differences satisfy

$$
\Delta\mu_{\mathrm{absolute}}-\Delta E_F
$$

The observed correspondence therefore indicates that the apparent centroid shifts arise almost entirely from differences in the calculated Fermi levels, rather than shifts in the absolute mean energies of the orbital-projected spectra. Furthermore, the O 2p spectral standard deviations remained nearly invariant across the investigated meshes, with differences generally on the order of 10⁻⁷ eV or smaller.

These observations are consistent with the stability of the low-order spectral moments of the same orthogonal TB Hamiltonian under the investigated reciprocal-space sampling schemes.

**Absolute-energy Wasserstein analysis**

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
| O           | fcc    | 1×1×1 | 0.293073 | 0.284643 | 0.147598 | +1.967e-06 | +4.892e-07 |
| O           | fcc | 3×3×1 | 0.048595 | 0.048798 | 0.035153 | +3.271e-07 | +1.310e-07 |
| O  | fcc | 5×5×1 | 0.018636 | 0.016579 | 0.014933 | +2.440e-07 | +2.201e-08 |
| O  | fcc | 9×9×1 | 0.007971 | 0.004197 | 0.003650 | +1.317e-07 | +1.952e-08 |
| O  | fcc | 12×12×1 | 0.000000 | 0.000000 | 0.000000 | +0.000e+00 | +0.000e+00 |
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

The absolute-energy analysis revealed substantially smaller Wasserstein distances than those obtained using independently Fermi-referenced spectra, particularly for O₂–hcp and OOH–ontop. For example, the Γ-point OOH–ontop O 2p Wasserstein distance decreased from 0.203826 eV to 0.022382 eV after aligning the spectra to a common absolute-energy scale. Similarly, the corresponding distance for O₂–hcp decreased from 0.051691 eV to 0.007406 eV.

These differences indicate that Fermi-level reference shifts made substantial contributions to the apparent k-point dependence of the O 2p spectra. Nevertheless, nonzero absolute-energy Wasserstein distances remained, reflecting changes in the detailed spectral distributions associated with reciprocal-space sampling. For OOH–ontop, the absolute-energy Pt 5d Wasserstein distance at the Γ-point remained relatively large (0.174894 eV), suggesting that its metal-projected spectral distribution is more sensitive to sparse k-point sampling than its O 2p distribution. 

With increasing k-point mesh density, the absolute-energy Wasserstein distances decreased substantially. At 9×9×1, all three configurations exhibited O 2p Wasserstein distances below 0.001 eV relative to the 12×12×1 reference. The corresponding Pt 5d distances were also small, with the largest value being 0.001766 eV for OOH–ontop.

These results support the numerical adequacy of the 9×9×1 mesh for the investigated PDOS calculations under the present broadening conditions. However, because the 12×12×1 mesh was used as the numerical reference rather than a demonstrably fully converged result, convergence with respect to still finer k-point meshes remains to be established.

### 4.6 Fitting Stability and Parameter Robustness

Since the refined TB model is obtained by numerical optimization, the observed spectral changes may depend not only on the expanded orbital basis itself but also on the stability of the fitted parameters. To examine this issue, additional repeated fitting tests were carried out. The purpose of this section is to determine whether the spectral differences discussed above are reproducible features of the refined model or whether they are strongly affected by the optimization trajectory and parameter sensitivity.

Two complementary analyses were performed. First, all 16 adsorbate–site configurations were re-fitted four additional times, giving a total of five fitting runs for each configuration. The changes in fitting parameters and spectral descriptors were then evaluated relative to the first fitting result. Second, two representative configurations, O–hcp and O–fcc, were selected for extended validation with a larger iteration budget in order to test whether additional optimization improves spectral stability.

#### 4.6.1 Five-run stability test across the 16 configurations

The preceding analysis demonstrated substantial configuration-dependent changes in the projected density of states (PDOS) following refinement of the tight-binding (TB) model. However, the reliability of these spectral differences depends partly on the reproducibility of the fitted Hamiltonian parameters. To assess this sensitivity, four additional fitting runs were performed for each of the 16 adsorption configurations, yielding five parameter sets per configuration. The additional runs employed perturbed initial parameters while retaining the same MACE-derived reference data, fitting objective, regularization settings, and refined orbital basis.

For each configuration, the original result (Fitting 1) was used as the reference for evaluating parameter and spectral variations. The fitted parameters were compared through their signed differences, while the resulting O 2p and Pt 5d PDOS were evaluated using the Wasserstein-1 distance on a common absolute-energy scale. All spectra were calculated with 9×9×1 k-point sampling, thereby excluding differences in reciprocal-space sampling from the repeated-fitting comparison.

Substantial configuration-dependent spectral variability was observed across the additional 64 fitting runs (Fig. 20). The median Wasserstein-1 distances relative to Fitting 1 were 0.517 eV for O 2p and 0.274 eV for Pt 5d, with maximum values reaching 2.590 eV and 2.177 eV, respectively. The O–hcp and O–fcc configurations exhibited particularly pronounced O 2p variability, whereas O–fcc also displayed substantial changes in its Pt 5d spectrum. In contrast, several other adsorption configurations showed comparatively smaller differences between repeated fitting results. These observations indicate that the sensitivity of the reconstructed electronic spectra to parameter initialization is not uniform across adsorption geometries.

<p align="center">
<img width="80%" alt="W1_stability_16cases" src="https://github.com/user-attachments/assets/92513247-b594-41a6-a887-2cff93f8b5f8" />

**Figure 20.** Absolute-energy Wasserstein-1 distances between Fitting 1 and Fitting 2–5 across 16 adsorption configurations for (a) O 2p and (b) Pt 5d PDOS. All spectra were evaluated using the refined TB model with 9×9×1 k-point sampling.

The relationship between fitting performance and spectral reproducibility was further examined by comparing changes in the final fitting cost with the corresponding Wasserstein-1 distances (Fig. 21). No clear monotonic relationship was observed. In several configurations, fitting runs with lower final costs than Fitting 1 nevertheless produced appreciably different electronic spectra. This suggests that improvement in the energy–force fitting objective does not necessarily imply convergence toward the same orbital-resolved spectral distribution. However, spectral similarity to Fitting 1 should not itself be regarded as a measure of physical accuracy, since the reference fitting result has not been independently validated against DFT.

<p align="center">
<img width="80%" alt="cost_vs_W1_stability" src="https://github.com/user-attachments/assets/f55e29d5-b026-4c1a-80f7-9c853330b84a" />

**Figure 21.** Relationship between the change in final fitting cost relative to Fitting 1 and the absolute-energy Wasserstein-1 distance for (a) O 2p and (b) Pt 5d. Each point represents an additional fitting run for one adsorption configuration. All runs reached the prescribed 100-iteration limit without satisfying the optimizer's convergence criteria.

Among the investigated adsorbates, atomic O exhibited the largest median O 2p Wasserstein-1 distance (0.992 eV), indicating comparatively pronounced sensitivity to fitting initialization, particularly at the hcp and fcc hollow sites. A related concern regarding the physical robustness of optimized tight-binding parameters was reported by Moradi et al. (2025), who investigated GFN1-xTB parameter optimization for water adsorption on metal surfaces, including Pt. Their study demonstrated that optimization against DFT-derived adsorption energies could improve fitting accuracy while producing nearly degenerate O 2s and O 2p atomic energy levels, highlighting potential limitations in the physical consistency of optimized orbital parameters [[11]](#ref11). Although their investigation did not directly address atomic O adsorption or PDOS reproducibility, these findings suggest that oxygen-related parameters may require additional physical constraints and sensitivity analysis. Accordingly, the relatively large spectral variability observed for atomic O in the present study warrants further examination of the O onsite energies and Pt–O hopping parameters, rather than being interpreted as direct evidence of parameter non-identifiability.

Additional insight was obtained from the relationships between fitted parameters and spectral descriptors (Fig. 22). The changes in the absolute-energy O 2p spectral centroid closely matched the changes in the corresponding O 2p onsite energy parameter. This numerical correspondence is consistent with the spectral-moment properties of the orthogonal TB Hamiltonian, in which the first moment of an orbital-projected spectrum is determined by the corresponding diagonal Hamiltonian element. Consequently, the centroid shift does not provide an independent measure of parameter robustness in the present model. In contrast, the variations in dominant-peak FWHM and spectral standard deviation were not directly interchangeable, reflecting their different sensitivities to local peak shapes and the overall distribution of spectral weight.

<p align="center">
<img width="80%" alt="descriptor_parameter_relationship" src="https://github.com/user-attachments/assets/504ec6a6-c328-4443-aec3-e434eec7c7b5" />

**Figure 22.** Relationships among fitted parameters and spectral descriptors across the repeated fitting experiments. (a) Absolute-energy O 2p spectral centroid change versus O 2p onsite energy parameter change, demonstrating their numerical correspondence. (b) Absolute changes in dominant-peak FWHM and spectral standard deviation, illustrating the distinction between local peak-width variation and overall spectral dispersion.

Despite these observations, all 80 fitting runs, including the original 16 results, terminated upon reaching the maximum limit of 100 energy–force optimization iterations without satisfying the optimizer's convergence criteria. The observed differences therefore characterize the sensitivity of the finite-iteration fitting results rather than the variability of fully converged solutions. In particular, the current analysis cannot distinguish intrinsic parameter non-identifiability from incomplete numerical optimization. This limitation motivated an extended fitting analysis for representative adsorption configurations.

#### 4.6.2. Extended 500-Iteration Validation for O–hcp and O–fcc

To examine the influence of incomplete optimization on the observed spectral variability, the maximum number of energy–force fitting iterations was increased from 100 to 500. Two configurations, O–hcp and O–fcc, were selected because both exhibited substantial O 2p spectral changes in the initial-to-refined comparison and considerable run-to-run variability in the preceding stability analysis. Five fitting runs were performed for each configuration using the same fitting formulation and controlled initialization settings.

Increasing the optimization limit reduced the final fitting costs in all ten runs. The corresponding projected gradient also decreased in eight of the ten runs, indicating further progress toward numerical optimization. Nevertheless, none of the extended fitting runs satisfied the optimizer's convergence criteria, with all ten terminating at the 500-iteration limit.

The increased optimization budget was accompanied by a general reduction in spectral variability (Fig. 23). For O–hcp, the mean absolute-energy O 2p Wasserstein-1 distance between Fitting 1 and Fitting 2–5 decreased from 1.466 eV at 100 iterations to 0.746 eV at 500 iterations, corresponding to a reduction of approximately 49.1%. The corresponding Pt 5d distance decreased from 0.363 to 0.143 eV, representing a reduction of approximately 60.6%. Similar trends were observed for O–fcc, where the mean O 2p and Pt 5d distances decreased by approximately 21.8% and 68.7%, respectively.

<p align="center">
<img width="80%" alt="W1_stability_100_vs_500_scatter" src="https://github.com/user-attachments/assets/0da8208c-0f3e-4eb7-9a08-e78ccdb5eccd" />

**Figure 23.** Comparison of absolute-energy Wasserstein-1 distances between repeated fitting results obtained with maximum iteration limits of 100 and 500 for (a) O 2p and (b) Pt 5d PDOS. Each point represents Fitting 2–5 relative to Fitting 1 for O–hcp or O–fcc. The dashed diagonal indicates equal spectral variability at both iteration limits; points below the line indicate reduced variability at 500 iterations.

Although the average Wasserstein-1 distances decreased for both orbital groups and configurations, the improvement was not uniform across individual fitting runs. In particular, the O 2p distance for O–fcc Fitting 2 increased from 0.983 to 1.699 eV when the iteration limit was extended. Thus, increasing the optimization budget generally reduced run-to-run variability but did not consistently drive all parameter sets toward an identical spectral distribution.

The remaining variability at 500 iterations was also configuration- and orbital-dependent (Fig. 24). O–fcc retained a larger mean O 2p Wasserstein-1 distance (1.001 eV) than O–hcp (0.746 eV), despite the reduction observed in both configurations. Meanwhile, the corresponding mean Pt 5d distances were smaller, reaching 0.430 eV for O–fcc and 0.143 eV for O–hcp. These differences indicate that the adsorbate- and metal-projected spectra respond differently to the continued parameter optimization.

<table>
  <tr>
    <td align="center">
      <img width="100%" alt="O2p_W1_to_fitting1_eV_absolute_heatmap" src="https://github.com/user-attachments/assets/299c90dd-21ce-4c9f-bd32-abfe923eea16" />
    </td>
    <td align="center">
      <img width="100%" alt="Pt5d_W1_to_fitting1_eV_absolute_heatmap" src="https://github.com/user-attachments/assets/cb621675-a68f-4f81-b002-51977f374f9e" />
    </td>
  </tr>
  <tr>
    <td align="center">(a) W-1 Distance of O 2p PDOS with respect to Fitting 1</td>
    <td align="center">(b) W-1 Distance of Pt 5d PDOS with respect to Fitting 1</td>
  </tr>
</table>

**Figure 24.** Absolute-energy Wasserstein-1 distances between Fitting 1 and Fitting 2–5 for O–hcp and O–fcc obtained with the 500-iteration optimization limit. Panels (a) and (b) represent O 2p and Pt 5d PDOS, respectively. All spectra were calculated using the same refined orbital basis and 9×9×1 k-point sampling.

The persistence of spectral variability was accompanied by differences in the optimized TB parameters (Fig. 25). The parameter-deviation analysis provides a complementary description of fitting stability by identifying variations in the Hamiltonian representation rather than solely in its calculated spectra. However, the magnitude of parameter variation cannot be directly equated with the magnitude of spectral differences, since distinct parameters may have different sensitivities and correlated effects on the resulting electronic eigenstates.

<table>
  <tr>
    <td align="center">
      <img width="100%" alt="O_fcc_parameter_delta_heatmap" src="https://github.com/user-attachments/assets/fd7b8a48-250a-431e-91a0-dd0415c18685" />
    </td>
    <td align="center">
      <img width="100%" alt="O_hcp_parameter_delta_heatmap" src="https://github.com/user-attachments/assets/06fdf192-3e2d-498b-9a7d-05e4b7cbd15a" />
    </td>
  </tr>
  <tr>
    <td align="center">(a) Differences in the O-fcc fitting parameters with respect to Fitting 1</td>
    <td align="center">(b) Differences in the O-hcp fitting parameters with respect to Fitting 1</td>
  </tr>
</table>

**Figure 25.** Changes in fitted Slater–Koster TB parameters across the 500-iteration repeated fitting experiments for (a) O–hcp and (b) O–fcc. Each heatmap displays parameter differences between Fitting 2–5 and Fitting 1, normalized by the corresponding parameter-bound widths. Positive and negative values indicate the direction of parameter variation.

Furthermore, the electronic structures obtained after 500 iterations remained appreciably different from their corresponding 100-iteration results. The mean absolute-energy Wasserstein-1 distances between the two optimization budgets were 1.057 eV for O 2p and 0.320 eV for Pt 5d in O–hcp, and 1.459 eV and 0.806 eV, respectively, in O–fcc. These differences indicate that the previously reported 100-iteration spectra had not reached a numerically stable endpoint.

Overall, the extended fitting experiments suggest that incomplete optimization contributes substantially to the initialization-dependent spectral variability observed in Section 4.6.1. However, the persistence of nonconverged solutions at 500 iterations prevents a definitive assessment of parameter identifiability. Additional optimization convergence tests are therefore required before determining whether the remaining spectral differences arise from distinct converged parameter sets or continued numerical optimization.

#### 4.6.3. Effect of Extending the Optimization Budget to 1500 Iterations

To examine whether the remaining differences in the refined-model PDOS were primarily caused by an insufficient optimization budget, the maximum number of fitting iterations was further increased from 500 to 1500 for two representative configurations, O–hcp and O–fcc. For each configuration, two fitting runs (Fitting 1 and Fitting 3) were examined so that both the optimization trajectory and the spectral response could be compared across iteration budgets.

<p align="center">
<img width="80%" alt="Figure_Optimization_History_100_500_1500" src="https://github.com/user-attachments/assets/20baeb41-a918-41d1-8cf3-a943e62ad0ff" />

**Figure 26.** Optimization trajectories for the 500- and 1500-iteration fittings. Objective values including regularization are plotted as a function of full-stage L-BFGS-B iteration for O–hcp and O–fcc, using Fitting 1 and Fitting 3 as representative cases. Extending the iteration budget from 500 to 1500 further lowers the objective in all tested runs, although the magnitude of the improvement depends on the configuration and fitting run.

The optimization histories show that extending the iteration budget from 500 to 1500 led to a further reduction in the fitting objective in all four tested cases (Figure 26). In O–hcp, Fitting 1 already exhibited a relatively low objective at 500 iterations and changed only modestly afterward, whereas Fitting 3 continued to decrease more visibly over the extended iteration range. A similar tendency was observed for O–fcc: Fitting 1 improved moderately beyond 500 iterations, while Fitting 3 showed a more prolonged decrease toward 1500 iterations. Thus, in terms of the energy–force fitting objective alone, increasing the optimization budget remained beneficial for all four runs.

<p align="center">
<img width="80%" alt="Figure_Optimization_Cost_Gradient_100_500_1500" src="https://github.com/user-attachments/assets/9d56a4e5-5207-4fae-bdca-cf9f7499c769" />

**Figure 27.** Summary of fitting objective and projected gradient as a function of iteration budget. The final fitting objective (left), and the projected gradient norm (right), for O–hcp and O–fcc in Fitting 1 and Fitting 3. While the fitting objective decreases monotonically as the iteration budget increases from 100 to 500 to 1500, the projected gradient does not improve uniformly across all runs.

This tendency is summarized more directly in Figure 27. The final fitting objective decreased monotonically as the full-stage iteration budget increased from 100 to 500 and then to 1500. However, the projected gradient did not decrease uniformly in all cases. In particular, O–hcp Fitting 3 and O–fcc Fitting 3 retained comparatively larger projected gradients even at 1500 iterations, whereas O–hcp Fitting 1 and O–fcc Fitting 1 reached substantially smaller gradient levels. Therefore, although the optimizer continued to reduce the objective function, the degree of local stabilization was not identical across fitting runs.

<p align="center">
<img width="60%" alt="Figure_SameRun_PDOS_W1_500_vs_1500" src="https://github.com/user-attachments/assets/ba0fca9b-5d38-4eb7-b6fb-8f2aa8900540" />

**Figure 28.** Absolute-energy PDOS change between the 500- and 1500-iteration results for the same initialization. Wasserstein-1 distances are shown separately for the O 2p and Pt 5d PDOS components. The O 2p spectrum exhibits greater sensitivity to the additional optimization than Pt 5d, particularly for O–hcp Fitting 1 and O–fcc Fitting 3.

The key question is whether the additional reduction in fitting objective also improved spectral robustness. To test this point under controlled conditions, the PDOS obtained at 500 and 1500 iterations was compared for the **same initialization** using an absolute-energy scale, and the resulting Wasserstein-1 distances are shown in Figure 28. The O 2p PDOS remained noticeably sensitive to the additional optimization, particularly for O–hcp Fitting 1, where the 500-to-1500 change was the largest among the tested cases. O–fcc Fitting 3 also showed a substantial O 2p change, while O–hcp Fitting 3 and O–fcc Fitting 1 were less sensitive. In contrast, the Pt 5d PDOS was consistently more stable than O 2p, although non-negligible changes were still observed, especially for O–fcc Fitting 3. These results indicate that extending the optimization can still alter the orbital-resolved spectrum appreciably, even when the fitting begins from the same parameter initialization.

<p align="center">
<img width="80%" alt="Figure_WithinBudget_W1_100_500_1500" src="https://github.com/user-attachments/assets/77f6edc6-d2de-45c2-8eea-23a85954c019" />

**Figure 29.** Iteration-budget dependence of the spectral difference between Fitting 1 and Fitting 3. Wasserstein-1 distances between independently repeated fittings are shown for O 2p and Pt 5d as a function of the full-stage iteration budget. For both O–hcp and O–fcc, the cross-run discrepancy decreases from 100 to 500 iterations and then increases again at 1500 iterations.

A complementary view is provided by comparing the spectral difference between Fitting 1 and Fitting 3 as a function of iteration budget (Figure 29). For both O–hcp and O–fcc, the O 2p Wasserstein-1 distance decreased substantially when the budget was increased from 100 to 500 iterations, indicating improved cross-run agreement at the intermediate optimization level. However, the same distance increased again at 1500 iterations for both configurations. The Pt 5d trend was qualitatively similar but smaller in magnitude. Therefore, the best agreement between independently repeated fittings was obtained at 500 iterations rather than at 1500 iterations.

Taken together, the 1500-iteration analysis suggests that a larger optimization budget improves the fitting objective but does not guarantee monotonic improvement in spectral reproducibility. In the present cases, the additional optimization beyond 500 iterations continued to refine the energy–force fit, yet the O 2p spectrum remained sensitive both to the optimization path and to the specific fitting run. Accordingly, the remaining refined-model variability is better interpreted as **residual optimization dependence of the Hamiltonian-to-spectrum mapping** rather than simply as a lack of further objective reduction. For the present report, the 500-iteration results may therefore be regarded as the most balanced operating point, because they provide a substantially improved fitting objective relative to 100 iterations while also giving the smallest cross-run spectral discrepancy among the tested budgets.

#### 4.6.4. Local Parameter Sensitivity and Objective Conditioning

The preceding fitting stability analyses revealed that improvements in the energy–force fitting objective did not necessarily correspond to improved reproducibility of the reconstructed electronic spectra. In particular, extending the optimization budget from 500 to 1500 iterations further reduced the fitting cost, while the run-to-run Wasserstein-1 distances increased for both O–hcp and O–fcc. These observations motivated a more direct examination of the local sensitivity and conditioning of the parameter optimization problem.

To investigate this behavior, the 1500-iteration results for Fitting 1 and Fitting 3 of O–hcp and O–fcc were selected for further analysis. The original energy–force residuals and parameter regularization were reconstructed without additional fitting, and the resulting objective values were verified against the saved optimization results. Local parameter sensitivity was evaluated using the residual Jacobian, the corresponding Gauss–Newton (GN) curvature approximation, and a finite-difference approximation to the full objective Hessian. The parameters were scaled by their respective optimization-bound widths to account for differences in their numerical ranges.

The GN curvature analysis revealed a highly nonuniform distribution of parameter sensitivities (Fig. 30). Across the four investigated fitting results, the eigenvalues spanned several orders of magnitude, indicating that the fitting objective responds strongly to certain combinations of parameters while remaining comparatively insensitive to others. The numerical ranks were 24/27 and 25/27 for O–hcp Fitting 1 and Fitting 3, respectively, and 26/27 for both O–fcc results. These values indicate the presence of weakly constrained parameter directions under the adopted numerical rank threshold, rather than exact rank deficiency or mathematically established parameter non-identifiability.

<p align="center">
<img width="80%" alt="FigureA_GN_curvature_eigenvalue_spectra" src="https://github.com/user-attachments/assets/d45428a9-aaab-4950-844f-dc0a05ae69f9" />

**Figure 30.** Gauss–Newton Curvature Eigenvalue Spectra of the Fitted TB Parameters. Eigenvalue distributions of the bound-scaled Gauss–Newton curvature matrices for (a) O–hcp Fitting 1, (b) O–hcp Fitting 3, (c) O–fcc Fitting 1, and (d) O–fcc Fitting 3 at the 1500-iteration optimization limit. The dashed lines indicate the numerical rank thresholds. The broad eigenvalue distributions and relatively small eigenvalues indicate substantial differences in sensitivity among parameter-space directions.

**Table 5.** Summary of Local Objective Conditioning for O–hcp and O–fcc. Numerical rank and extremal Gauss–Newton eigenvalues were calculated for the 27 fitted parameters using bound-scaled coordinates. The final column reports the number of negative eigenvalues obtained from the numerically approximated full objective Hessian.

| Configuration | Fitting | GN Numerical Rank | Maximum GN Eigenvalue | Minimum GN Eigenvalue | Negative Full-Hessian Eigenvalues |
|:-------------|:-------:|:-----------------:|----------------------:|----------------------:|:-------------------------------:|
| O–hcp | 1 | 24/27 | 1.17 × 10⁶ | 5.19 × 10⁻³ | 1 |
| O–hcp | 3 | 25/27 | 3.94 × 10⁶ | 4.51 × 10⁻⁵ | 1 |
| O–fcc | 1 | 26/27 | 1.53 × 10⁶ | 7.25 × 10⁻³ | 0 |
| O–fcc | 3 | 26/27 | 7.66 × 10⁵ | 2.33 × 10⁻⁴ | 2 |

The full objective Hessian additionally exhibited negative eigenvalues in three of the four fitting results. This observation suggests that the local optimization landscape may retain directions of negative curvature at the investigated parameter points. However, since the optimization stopping criteria were not fully satisfied and the Hessian was evaluated numerically, these results should not be interpreted as definitive evidence of distinct local minima.

The parameter-wise residual sensitivity analysis revealed substantial differences in the influence of individual TB parameters on the energy–force residuals (Fig. 31). Strong sensitivities were observed for several Pt–Pt and Pt–O interaction parameters, whereas other parameters contributed much less to the local residual response. Nevertheless, individual parameter sensitivities do not fully characterize the conditioning of the optimization problem, since correlated parameter variations may produce weaker changes in the objective than independent variations along the original parameter axes.

<p align="center">
<img width="80%" alt="FigureB_parameter_sensitivity_jacobian" src="https://github.com/user-attachments/assets/a6a166a2-3c46-4403-878b-1ab8095cb75a" />

**Figure 31.** Parameter-Wise Energy–Force Residual Sensitivity of the Refined TB Model. Bound-scaled Jacobian column norms for the fitted TB parameters evaluated at O–hcp Fitting 1 and 3 and O–fcc Fitting 1 and 3. The logarithmic color scale represents the sensitivity of the energy–force residual vector to each parameter. These values characterize individual parameter sensitivities and should be distinguished from the collective parameter-space directions obtained through Gauss–Newton eigenvalue analysis.

To examine whether the weak GN directions also corresponded to small variations in the actual fitting objective, the parameters were perturbed along the softest and stiffest GN eigenvectors. The resulting objective evaluations exhibited pronounced differences between the two directions (Fig. 32). Perturbations along the softest directions produced only minor changes in the fitting cost, whereas substantially larger increases were observed along the stiffest directions. This behavior was observed for both adsorption configurations and fitting runs, providing direct numerical evidence that the objective is highly anisotropic in parameter space.

<p align="center">
<img width="80%" alt="FigureC_actual_objective_loss_mode_scans" src="https://github.com/user-attachments/assets/de117eae-0580-4079-851e-bcdf05d7c88c" />

**Figure 32.** Actual Objective Variation Along Weakly and Strongly Constrained Parameter Directions. Changes in the reconstructed fitting objective, including regularization, under perturbations along the softest and stiffest Gauss–Newton eigenvectors for (a) O–hcp Fitting 1, (b) O–hcp Fitting 3, (c) O–fcc Fitting 1, and (d) O–fcc Fitting 3. Parameter displacements are expressed in bound-scaled coordinates. The soft directions exhibit comparatively flat objective profiles, whereas the stiff directions show pronounced increases in fitting cost. The permitted displacement ranges differ between directions because of parameter-bound constraints.

The weakest GN eigenvectors were dominated by parameters associated with the distance dependence of Pt–O hopping interactions (Fig. 33). In O–hcp Fitting 1, the largest contributions arose from the Pt–O ssσ and dsσ decay coefficients, whereas the Pt–O ssσ decay coefficient dominated the corresponding direction in Fitting 3. For O–fcc, the Pt–O dsσ decay coefficient was the dominant contribution in both fitting runs.

<p align="center">
<img width="80%" alt="FigureD_weakest_mode_parameter_loadings" src="https://github.com/user-attachments/assets/118e5eb7-0ccc-4cbe-9782-deced411671f" />

**Figure 33.** Dominant Parameter Contributions to the Weakest Gauss–Newton Directions. Signed eigenvector loadings for the weakest GN mode in (a) O–hcp Fitting 1, (b) O–hcp Fitting 3, (c) O–fcc Fitting 1, and (d) O–fcc Fitting 3. The dominant contributions are associated primarily with Pt–O hopping decay coefficients. The overall sign of an eigenvector is arbitrary; consequently, relative magnitudes and parameter combinations, rather than the absolute sign of an individual loading, are relevant to its interpretation.

The prominence of these decay coefficients is consistent with the exponential distance dependence of the Slater–Koster hopping integrals,

$$
V_{\lambda}(R)=V_{\lambda,0}
\exp[-\beta_{\lambda}(R-R_0)]f_c(R).
$$

In this expression, the decay coefficient ($\beta_{\lambda}$) controls the sensitivity of the hopping amplitude to changes in interatomic separation. If the reference configurations sample only a restricted range of relevant bond lengths, certain decay coefficients or combinations of hopping parameters may be insufficiently constrained by the available energy–force information. The present analysis is consistent with this possibility, although the individual contribution of the structural sampling range has not been independently quantified.

Related difficulties have been reported in previous parameter-optimization studies. Wen et al. (2017) investigated weakly constrained parameter combinations in interatomic potential fitting, demonstrating that the sensitivity of a model may vary substantially across parameter-space directions [[12]](#ref12). More generally, Gutenkunst et al. (2007) characterized such highly anisotropic sensitivity landscapes as sloppy parameter sensitivities, in which certain parameter combinations are strongly constrained while others remain weakly determined [[13]](#ref13).

For tight-binding parameterization, Moradi et al. (2025) similarly emphasized the importance of sensitivity-guided optimization and physically motivated parameter constraints for metal–water interactions. Their study reported that improvements in adsorption-energy fitting could be accompanied by nearly degenerate oxygen 2s and 2p atomic energy levels, illustrating that a lower fitting error does not necessarily ensure physically consistent orbital parameters [[11]](#ref11).  Schattauer et al. (2022) further highlighted the non-unique nature of inverse tight-binding parameterization, noting that different parameter sets may reproduce similar electronic observables and that the physical validity of the resulting models should be assessed through their predicted electronic properties [[14]](#ref14). Although these studies employ different Hamiltonian models, reference data, or target properties, they provide relevant context for the parameter-conditioning behavior observed here.

The relationship between weak parameter directions and differences between repeated fitting results was examined further by projecting the Fitting 1–Fitting 3 parameter displacement onto the three weakest GN directions. The resulting projection fractions varied from approximately 14.8% to 61.2%, depending on the adsorption configuration and the reference fitting point. Thus, the difference between independently optimized parameter sets cannot be attributed exclusively to the weakest local directions, and the local sensitivity structure may itself change across parameter space.

Overall, the conditioning analysis establishes that the present energy–force fitting objective contains strongly and weakly constrained combinations of TB parameters, particularly within the Pt–O hopping parameterization. These weak directions provide a plausible explanation for the observed sensitivity of fitted parameters to initialization and optimization history. However, the present results do not demonstrate that these directions alone account for the measured PDOS differences or establish the existence of multiple fully converged solutions.

Together with the preceding stability analyses, these findings demonstrate the importance of evaluating the fitting objective, parameter robustness, and electronic-structure reproducibility as distinct aspects of Hamiltonian reconstruction. They also highlight a fundamental limitation of relying exclusively on atomic energy and force constraints when the intended application requires reliable orbital-resolved electronic structures.

#### 4.6.5. Relationship Between Objective Conditioning and Spectral Sensitivity

The preceding conditioning analysis identified weakly constrained parameter directions dominated primarily by Pt–O hopping decay coefficients. However, weak sensitivity to the fitting objective does not necessarily imply strong sensitivity in the reconstructed electronic structure. To examine this relationship directly, orbital-resolved PDOS variations were evaluated under parameter perturbations along the Gauss–Newton (GN) eigenvector directions.

<p align="center">
<img width="80%" alt="FigureE_soft_mode_PDOS_sensitivity" src="https://github.com/user-attachments/assets/28318fa9-8bf1-41bc-9d15-d8587a14e2fa" />

**Figure 34.** Objective and Spectral Variations Along the Softest Gauss–Newton Directions. Changes in the actual fitting objective (left) and the Wasserstein-1 distances of the O 2p and Pt 5d PDOS (right) under parameter perturbations along the softest GN eigenvector for O–hcp and O–fcc in Fitting 1 and Fitting 3. Parameter displacements are expressed in bound-scaled coordinates, and spectral comparisons are performed on an absolute-energy scale.

Figure 34 presents the changes in the actual fitting objective and the corresponding Wasserstein-1 distances of the O 2p and Pt 5d PDOS under perturbations along the softest GN direction. Although the fitting objective exhibited only minor variations, the resulting spectral changes were also relatively small. In all four investigated fitting results, the O 2p PDOS exhibited greater sensitivity than the Pt 5d PDOS along the softest direction. Nevertheless, the magnitude of these spectral variations was considerably smaller than the discrepancies previously observed between the 500- and 1500-iteration fitting results (Fig. 28), although the corresponding parameter displacements were not directly equivalent. These observations suggest that the softest GN direction alone is insufficient to explain the observed electronic-structure variability.

To extend this analysis beyond the softest direction, spectral sensitivities were evaluated along all 27 GN eigenvector directions. For each mode, the spectral sensitivity was estimated using the Wasserstein-1 distances obtained under positive and negative parameter perturbations:

$$
S_i^{\mathrm{PDOS}} =
\frac{W_1(D_0,D_{i,+})+W_1(D_0,D_{i,-})}
{2\alpha_i}
$$

where $D_0$ denotes the reference PDOS calculated using the fitted parameters, $D_{i,+}$ and $D_{i,-}$ represent the spectra obtained under positive and negative perturbations along the $i$-th GN eigenvector, respectively, and $alpha_i$ denotes the corresponding displacement magnitude in bound-scaled parameter coordinates. The displacement magnitude was reduced where necessary to satisfy the parameter bounds. All spectral comparisons were performed on an absolute-energy scale.

The resulting relationships between GN curvature and spectral sensitivity are presented in Figure 35. Distinct sensitivity distributions were observed for the O 2p and Pt 5d states. The O 2p PDOS exhibited pronounced spectral sensitivities along several directions with relatively low or intermediate GN curvature, whereas the Pt 5d PDOS generally exhibited higher sensitivities along directions with larger GN eigenvalues. Furthermore, neither the softest nor the stiffest GN direction consistently produced the largest spectral response.

<p align="center">
<img width="80%" alt="FigureF_GN_curvature_vs_PDOS_sensitivity" src="https://github.com/user-attachments/assets/0f746377-85e0-43ce-a9e1-a46557452ab8" />

**Figure 35.** Relationship Between Gauss–Newton Curvature and Orbital-Resolved Spectral Sensitivity. Spectral sensitivities of the O 2p (left) and Pt 5d (right) PDOS evaluated along the 27 GN eigenvector directions for O–hcp and O–fcc in Fitting 1 and Fitting 3. Each point represents an individual GN mode, indexed in ascending order of its eigenvalue. Spectral sensitivities are estimated from the Wasserstein-1 distances under symmetric parameter perturbations in bound-scaled coordinates.

These results demonstrate that parameter sensitivity with respect to the energy–force fitting objective does not necessarily correspond to sensitivity in the reconstructed orbital-resolved electronic structures. This discrepancy reflects the different ways in which combinations of tight-binding parameters influence atomic energetics and electronic spectra.

Consequently, objective conditioning alone is insufficient to characterize electronic-structure robustness in the investigated models. The present findings suggest that future Hamiltonian parameterization strategies should consider not only the accuracy and conditioning of the energy–force fitting objective but also the sensitivity of the electronic observables intended for subsequent prediction.

## 5. Conclusion

Although density functional theory (DFT) provides a powerful framework for investigating electronic structures, its computational cost can become prohibitive when large-scale systems, extensive structural configurations, or repeated electronic-structure calculations are required. Recent studies have therefore explored machine-learning-assisted Hamiltonian construction and tight-binding parameterization as alternative approaches to efficient electronic-structure modeling [1–6]. These approaches aim not to eliminate the physical foundations of DFT, but to reduce the computational burden associated with repeated first-principles calculations while retaining relevant electronic information.

Motivated by these developments, the present study investigated whether electronic Hamiltonians could be reconstructed using atomic energies and forces obtained from a pretrained MACE interatomic potential, without explicitly incorporating DFT-derived electronic-structure data into the tight-binding parameterization process. By integrating machine-learning-derived atomic energetics with Slater–Koster tight-binding theory, an automated computational workflow was developed to construct physically parameterized Hamiltonians and calculate orbital-resolved electronic structures for Pt(111) adsorption systems.

The initial reduced-basis model exhibited relatively sharp oxygen-related spectral features compared with the available DFT reference. Subsequent orbital basis expansion, parameter refitting, and reciprocal-space sampling produced configuration-dependent changes in both adsorbate- and metal-projected spectra. Quantitative analyses using the Wasserstein-1 distance and complementary spectral descriptors across 16 adsorption configurations further demonstrated that these changes could not be sufficiently characterized by individual spectral peaks alone. The k-point convergence analysis also highlighted the importance of consistent energy-reference alignment in electronic-structure comparisons.

**A central finding of this study is that improved energy–force fitting performance does not necessarily guarantee improved electronic-structure reproducibility.** Although increasing the optimization limit from 100 to 500 iterations generally reduced spectral discrepancies between independently fitted models, further extending the limit to 1500 iterations increased spectral differences in the selected comparisons despite continued reductions in the fitting objective. These observations demonstrate that a lower fitting cost alone is insufficient to establish the robustness or physical accuracy of the reconstructed electronic structure.

Local parameter sensitivity and conditioning analyses were subsequently performed to investigate this inconsistency. The Gauss–Newton curvature matrices exhibited eigenvalues spanning several orders of magnitude, indicating the presence of strongly and weakly constrained parameter directions. The weakest directions were primarily associated with Pt–O hopping decay coefficients. However, direct spectral sensitivity analysis revealed that these directions did not necessarily produce the largest variations in the reconstructed PDOS. Furthermore, analysis across all Gauss–Newton eigenvector directions revealed distinct spectral sensitivity distributions for O 2p and Pt 5d states. These results indicate that the conditioning of the energy–force fitting objective alone is insufficient to characterize orbital-resolved electronic-structure sensitivity.

Importantly, these findings provide insight into why many existing Hamiltonian learning frameworks explicitly incorporate DFT-derived Hamiltonian matrices, electronic eigenvalues, or orbital-resolved spectral information into their parameterization procedures. Although conventional machine-learning interatomic potentials implicitly capture the energetic consequences of electronic interactions, their energy and force outputs do not explicitly constrain the underlying orbital-resolved electronic structures. Consequently, reconstructing electronic Hamiltonians from such energetic information represents a distinct inverse problem, in which agreement with atomic energies and forces does not necessarily ensure agreement in electronic properties.

While the present study initially explored the possibility of reducing dependence on explicit electronic-structure supervision, the observed discrepancies between energetic and spectral sensitivities instead highlight the importance of the electronic information employed in existing Hamiltonian learning approaches. This does not imply that DFT-derived electronic-structure data are indispensable for all Hamiltonian parameterization strategies, but rather that additional physical constraints or complementary information may be necessary to achieve reliable electronic-structure reconstruction.

Several limitations remain. The investigated fitting runs did not fully satisfy the prescribed optimization convergence criteria, and the observed spectral variability cannot therefore be attributed exclusively to intrinsic parameter non-identifiability. Furthermore, the available DFT comparisons remained primarily qualitative, and the generalizability of the observed sensitivity relationships beyond the investigated Pt(111) adsorption systems has not been established.

Future work should prioritize improved optimization convergence, independent electronic-structure validation, and the development of parameterization strategies that account for the sensitivity of the intended electronic observables. Additional physical constraints, more diverse structural configurations, and sensitivity-guided optimization may provide opportunities to improve electronic-structure robustness while limiting the need for explicit electronic-structure supervision.

Overall, this study demonstrates both the feasibility and limitations of integrating machine-learning-derived atomic energetics with physics-based Hamiltonian construction. Rather than establishing a direct replacement for DFT, the present work provides **insight into the information required for reliable electronic-structure reconstruction** and emphasizes the complementary roles of first-principles calculations, machine-learning interatomic potentials, and physically interpretable Hamiltonian models in the development of efficient and automated computational materials science workflows.

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

<a id="ref11"></a>
[11] Moradi, S., Dabbaghi, P., & Stein, C. J. (2025). Optimizing Extended Tight‐Binding Methods for Metal‐Surface Interactions. ChemPhysChem, 26(23), e202500463.

<a id="ref12"></a>
[12] Wen, M., Li, J., Brommer, P., Elliott, R. S., Sethna, J. P., & Tadmor, E. B. (2017). A KIM-compliant potfit for fitting sloppy interatomic potentials: application to the EDIP model for silicon. Modelling and Simulation in Materials Science and Engineering, 25(1), 014001.

<a id="ref13"></a>
[13] Gutenkunst, R. N., Waterfall, J. J., Casey, F. P., Brown, K. S., Myers, C. R., & Sethna, J. P. (2007). Universally sloppy parameter sensitivities in systems biology models. PLoS computational biology, 3(10), e189.

<a id="ref13"></a>
[14] Schattauer, C., Todorović, M., Ghosh, K., Rinke, P., & Libisch, F. (2022). Machine learning sparse tight-binding parameters for defects. NPJ Computational Materials, 8(1), 116.

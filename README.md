# Problem 1: Measurements of W mass and width (10 + 2\* points)

Determine the mass \( $m_W$ \) and decay width \( $\Gamma_W$ \) of the W boson using simulated data as a replacement for real experimental data. The most popular Monte Carlo event generator that can be used is **Pythia**. You may assume an ideal detector with perfect particle identification, perfect momentum measurement, and perfect background rejection. The focus is on physics reaction and kinematics.

## Workflow and grading criteria

1. **Come up with a plan:**
   - Specify the reaction and kinematics (**2 pts**)
   - Which particles are reconstructed and how the information is processed to obtain the desired measurement (**2 pts**)

2. **Generate a sample with Pythia** (**2\*** pts) [bonus]  
   If needed, ask for help—no penalty for getting assistance with Pythia samples.

3. **Process the data and obtain the values for \( $m_W$ \) and \( $\Gamma_W$ \) with their uncertainties:**
   - \( $m_W$ \) (**2 pts**)
   - \( $\Gamma_W$ \) (**2 pts**)
   - Uncertainties (**1 pt**)

4. **Critically discuss (1 pt):**
   - What limits the precision
   - Which assumptions matter
   - Robustness of your result

## Submission

### Reaction and kinematics

#### Production Process

- **Collision type:** Proton-proton (pp) collisions
- **Center-of-mass energy:** √s = 13 TeV (matching LHC Run 2 conditions)
- **Subprocess:** W boson production via quark-antiquark annihilation
  - W⁺ production: u d̄ → W⁺
  - W⁻ production: d ū → W⁻
- **Both W⁺ and W⁻ are generated** to maximize statistics and match experimental methodology

#### Decay Channel

- **W decay mode:** W → μ ± ν_μ (muonic decay)
- **Branching ratio:** ~10.8% per charge
- **Choice rationale:** 
  - Simple two-body final state (muon + neutrino) suitable for transverse mass analysis
  - Standard channel used in LHC W mass measurements (ATLAS, CMS)
  - Clean signature in ideal detector

#### Reconstruction Method

1. **Particles reconstructed:**
   - **Muon:** Record 4-momentum (p_T, η, φ, E)
   - **Neutrino:** Inferred from missing transverse momentum (MET)
     - In ideal detector, neutrino carries exactly the missing transverse energy
     - Transverse momentum obtained from momentum conservation: $\vec{p_T^\nu} = -\vec{p_T^\mu}$ (in transverse plane)

2. **Kinematic variable for measurement:**
   - **Transverse mass ($m_T$):** The primary observable for W mass and width extraction
   - Definition: $m_T = \sqrt{2 p_T^\mu p_T^\nu (1 - \cos \Delta\phi)}$
     - p_T^μ: muon transverse momentum
     - p_T^ν: neutrino transverse momentum
     - Δφ: azimuthal angle difference between muon and neutrino
   - **Why transverse mass:**
     - Cannot measure neutrino longitudinal momentum → cannot reconstruct full W mass event-by-event
     - $m_T$ distribution peaks near $m_W$ and has shape sensitive to W width
     - Jacobian factor creates characteristic edge structure

3. **Analysis procedure:**
   - Build histogram of $m_T$ distribution from all events
   - Fit with Breit-Wigner resonance to extract:
     - **$m_W$:** Position of the peak
     - **$\Gamma_W$:** Width parameter
   - Extract statistical uncertainties from fit covariance matrix

#### Event Generation Settings

- **Event generator:** Pythia 8
- **Number of events:** 100,000 (balance between statistics and computational time)
- **Pythia configuration:**
  - Parton shower: ON (more realistic physics)
  - Hadronization: ON (full event simulation)
  - Multiple parton interactions: ON (underlying event)
  - Cross-section weighting: included
- **Output format:** JSON file containing muon and neutrino 4-vectors for each event

#### Acceptance and Selection Criteria

- **Ideal detector assumption:** No geometric acceptance cuts or detector efficiency corrections
- **No kinematic cuts applied** (perfect detector means all generated particles detected perfectly)
- Analysis focuses purely on physics kinematics, not detector limitations

#### Assumptions and Limitations

- **Ideal detector:** Perfect particle identification, perfect momentum measurement, perfect background rejection
- **Generator-level analysis:** No consideration of detector smearing or resolution effects
- **Parton-level only:** Initial and final state radiation included via Pythia; photon radiation effects modeled within generator

## Authors

- Fabian Steube
- Colin Beckmann

## License

This project was created as a homework assignment for the Particle Physics course.

## Use of Generative AI

Generative AI tools were used during work on this project.

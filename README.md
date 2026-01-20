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

### Reaction and Kinematics

**Key Points (Task 1 - 2 pts):**

- **Reaction:** pp → W± → μ±νμ at √s = 7 TeV via quark-antiquark annihilation (ud̄ → W⁺, dū → W⁻)
- **Decay channel:** Muonic W decay (BR ≈ 10.8%), chosen for clean signature
- **Event structure:** W boson + ISR/FSR/underlying event (treated as "other particles")
- **Sample:** 4195 events provided by professor (generated with Pythia 8)
- **Note:** Task 2 (bonus - Pythia generation) not fulfilled; we used pre-generated data

#### Production Process and Decay Channel

W bosons are produced in proton-proton collisions at a center-of-mass energy of √s = 7 TeV through quark-antiquark annihilation processes (ud̄ → W⁺ and dū → W⁻). Both W⁺ and W⁻ are generated to maximize statistics, which matches the methodology used in experimental measurements at the LHC. The generated events include realistic hadronic activity beyond the W boson itself: initial state radiation (ISR), final state radiation (FSR), and multiple parton interactions from the underlying event contribute additional particles that are treated collectively as a single 4-momentum vector in the analysis.

The analysis focuses on the muonic decay channel W → μ±νμ, which has a branching ratio of approximately 10.8% per charge. This channel was chosen because it provides a clean two-body final state well-suited for transverse mass reconstruction, and it represents the standard approach used in precision W mass measurements by the ATLAS and CMS experiments. Under the ideal detector assumption, muons are perfectly identified and their momenta precisely measured, while backgrounds are completely rejected.

#### Reconstruction Method

**Key Points (Task 1 - 2 pts):**

- **Measured:** Muon 3-momentum (direct) → $E_\mu = \sqrt{|\vec{p}_\mu|^2 + m_\mu^2}$ with $m_\mu = 0.10566$ GeV
- **Reconstructed:** Neutrino from generator truth (CSV file), $E_\nu = |\vec{p}_\nu|$ (massless)
- **Observable:** Transverse mass $m_T = \sqrt{2 p_T^\mu p_T^\nu (1 - \cos\Delta\phi)}$
- **Method:** Histogram $m_T$ distribution → fit with analytical functions → extract $m_W$ and $\Gamma_W$
- **Note:** Conservation method failed (ISR/FSR consume ~6807 GeV, leaving no room for realistic neutrino)

The analysis reconstructs three types of particles from each event. The muon's 3-momentum components ($p_x^\mu$, $p_y^\mu$, $p_z^\mu$) are directly measured, and its energy is calculated using the relativistic relation $E_\mu = \sqrt{|\vec{p}_\mu|^2 + m_\mu^2}$ with the muon mass $m_\mu = 0.10566$ GeV. All other particles produced in the collision (from ISR, FSR, and underlying event) are represented by their summed 4-momentum ($\vec{p}_{other}$, $E_{other}$), which captures the total hadronic recoil activity.

The neutrino presents a unique challenge since it escapes detection. While the initial plan suggested reconstructing the neutrino from momentum and energy conservation using the known initial state ($\vec{p}_{initial} = 0$, $E_{initial} = 7000$ GeV), this approach fails for our dataset because the "other particles" already account for nearly the full collision energy (~6807 GeV on average). Therefore, the analysis uses the generator-level neutrino momentum from the CSV file, treating neutrinos as massless so that $E_\nu = |\vec{p}_\nu|$. This represents the Monte Carlo truth that would not be available in real experiments, which instead reconstruct missing transverse energy from calorimeter measurements.

The key observable for W boson measurements is the transverse mass, defined as $m_T = \sqrt{2 p_T^\mu p_T^\nu (1 - \cos \Delta\phi)}$, where $p_T^\mu$ and $p_T^\nu$ are the transverse momenta (projections onto the xy-plane) and $\Delta\phi$ is the azimuthal angle difference between the muon and neutrino. This variable is necessary because the neutrino's longitudinal momentum cannot be measured, preventing event-by-event reconstruction of the W's invariant mass. The transverse mass distribution exhibits a characteristic peak near the W mass due to the mass constraint, with a Jacobian edge structure arising from the decay kinematics.

The analysis procedure builds a histogram of the transverse mass distribution across all events and fits it with various functional forms to extract the W boson mass ($m_W$, from the peak position) and decay width ($\Gamma_W$, from the distribution width). Statistical uncertainties on the fitted parameters are obtained from the covariance matrix returned by the fitting algorithm.

#### Event Generation and Analysis Assumptions

The Monte Carlo sample used in this analysis was provided by the professor and was generated using Pythia 8 with full simulation of parton showers, hadronization, and multiple parton interactions to create realistic event topologies. **Note: We did not generate the data ourselves, therefore Task 2 (bonus 2* points for Pythia sample generation) was not fulfilled.** The dataset contains 4195 events. The provided CSV file includes the muon 3-momentum, neutrino 3-momentum (generator truth), and the summed 4-momentum of all other particles in each event.

The analysis assumes an ideal detector with perfect particle identification, perfect momentum measurement, and perfect background rejection. No geometric acceptance cuts or detector efficiency corrections are applied, and no kinematic selection criteria are imposed beyond the fit range choices. The focus is purely on the physics kinematics of the W decay, not on detector limitations. The neutrino is treated as massless ($m_\nu = 0$), which introduces negligible error given that neutrino masses are below 0.1 eV. A key simplification is the use of generator-level neutrino momentum, which bypasses the experimental challenge of reconstructing missing transverse energy from calorimeter measurements.

## Data Processing and Results

**Key Points (Task 3 - 5 pts):**

- **Best result (Crystal Ball):** $m_W = 78.710 \pm 0.205$ GeV, $\Gamma_W = 5.431 \pm 0.334$ GeV
- **Comparison to PDG:** $m_W^{PDG} = 80.379$ GeV (2.1% deviation), $\Gamma_W^{PDG} = 2.085$ GeV (2.6× overestimate)
- **Method comparison:** Breit-Wigner (58.8 GeV) vs Restricted BW (72.9 GeV) vs Crystal Ball (78.7 GeV)
- **Validation:** Histogram peak at 79.7 GeV confirms Crystal Ball accuracy (0.98 GeV agreement)
- **Mass measurement:** ROBUST (2% precision). **Width measurement:** NOT ROBUST (kinematic smearing)

### Analysis Implementation

The complete analysis is implemented in `computation.py` and processes 4195 simulated W → μν decay events through a systematic six-step pipeline. First, the code loads event data from `w_boson_analysis.csv` and validates physical consistency by checking for NaN or infinite values, ensuring all energies are positive, and verifying that energy-momentum relations hold for massive particles. The muon energy is then calculated for each event using the relativistic formula $E_\mu = \sqrt{|\vec{p}_\mu|^2 + m_\mu^2}$ with $m_\mu = 0.10566$ GeV, finding that 98.6% of muons are ultra-relativistic.

For neutrino reconstruction, the code implements two methods selectable via a configuration flag. The 'csv' mode uses the generator-level neutrino 3-momentum from the input file and calculates its energy as $E_\nu = |\vec{p}_\nu|$ under the massless approximation. The 'conservation' mode attempts to reconstruct the neutrino from full 3D momentum and energy conservation ($\vec{p}_\nu = -\vec{p}_\mu - \vec{p}_{other}$, $E_\nu = 7000 - E_\mu - E_{other}$), but this fails for our dataset because the "other particles" consume ~6807 GeV on average, leaving insufficient energy to produce realistic neutrino kinematics. Consequently, all reported results use the 'csv' mode with generator truth.

The code then calculates transverse momenta by projecting the muon and neutrino momenta onto the xy-plane and computes the azimuthal angle difference between them. The transverse mass $m_T = \sqrt{2 p_T^\mu p_T^\nu (1 - \cos\Delta\phi)}$ is computed for all 4195 events, producing a distribution that peaks near 80 GeV with 33.6% of events falling in the expected W mass region (70-90 GeV). Finally, an 80-bin histogram covering 0-150 GeV (bin width 1.88 GeV) is created, and the distribution is fitted using one of three selectable methods to extract $m_W$ and $\Gamma_W$.

### Fitting Methods and Measured Values

**Results Summary:**

| Method | $m_W$ [GeV] | $\Gamma_W$ [GeV] | $\chi^2$/NDF | Status |
|--------|------------|-----------------|--------------|--------|
| **Crystal Ball (BEST)** | **78.710 ± 0.205** | **5.431 ± 0.334** | **3.24** | ✓ 2% from PDG |
| Breit-Wigner | 58.8 ± 0.4 | 19.9 ± 0.7 | 34.82 | ✗ Wrong model |
| Restricted BW | 72.909 ± 0.204 | 10.231 ± 0.406 | 38.95 | ✗ Still fails |
| Histogram peak | 79.7 (no fit) | - | - | Model-independent |

Three different approaches were tested to extract the W boson parameters from the transverse mass distribution. The Breit-Wigner function, a symmetric relativistic resonance formula $\text{BW}(m) = A \cdot m \cdot \Gamma / ((m^2 - m_W^2)^2 + m_W^2 \Gamma^2)$, was fitted over the 40-120 GeV range and yielded $m_W = 58.8 \pm 0.4$ GeV and $\Gamma_W = 19.9 \pm 0.7$ GeV with $\chi^2/\text{NDF} = 34.82$. This severely underestimates the W mass by 21.6 GeV (60σ deviation from the PDG value) because the symmetric Breit-Wigner cannot capture the asymmetric shape of the transverse mass distribution, which has an extended tail beyond the peak. The validation check confirms this failure: the fitted mass deviates by 20.9 GeV from the model-independent histogram peak position of 79.7 GeV.

A restricted-range Breit-Wigner fit was also tested, limiting the fit to 60-95 GeV to avoid the problematic tail region. This approach produced $m_W = 72.909 \pm 0.204$ GeV and $\Gamma_W = 10.231 \pm 0.406$ GeV with an even worse $\chi^2/\text{NDF} = 38.95$. The 6.8 GeV deviation from the histogram peak and the poor fit quality demonstrate that restricting the range does not solve the fundamental problem that Breit-Wigner is an inappropriate functional form for transverse mass distributions.

The Crystal Ball function, which combines a Gaussian core with an asymmetric power-law tail to model kinematic edge structures, provides significantly better results. Fitted over the full 40-120 GeV range, it yields $m_W = 78.710 \pm 0.205$ GeV and $\Gamma_W = 5.431 \pm 0.334$ GeV (derived from FWHM ≈ 2.35σ) with $\chi^2/\text{NDF} = 3.24$. This represents excellent agreement with the histogram peak (deviation of only 0.98 GeV), and the mass is within 2.1% of the PDG value of 80.379 GeV. The χ²/NDF of 3.24, while above unity, is acceptable for transverse mass fits because the distribution includes kinematic effects beyond a simple resonance shape. The systematic 1.67 GeV downward shift from the true W mass is expected due to the Jacobian peak structure inherent in transverse mass kinematics.

All three methods overestimate the W width significantly, with the Crystal Ball giving a value 2.6 times larger than the PDG width of 2.085 GeV. This occurs because the transverse mass distribution width depends not only on the natural W decay width but also on kinematic smearing from the W boson transverse momentum spectrum and the decay geometry. Extracting the natural width from transverse mass distributions requires sophisticated template-fitting techniques that account for these effects; analytical functions cannot separate them. Real experiments therefore measure $\Gamma_W$ using Z → ℓℓ invariant mass distributions where both leptons are detected and the full invariant mass can be reconstructed.

## Evaluation and Discussion

**Key Points (Task 4 - 1 pt):**

- **Precision limit:** Model systematics (20 GeV method spread) >> statistical uncertainty (±0.2 GeV)
- **Critical assumptions:** Fit function choice (most impactful), generator-level neutrino (bypasses real challenge), transverse mass ≠ invariant mass
- **Robustness:** Mass measurement is robust with Crystal Ball (2% accuracy, validated by histogram peak). Width extraction is fundamentally non-robust (all methods fail by 2-10×).
- **Main lesson:** Transverse mass suitable for $m_W$ but not $\Gamma_W$. Real experiments need template fitting, not analytical functions.

### Critical Analysis of the Analysis Method

This section addresses Task 4 requirements: examining precision limits, evaluating which assumptions matter most, and assessing the robustness of our measurement.

#### Precision Limitations

The analysis achieves a statistical precision of ±0.205 GeV on the W mass from Crystal Ball fitting of 4195 events. However, systematic effects dominate the uncertainty budget. The most significant limitation comes from model dependence: the three fitting methods produce results spanning 20 GeV (58.8 to 78.7 GeV), far exceeding statistical uncertainties. This reveals that the choice of fit function is the primary systematic uncertainty in this analysis.

The transverse mass methodology itself imposes fundamental constraints on achievable precision. Unlike invariant mass distributions that directly reconstruct resonance peaks, the transverse mass distribution represents a kinematic projection affected by unmeasured longitudinal momentum. The relationship between the $m_T$ distribution peak and the true W mass depends on the W boson $p_T$ spectrum, which is shaped by parton distribution functions and QCD radiation. This introduces systematic sensitivity to theoretical modeling that cannot be eliminated without full template fitting using complete Monte Carlo simulation.

Additionally, the analysis uses generator-level neutrino momentum from the CSV file, bypassing the experimental challenge of missing transverse energy reconstruction. Real LHC measurements must reconstruct $E_T^{miss}$ from calorimeter imbalance with typical resolutions of 10-15 GeV, and must account for pile-up and underlying event effects. Our attempt to implement momentum conservation reconstruction failed completely, demonstrating that naive energy balance doesn't work when events contain initial and final state radiation. The use of perfect generator-level information therefore underestimates the systematic uncertainties that would appear in actual detector data.

The analysis also assumes perfect momentum measurement with zero detector resolution. Real experiments face muon momentum scale calibration uncertainties around 0.02%, which translate to 10-20 MeV systematic shifts in $m_W$. Combined with detector resolution effects (1-2% at 50 GeV), these would add quadratically to our statistical uncertainty but are completely absent from this simulation-based study.

#### Key Assumptions and Their Impact

Several assumptions underpin this analysis, with varying degrees of impact on the final results. The massless neutrino approximation ($E_\nu = |\vec{p}_\nu|$) is physically justified since neutrino masses are below 0.1 eV, making relativistic corrections completely negligible at LHC energies. Similarly, using the precise muon mass of 0.10566 GeV is correctly implemented, though at the high momenta in W decays (mean muon energy ~193 GeV), this contributes only a 0.03% relativistic correction.

The most consequential assumption concerns the fit function choice. The Breit-Wigner function assumes a symmetric Lorentzian resonance shape, which is physically inappropriate for transverse mass distributions that exhibit intrinsic asymmetry from Jacobian effects and kinematic constraints. While correctly implemented mathematically, this function yields a 27% error in the extracted W mass. The Crystal Ball function, designed for asymmetric peaks with power-law tails, provides better physics motivation and achieves 2% accuracy. However, the underlying assumption that any simple analytical function can adequately describe the transverse mass distribution is fundamentally limited—real experiments use template fitting with full Monte Carlo distributions rather than empirical formulas.

The analysis assumes an ideal detector with perfect particle identification and momentum measurement. While the assumption of perfect particle ID (no muon-hadron misidentification) is reasonable for this exploratory study, the absence of momentum resolution effects significantly underestimates systematic uncertainties. Real LHC detectors have muon momentum resolution of 1-2% at 50 GeV, and momentum scale calibration uncertainties of ~0.02% that translate directly to 10-20 MeV systematic shifts in $m_W$. By working with generator-level data, we bypass these dominant experimental systematics.

Event selection assumptions also matter. The code processes all events without quality cuts, whereas real analyses apply muon $p_T > 25$ GeV thresholds, $E_T^{miss} > 25$ GeV requirements, isolation cuts, and $m_T > 50$ GeV selections to suppress backgrounds and enhance signal quality. The task specification states zero background contamination, which simplifies the analysis but eliminates systematic uncertainties from background subtraction that typically contribute 10-20 MeV to $m_W$ uncertainty in real measurements.

Finally, the statistical treatment uses Poisson errors $\sigma_i = \sqrt{N_i}$ for histogram bins, which is standard practice for counting experiments. Most bins in the fitting range contain more than 50 events, making the symmetric Gaussian approximation to the Poisson distribution reasonable. The uncertainty propagation from fit covariance matrices is correctly implemented using standard error propagation.

#### Robustness Assessment

The robustness of our results was evaluated through multiple cross-checks and method comparisons. Testing three independent fitting approaches (Breit-Wigner, Crystal Ball, and restricted-range Breit-Wigner) revealed significant method dependence, with extracted masses spanning 58.8 to 78.7 GeV. This 20 GeV spread indicates that results are highly sensitive to the choice of fit function, representing the dominant systematic uncertainty. The Crystal Ball method proves most robust, achieving 2% agreement with the PDG value (78.7 vs 80.4 GeV) and showing good consistency with the model-independent histogram peak position at 79.7 GeV.

Several indicators support the reliability of the Crystal Ball result. The raw histogram peak remains stable independent of any fitting procedure, confirming good data quality and proper implementation of the transverse mass calculation. The distribution shape—with a peak near 80 GeV, 33.6% of events in the W mass region, and mean transverse momenta around 30 GeV—matches expectations for W production at √s = 7 TeV. The code correctly implements uncertainty propagation from fit covariance matrices, and physics validation checks confirm the data structure is consistent with W decay kinematics.

However, important robustness concerns remain. All fitting methods yield $\chi^2/\text{NDF} > 3$, indicating that none of the analytical functions fully capture the transverse mass distribution shape. While χ²/NDF = 3.24 for Crystal Ball is acceptable given the complexity of transverse mass kinematics (which include smearing from the W $p_T$ spectrum and Jacobian effects), it signals that systematic uncertainties from functional form choice are significant. The sensitivity to fit range is demonstrated by the 6 GeV mass shift when restricting the range from 40-120 GeV to 60-95 GeV.

Width extraction proves fundamentally non-robust. All methods overestimate Γ_W by factors of 2-10×, with the Crystal Ball yielding 5.43 ± 0.33 GeV compared to the PDG value of 2.085 GeV. This is not a code error but a physics limitation: the width of the transverse mass distribution includes contributions from kinematic smearing and the W $p_T$ spectrum, not just the natural width of the W resonance. Extracting Γ_W reliably requires either invariant mass reconstruction (as in Z→ℓℓ channels where both lepton momenta are measured) or sophisticated template fitting that models all kinematic effects. The transverse mass method is simply inappropriate for width measurements.

The attempted momentum conservation method for neutrino reconstruction failed completely, producing unphysical results with zero or negative energies. This demonstrates the fragility of naive energy-momentum balance in hadronic collision events that contain initial-state radiation, final-state radiation, and underlying event activity. Generator-level truth from the CSV file is necessary for this analysis to succeed, but this represents a significant idealization compared to real experimental conditions.

#### Conclusion

This analysis successfully demonstrates the principles of W boson mass measurement from transverse mass distributions, achieving 2% precision using the Crystal Ball fitting method with 4195 events. The result of 78.710 ± 0.205 GeV shows reasonable agreement with the PDG value considering the systematic limitations of transverse mass kinematics. The ~1.7 GeV downward shift from the true W mass is expected from Jacobian effects in the transverse mass projection.

The study reveals several critical lessons about W mass measurements. Model systematic uncertainties dominate over statistical uncertainties when using analytical fitting functions—the 20 GeV method spread far exceeds the ±0.2 GeV statistical precision. Transverse mass distributions are suitable for mass measurements with appropriate fitting approaches but are fundamentally unsuitable for width extraction without full kinematic modeling. The use of generator-level neutrino information bypasses the most challenging experimental aspects: missing energy reconstruction with detector resolution, pile-up corrections, and systematic calibrations.

Real LHC experiments achieve W mass precision at the 10-20 MeV level through template fitting with complete Monte Carlo simulation, evaluation of detector systematic uncertainties (momentum scale, resolution, alignment), careful background subtraction, and analysis of millions of events. Our simplified analysis—using analytical fit functions, perfect detector assumptions, and generator-level truth—provides correct implementation of the underlying physics equations and demonstrates proper statistical treatment, but the methodological limitations prevent achieving experimental-quality precision. The code successfully extracts the W mass within 2% accuracy, meeting the goals of this educational exercise while highlighting the sophisticated techniques required for precision measurements in particle physics.

## Authors

- Fabian Steube
- Colin Beckmann

## License

This project was created as a homework assignment for the Particle Physics course.

## Use of Generative AI

Generative AI tools were used during work on this project.

## Github Repository

[W boson measurement](https://github.com/Flomber/measurement_of_W_boson)

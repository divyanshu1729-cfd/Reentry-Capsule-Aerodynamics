# Re-entry Capsule Aerodynamics
### Analytical Modelling, Compressible Flow Simulation, and Aerodynamic Characterisation

An undergraduate computational aerodynamics project investigating high-speed flow around a re-entry capsule configuration. The study combines classical ballistic-entry theory, numerical implementation in Python, and compressible-flow simulations using OpenFOAM.

**Project status:** Completed
**CFD framework:** OpenFOAM Foundation v13  
**Programming:** Python  
**Initial computational configuration:** Axisymmetric D-5 / P-2B  
**Principal references:** NACA Report 1381 and NASA Technical Note D-3088

---

## 1. Project Overview

Atmospheric entry presents a challenging compressible-flow problem involving substantial aerodynamic loading, strong pressure gradients, shock-wave formation, and significant changes in the thermodynamic state of the surrounding flow.

The objective of this project is to investigate the aerodynamic behaviour of a re-entry capsule through a combination of analytical modelling and computational fluid dynamics (CFD).

The work begins with the classical ballistic-entry formulation associated with Allen and Eggers, documented in NACA Report 1381. This provides the theoretical foundation for understanding the aerodynamic deceleration of a body entering an atmosphere at high velocity.

The investigation subsequently develops an axisymmetric computational model of the D-5 / P-2B configuration, using published NASA reference material as the basis for numerical comparison.

Rather than treating CFD as a black-box simulation, the project emphasises the mathematical formulation, governing conservation equations, numerical methodology, mesh quality, and physical interpretation of the computed flow field.

## 2. Objectives

The principal objectives are:

- Study the theoretical foundations of ballistic atmospheric entry using the Allen–Eggers formulation.
- Implement the relevant analytical relations in Python and examine their numerical behaviour.
- Develop a geometrically consistent re-entry capsule configuration using CAD.
- Construct an axisymmetric computational domain and mesh for the D-5 / P-2B configuration.
- Investigate high-speed compressible flow using OpenFOAM Foundation v13.
- Examine pressure distributions, aerodynamic forces, shock structures, and associated flow-field characteristics.
- Establish a systematic verification and validation procedure using published reference data.

## 3. Theoretical Foundation

### 3.1 Ballistic-entry mechanics

The initial analytical investigation is based on the treatment of ballistic atmospheric entry presented by Allen and Eggers in NACA Report 1381.

The central physical problem concerns the interaction between aerodynamic forces, atmospheric properties, and the velocity of an entering body.

For a simplified ballistic-body formulation, the aerodynamic drag is expressed as

\[
D = \frac{1}{2}\rho V^2 C_D A_{\mathrm{ref}}
\]

where:

- \(\rho\) is the local atmospheric density,
- \(V\) is the vehicle velocity relative to the atmosphere,
- \(C_D\) is the drag coefficient,
- \(A_{\mathrm{ref}}\) is the reference area.

The corresponding aerodynamic deceleration is

\[
a_D = \frac{D}{m}
\]

where \(m\) denotes the vehicle mass.

These relations provide the starting point for understanding the coupling between atmospheric density, velocity, and aerodynamic deceleration.

The Python implementation will be used to reproduce selected analytical results and investigate the sensitivity of the predicted quantities to the governing parameters and modelling assumptions.

### 3.2 Governing equations for CFD

The computational investigation is based on the conservation equations of compressible fluid dynamics.

**Continuity equation**

\[
\frac{\partial \rho}{\partial t}
+
\nabla\cdot(\rho\mathbf{u})=0
\]

**Momentum conservation**

\[
\frac{\partial(\rho\mathbf{u})}{\partial t}
+
\nabla\cdot(\rho\mathbf{u}\otimes\mathbf{u})
=
-\nabla p+\nabla\cdot\boldsymbol{\tau}
\]

where \(\boldsymbol{\tau}\) denotes the viscous stress tensor.

**Total energy conservation**

\[
\frac{\partial(\rho E)}{\partial t}
+
\nabla\cdot[(\rho E+p)\mathbf{u}]
=
\nabla\cdot(\boldsymbol{\tau}\cdot\mathbf{u})
-
\nabla\cdot\mathbf{q}
\]

where \(E\) represents the total specific energy and \(\mathbf{q}\) is the heat-flux vector.

The thermodynamic closure, transport-property models, and numerical treatment of these equations must be documented for each computational configuration.

## 4. Geometry and Computational Configuration

The initial CFD investigation considers an axisymmetric D-5 / P-2B configuration based on the NASA reference material.

The geometry is being developed using FreeCAD, with particular attention to the capsule profile, nose curvature, body transitions, and consistency between the CAD representation and the computational domain.

Preliminary geometry development has included a rounded nose and biconic-style body configuration. The final geometric dimensions and reference locations will be documented against the exported CAD model.

### Computational domain

The initial investigation is axisymmetric.

The computational configuration is intended to resolve:

- The bow-shock structure ahead of the capsule.
- The pressure distribution along the body surface.
- Flow acceleration and deceleration around the capsule.
- Boundary-layer development and associated viscous effects.
- The downstream flow field and wake.

A subsequent three-dimensional extension may be considered after the axisymmetric model has been adequately verified and compared with reference data.

## 5. Computational Methodology

### 5.1 Analytical implementation

Python is being used to implement the relevant ballistic-entry relations and perform numerical comparisons with published analytical results.

The validation workflow will include:

- Implementation of the governing relations.
- Consistent treatment of physical units and atmospheric parameters.
- Comparison against selected reference values.
- Quantification of numerical discrepancies.
- Visualisation of the resulting aerodynamic and trajectory-related quantities.

### 5.2 Mesh generation

The computational mesh is developed for an axisymmetric representation of the capsule.

Mesh quality is assessed using OpenFOAM's `checkMesh` utility, with attention to cell geometry, non-orthogonality, skewness, and other relevant mesh-quality indicators.

The current medium-resolution case has a working target of approximately 200,000 cells.

A successful mesh-quality assessment is treated as a prerequisite for simulation, not as proof of mesh independence.

### 5.3 OpenFOAM simulations

The CFD investigation uses OpenFOAM Foundation v13.

The initial computational cases are being developed for the D-5 / P-2B axisymmetric configuration.

The exploratory Mach-number range considered during the initial setup is approximately

\[
3 \leq M_\infty \leq 5
\]

The final operating conditions and numerical configuration will be reported using the actual case dictionaries.

Particular attention will be given to:

- Compressible-flow formulation.
- Thermophysical and transport-property models.
- Boundary conditions.
- Spatial discretisation schemes.
- Numerical convergence.
- Surface pressure and aerodynamic-force evaluation.

## 6. Verification and Validation

An important distinction is maintained between numerical verification and physical validation.

**Verification** addresses whether the governing equations have been implemented and solved consistently within the numerical framework.

The verification process will consider:

- Mesh-resolution sensitivity.
- Iterative convergence.
- Numerical discretisation.
- Conservation behaviour.
- Sensitivity to relevant computational settings.

**Validation** addresses whether the computational predictions adequately represent the physical behaviour documented by independent reference data.

The principal published reference material for the initial comparison is NASA Technical Note D-3088, associated with the D-5 / P-2B investigation.

Where comparable reference quantities are available, the analysis will examine:

- Surface-pressure distributions.
- Aerodynamic force coefficients.
- Shock-wave characteristics.
- Relevant flow-field quantities.

The reference geometry, freestream conditions, thermophysical assumptions, and definitions of measured quantities must be consistent before a quantitative comparison is made.

## 7. Repository Structure

The repository is organised to separate analytical implementation, geometry, numerical simulation, and reference material.

The repository will progressively incorporate:

- Python analytical implementations and validation plots.
- FreeCAD geometry and associated exports.
- OpenFOAM case dictionaries and mesh-generation files.
- Numerical convergence histories and post-processing scripts.
- Comparison plots and relevant reference-data documentation.

Large transient simulation outputs will be managed separately from the essential reproducibility files.

## 8. Limitations and Future Scope

The present investigation is an evolving computational study and should not be interpreted as a complete model of atmospheric re-entry physics.

The initial modelling framework does not automatically account for:

- High-temperature real-gas effects.
- Chemical nonequilibrium.
- Radiation and radiative heating.
- Material ablation.
- Conjugate heat transfer.
- Fully coupled vehicle trajectory and aerodynamic heating.

These effects require additional physical models and independent validation before being included in any quantitative conclusions.

### Future directions

Following completion of the initial axisymmetric investigation, potential extensions include:

1. More systematic mesh-convergence and numerical-sensitivity studies.
2. Detailed analysis of shock and boundary-layer interactions.
3. Surface heat-transfer analysis, subject to an appropriately developed thermal model.
4. Three-dimensional flow investigation.
5. Comparison of additional computational predictions against independent experimental or numerical benchmarks.

## 9. References

1. Allen, H. J., and Eggers, A. J. NACA Report 1381. The ballistic-entry formulation and associated aerodynamic analysis.

2. NASA Technical Note D-3088. D-5 / P-2B reference material for the computational comparison.

Complete bibliographic information, relevant figures, tables, page numbers, and the exact reference conditions used will be maintained in the `references/` directory.

---

## Project Note

This repository documents an undergraduate computational aerodynamics investigation with an emphasis on physical interpretation, mathematical formulation, numerical methodology, and reproducibility.

The aim is not merely to obtain a converged CFD solution, but to understand the underlying fluid mechanics and establish the extent to which the numerical predictions are supported by independent reference evidence.

# BEM_2D — Boundary Element Method for Earthquake Cycle Simulations

**BEM_2D** is a Python-based research code that models the **earthquake cycle** using the **Boundary Element Method (BEM)** with **rate-and-state friction laws**.  
The solver supports planar fault geometries and spring-slider analog systems, providing a fast and flexible framework for exploring fault mechanics, nucleation, and stress interactions across seismic cycles.

## Overview

The **Boundary Element Method (BEM)** is widely used in computational seismology for modeling quasi-dynamic fault slip.  
BEM_2D implements 2D elastostatic boundary integral equations with rate-and-state frictional constitutive laws, allowing users to explore:

- Nucleation of seismic events  
- Quasi-static and dynamic slip evolution  
- Earthquake recurrence intervals  
- Stress transfer and afterslip  
- Simplified spring-slider analogs

The code is entirely written in **Python**, with performance-critical kernels accelerated using **Numba** for Just-In-Time (JIT) compilation.

---

## Key Features

- **Boundary Element Solver:** Fast computation of tractions and displacements for 2D elastic media.
- **Numba Acceleration:** Efficient JIT-compiled numerical kernels.
- **Rate-and-State Friction:** Includes both aging and slip laws.
- **Spring-Slider Systems:** One- and two-block configurations for conceptual studies.
- **Visualization:** Built-in tools for slip rate, stress, slip profiles, and animations.
- **Industrial Activities** 3D Boussinesq's solution provides an  investigation of anthropogenic effects.

---

## Installation

Clone and install the package using:

```bash
git clone https://github.com/yourusername/BEM_2D.git
cd BEM_2D
pip install -r requirements.txt

## References

For citation
Sopacı, E., & Özacar, A. A. (2025). Simulation of large earthquake synchronization and implications on North Anatolian fault zone. Tectonophysics, 914, 230902. https://doi.org/10.1016/j.tecto.2025.230902

# THE CODE DEVELOPMENT HAS BEEN ONGOING!

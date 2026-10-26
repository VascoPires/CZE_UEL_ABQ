# UEL interface

The **Fortran source** defines the current interface. Some presentation slides use an older interface with separate shear strengths; use this table when preparing a new model.

| Position | Symbol | Meaning |
| --- | --- | --- |
| `PROPS(1)` | GIc | Mode I fracture energy |
| `PROPS(2)` | GIIc | Shear fracture energy, shared by modes II and III |
| `PROPS(3)` | tau3c | Normal strength |
| `PROPS(4)` | tau_sh | Shear strength |
| `PROPS(5)` | eta | Benzeggagh–Kenane exponent |
| `PROPS(6)` | K3 | Normal penalty stiffness |
| `PROPS(7)` | rho | Viscous regularization time scale; zero disables it |
| `PROPS(8)` | zeta | Optional numerical tangent perturbation, in separation/length units; finite and strictly positive |
| `JPROPS(1)` | integration | `1`: Gauss; `2`: Newton–Cotes |

For example, using a consistent N–mm–MPa unit system (fracture energy in N/mm, penalty stiffness in N/mm³):

```text
*USER ELEMENT, TYPE=U8, NODES=8, COORDINATES=3, PROPERTIES=7, IPROPERTIES=1, VARIABLES=200, UNSYMM
1, 2, 3
*UEL PROPERTY, ELSET=INTERFACE
0.187, 0.781, 38., 52., 1.22, 1.E5, 0., 2
```

The last entry is the integer integration property, following the seven real properties. `UNSYMM` is required to retain the full mixed-mode tangent. The default generator configurations and basic element tests use `rho = 0`, matching the rate-independent Turon 2018 law. Positive `rho` enables an optional delayed-damage extension; select it explicitly and check time-step sensitivity. Follow the face pairing and node ordering in the element decks: nodes 1–4 form one face and 5–8 the corresponding opposite face. The initial surfaces may coincide. Keep a consistent face orientation so opening has the intended sign.

See [output fields and energy conventions](../visualization/README.md#state-variables-and-output-files).

Version 1.4 accepts an optional eighth real property. Seven-property decks retain
the original `zeta = 1.E-6` default in the model's length units. To select
`1.E-8` in an N–mm–MPa model:

```text
*USER ELEMENT, TYPE=U8, NODES=8, COORDINATES=3, PROPERTIES=8, IPROPERTIES=1, VARIABLES=200, UNSYMM
1, 2, 3
*UEL PROPERTY, ELSET=INTERFACE
0.187, 0.781, 38., 52., 1.22, 1.E5, 0., 1.E-8
2
```

The first line contains all eight **real** properties; the second contains the
integer integration scheme. Do not leave the scheme in position eight when
declaring eight real properties. Abaqus reads reals before integers and uses
[eight values per data line](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEKEYRefMap/simakey-r-uelproperty.htm).

The preparation runner checks a uniform real-property count across user-element
declarations in a deck. To vary the perturbation between interfaces, use eight
real properties throughout and set each interface's value in its own property block.

For the converter and DCB/MMB launchers, add this optional JSON field:

```json
"cohesive_UEL": {
  "tangent_perturbation": 1e-8
}
```

Add it to the existing `cohesive_UEL` object alongside the material and integration
settings. Omitting it keeps the seven-property format. A complete input example is
[opening_tangent.inp](../examples/element/inputs/opening_tangent.inp).
The validation builders also accept `--tangent-perturbation 1e-8`.

`zeta` changes numerical differentiation of the traction response. It does not
change penalty stiffness, loading increments, viscosity or the damage law.
The tested `1.E-8` value improved particular mm-based cases; smaller values are not
automatically more accurate because rounding and branch transitions also matter.
Nonpositive or nonfinite supplied values are rejected. The known geometric
energy-consistency limitations remain; see the [feature checks](../validation/tangent_input_20261026/README.md).

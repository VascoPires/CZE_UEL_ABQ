# Benchmark references for the UEL (read 2026-10-25)

PDFs in `references/`. Values from the papers; page = journal page.

## turon2018CS: Composite Structures 184 (2018) 506-511
- MMB (Sec. 3.1, p. 509, Table 1): AS4/PEEK props E11 122.7, E22=E33 10.1, G12=G13 5.5, G23 3.7 GPa, nu12 0.25, nu23 0.45;
  GIc 0.969, GIIc 1.719 kJ/m2, eta 2; tau3 5..80 MPa, tau_sh 80..120 MPa.
- 25.4 wide, 102 long, arms 1.56 mm, a0 32.64 mm. 2D plane strain CPE4I, 5 elements per arm, 0.1 mm along length,
  4-node cohesive. K3 = 1e5 N/mm3, perturbation 1e-6 mm. Mixities 20, 50, 70 % (set "using LEFM").
- Reference: closed-form LEFM. Lever length c, NLGEOM, solver settings not stated.
- Skin-stringer (Sec. 3.2): Krueger et al. NASA/TM-2001-210842; generalised plane strain, C3D8I row, 0.1 mm;
  GIc 0.208, GIIc 1.334, eta 4.46 (Hansen & Martin 1999).

## turon2010CS: Composite Structures 92 (2010) 1857-1864
- DCB, ENF, MMB: 150 long, 20 wide, arms 1.55, a0 35 mm. 2D CPE4 (text says plane stress: inconsistent), 0.15 mm
  cohesive elements, 10 through thickness. E11 120, E22 10.5, G12 5.25, G23 3.48 GPa, nu 0.3/0.5;
  tau3 15, tau_sh 30/45/60 MPa; GIc 0.260, GIIc 1.002 kJ/m2, eta 2.0. MMB 50 %: c = 63.18 mm. 20, 80 % too.
- LEFM closed form in Appendix A (DCB after ASTM D5528, MMB after ASTM D6671).

## turon2006MM: Mechanics of Materials 38 (2006) 1072-1089
- AS4/PEEK DCB/ENF/MMB vs Reeder & Crews experiments. 102 x 25.4, arms 1.56. 3D 8-node decohesion, 150 along
  length, 4 across width (DCB 102). K = 1e6. Table 1 strengths 80 / 100 MPa. Table 2: Gc 0.969/1.103/1.131/1.376/1.719,
  a0 32.9/33.7/34.1/31.4/39.2 mm for 0/20/50/80/100 %. eta 2.284. Lever not modelled: end and middle loads after
  Camanho et al. 2003. Table 3: predicted vs experimental max loads (errors 3.4 to 7.0 %).

## krueger2012MMB_conf (NTRS 20120015485), MMB 50 % benchmark
- IM7/8552 [0]24, E11 161, E22=E33 11.38, G12=G13 5.2, G23 3.9 GPa, nu12 0.32, nu23 0.45; GIc 0.212, GIIc 0.774, eta 2.1.
- 2D CPE4I, 6 elements through 2h, da 0.5 mm, rigid R2D2 lever + MPCs; 3D C3D8I, 25 across width, R3D4 lever.
- a0 25.4 mm; P_crit 385 N, w_crit 1.34 mm. Benchmark from 16 VCCT models, 25.4 <= a0 <= 70.6 mm.
- Geometry details and the 20 / 80 % cases: NASA/CR-2012-217562 (not yet available).

## Finding
The repo MMB (IM7/8552, Krueger geometry, c 41.3/92.9) was compared with a curve labelled "Turon". No Turon paper
above uses IM7/8552 for the MMB. Trace that curve's source before using it.

## krueger2012CR217562: NASA/CR-2012-217562 (MMB benchmarks, all mixities)
- Fig. 1 (p. 24): b 25.4, 2h 4.5, 2L 100.8, 2L* 152.4, a0 25.4 mm, IM7/8552 [0]24. Tables I-II (p. 19): props as above,
  GIc 0.212, GIIc = GIIIc 0.774, eta 2.1.
- c = 92.9 (0.2, Fig. 3), 41.3 (0.5, Fig. 4a), 27.3 mm (0.8, Fig. 4b). 2D CPE4I, 6 through 2h, da 0.5, R2D2 lever:
  MPC same u,w + free rotation at the specimen tip; MPC same w, free u (sliding) at mid-span; w=0 lower supports.
  3D C3D8I, 25 across width, R3D4 lever.
- w_crit: 1.64 (0.2, Sec. 5.2), 1.34 (0.5, Sec. 5.3), 1.65 mm (0.8, Sec. 5.4); P_crit 385 N (0.5), 751 N (0.8).
  P_crit at 0.2 only in figures.

## krueger2008TM215123: NASA/TM-2008-215123 (DCB, SLB)
- DCB (Fig. 3a, p. 30): B 25.0, 2h 3.0, 2L 150, a 30 mm, [0]24. 3D C3D8I, 6 through 2h.
- Inconsistency: text and Table 1 say T300/1076 elastic (E11 139.4, E22 10.16, G12 4.6, G23 3.54 GPa, nu 0.30/0.436),
  Fig. 3a and the fracture criterion (Fig. 2a) say T300/914C: GIc 170.3, GIIc 493.6 J/m2, eta 1.62.
- Benchmark P_crit, delta_crit only in Fig. 14.

## camanho2002TM211737: NASA/TM-2002-211737 (precursor of Camanho et al. 2003)
- DCB T300/977-2: 150 long, 20 wide, arms 1.98, a0 55 mm (Morais et al. tests). Table 1: E11 150, E22 11, G12 6,
  G23 3.7 GPa; GIc 0.268 kJ/m2, tau 45 MPa; K 1e6; 21-node solids, 18-node decohesion, 1 mm.
- AS4/PEEK DCB/ENF/MMB: Tables 2-4 as Turon 2006 (same data); MMB end/middle load split in Appendix B,
  experimental eta in Appendix D.
- Repo T300_9772 DCB matches this geometry (b given as 2.0 in the repo vs 20 mm here) but its cohesive set
  (GIc 0.352, 30 MPa) does not match Table 1 (0.268, 45 MPa). Check.

## Optional
- ASTM D5528 / D6671 (LEFM formulas are also in Turon 2010 App. A and Camanho 2002 App. B)
- Krueger NASA/CR-2011-217305 (ENF mode II benchmark), only if ENF is added
- Reeder & Crews 1990, Morais et al. (experimental curves) for the validation step

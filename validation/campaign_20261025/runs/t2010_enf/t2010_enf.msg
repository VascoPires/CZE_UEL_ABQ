1

   Abaqus 2023                                  Date 25-Oct-2026   Time 18:22:50
   For use by MONTANUNIVERSITAT LEOBEN under license from Dassault Systemes or its subsidiary.

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     S T A T I C   A N A L Y S I S


                                                                                          

     AUTOMATIC TIME CONTROL WITH -
          A SUGGESTED INITIAL TIME INCREMENT OF                2.000E-02
          AND A TOTAL TIME PERIOD OF                            1.00    
          THE MINIMUM TIME INCREMENT ALLOWED IS                1.000E-10
          THE MAXIMUM TIME INCREMENT ALLOWED IS                2.000E-02

     LINEAR EQUATION SOLVER TYPE         DIRECT SPARSE

 CONVERGENCE TOLERANCE PARAMETERS FOR FORCE    
     CRITERION FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR DISP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE FORCE                               1.000E-02
     AVERAGE FORCE     IS TIME AVERAGE FORCE    
     ALTERNATE CRIT. FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO FORCE     RELATIVE TO TIME AVRG. FORCE         1.000E-05
     CRITERION FOR RESIDUAL FORCE     WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR DISP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL FORCE     FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO FORCE     REL. TO TIME AVRG. MAX. FORCE        1.000E-05
     CRITERION FOR ZERO DISP.    RELATIVE TO CHARACTERISTIC LENGTH     1.000E-08

     VOLUMETRIC STRAIN COMPATIBILITY TOLERANCE FOR HYBRID SOLIDS       1.000E-05
     AXIAL STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS             1.000E-05
     TRANS. SHEAR STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS      1.000E-05
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P>P0          5.000E-03
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P=0.0         0.100    
     CONTACT FORCE ERROR TOLERANCE FOR CONVERT SDI=YES                 1.00    
     DISPLACEMENT COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS           1.000E-05
     ROTATION COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS               1.000E-05

 EQUILIBRIUM WILL BE CHECKED FOR SEVERE DISCONTINUITY ITERATIONS

 TIME INCREMENTATION CONTROL PARAMETERS:
     FIRST EQUILIBRIUM ITERATION FOR CONSECUTIVE DIVERGENCE CHECK              4
     EQUILIBRIUM ITERATION AT WHICH LOG. CONVERGENCE RATE CHECK BEGINS         8
     EQUILIBRIUM ITERATION AFTER WHICH ALTERNATE RESIDUAL IS USED              9
     MAXIMUM EQUILIBRIUM ITERATIONS ALLOWED                                   16
     EQUILIBRIUM ITERATION COUNT FOR CUT-BACK IN NEXT INCREMENT               10
     MAXIMUM EQUILIB. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      4
     MAXIMUM ITERATIONS FOR SEVERE DISCONTINUITIES                            50
     MAXIMUM ATTEMPTS ALLOWED IN AN INCREMENT                                  5
     MAXIMUM DISCON. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      50
     MAXIMUM CONTACT AUGMENTATIONS FOR *SURFACE BEHAVIOR,AUGMENTED LAGRANGE   50
     CUT-BACK FACTOR AFTER DIVERGENCE                                 0.2500    
     CUT-BACK FACTOR FOR TOO SLOW CONVERGENCE                         0.5000    
     CUT-BACK FACTOR AFTER TOO MANY EQUILIBRIUM ITERATIONS            0.7500    
     CUT-BACK FACTOR AFTER TOO MANY SEVERE DISCONTINUITY ITERATIONS   0.2500    
     CUT-BACK FACTOR AFTER PROBLEMS IN ELEMENT ASSEMBLY               0.2500    
     INCREASE FACTOR AFTER TWO INCREMENTS THAT CONVERGE QUICKLY        1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED                       1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DYNAMICS)            1.250    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DIFFUSION)           2.000    
     MINIMUM TIME INCREMENT RATIO FOR EXTRAPOLATION TO OCCUR          0.1000    
     MAX. RATIO OF TIME INCREMENT TO STABILITY LIMIT                   1.000    
     FRACTION OF STABILITY LIMIT FOR NEW TIME INCREMENT               0.9500    
     TIME INCREMENT INCREASE FACTOR BEFORE A TIME POINT                1.000    
     GLOBAL STABILIZATION CONTROL IS NOT USED

          PRINT OF INCREMENT NUMBER, TIME, ETC., EVERY    1  INCREMENTS

 LINE SEARCH TOLERANCE PARAMETERS:
 *** MAXIMUM NUMBER OF ZERO-FINDING ITERATIONS                                 5
     MAXIMUM CORRECTION SCALE FACTOR                                    1.00    
     MINIMUM CORRECTION SCALE FACTOR                                   1.000E-04
     RESIDUAL REDUCTION FACTOR TO END LINE SEARCHING                   0.250    
     RATIO OF NEW TO OLD SCALE FACTORS TO END LINE SEARCHING           0.100    

 *** INDICATES USE OF NON-DEFAULT LINE SEARCH TOLERANCE PARAMETER
     SELECTED WITH *CONTROLS

     THE MAXIMUM NUMBER OF INCREMENTS IN THIS STEP IS                  100000

          LARGE DISPLACEMENT THEORY WILL BE USED

          UNSYMMETRIC MATRIX STORAGE AND SOLUTION WILL BE USED

     LINEAR EXTRAPOLATION WILL BE USED

     CHARACTERISTIC ELEMENT LENGTH     0.612    

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.000E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13830
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.742e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.

     NEGATIVE EIGENVALUES MEAN THAT THE SYSTEM MATRIX IS NOT 
     POSITIVE DEFINITE:
     FOR EXAMPLE, A BIFURCATION (BUCKLING) LOAD MAY HAVE BEEN EXCEEDED.

 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3456.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3456. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3445.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3445. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3434.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3434. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3433.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3433. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3454.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3454. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3455.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3455. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3460.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3460. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3461.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3461. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3464.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3464. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3463.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3463. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3462.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3462. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3453.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3453. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3452.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3452. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3459.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3459. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3458.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3458. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3457.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3457. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3438.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3438. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3437.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3437. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3436.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3436. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3435.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3435. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3439.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3439. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3444.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3444. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3443.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3443. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3447.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3447. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3446.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3446. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3441.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3441. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3440.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3440. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3448.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3448. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3451.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3451. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3450.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3450. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3449.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3449. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3473.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3473. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3474.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3474. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3483.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3483. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3482.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3482. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3475.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3475. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3476.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3476. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3478.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3478. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3477.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3477. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3481.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3481. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3480.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3480. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3479.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3479. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3472.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3472. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3471.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3471. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3470.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3470. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3468.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3468. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3467.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3467. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3466.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3466. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 3465.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 3465. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: Solver problem. Zero pivot when processing D.O.F. 1 of 49 nodes. 
             The nodes have been identified in node set 
             WarnNodeSolvProbZeroPiv_1_1_1_1_1.

                   50 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   50 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 289.958E-06   AT NODE 5194 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -6.87325E+15 AT NODE 5199 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      5.003E+11   TIME AVG. FORCE       5.003E+11
 LARGEST SCALED RESIDUAL FORCE      5.229E+15   AT NODE       3483   DOF  2
  CORRESPONDING RESIDUAL FORCE      5.229E+15
 LARGEST INCREMENT OF DISP.        -0.182       AT NODE       1632   DOF  2
 LARGEST CORRECTION TO DISP.       -0.182       AT NODE       1632   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 3483 
             IS 95.0152E+12  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13780
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.744e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 48 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

 ***WARNING: OVERCONSTRAINT CHECKS: THE MAXIMUM NUMBER OF ZERO PIVOT-BASED 
             OVERCONSTRAINT CHECKS MESSAGES PRINTED FOR THIS STEP HAS BEEN 
             REACHED. THE OUTPUT OF THESE MESSAGES IS SUPRESSED UNTIL THE END 
             OF THE STEP TO AVOID POTENTIALLY LARGE INCREASES IN THE SYSTEM 
             TIME NEEDED TO COMPLETE THE ANALYSIS.
   ELEMENT 1 IS DISTORTING EXCESSIVELY.
   ELEMENT 1 IS DISTORTING EXCESSIVELY.
   ELEMENT 1 IS DISTORTING EXCESSIVELY.
   ELEMENT 1 IS DISTORTING EXCESSIVELY.
   ELEMENT 599 IS DISTORTING EXCESSIVELY.
   ELEMENT 599 IS DISTORTING EXCESSIVELY.
   ELEMENT 606 IS DISTORTING EXCESSIVELY.
   ELEMENT 606 IS DISTORTING EXCESSIVELY.
   ELEMENT 613 IS DISTORTING EXCESSIVELY.
   ELEMENT 613 IS DISTORTING EXCESSIVELY.
   ELEMENT 615 IS DISTORTING EXCESSIVELY.
   ELEMENT 615 IS DISTORTING EXCESSIVELY.
   ELEMENT 616 IS DISTORTING EXCESSIVELY.
   ELEMENT 616 IS DISTORTING EXCESSIVELY.
   ELEMENT 616 IS DISTORTING EXCESSIVELY.
   ELEMENT 616 IS DISTORTING EXCESSIVELY.
   ELEMENT 616 IS DISTORTING EXCESSIVELY.
   ELEMENT 616 IS DISTORTING EXCESSIVELY.
   ELEMENT 616 IS DISTORTING EXCESSIVELY.
   ELEMENT 616 IS DISTORTING EXCESSIVELY.
   ELEMENT 617 IS DISTORTING EXCESSIVELY.
   ELEMENT 617 IS DISTORTING EXCESSIVELY.
   ELEMENT 880 IS DISTORTING EXCESSIVELY.
   ELEMENT 880 IS DISTORTING EXCESSIVELY.
   ELEMENT 880 IS DISTORTING EXCESSIVELY.
   ELEMENT 880 IS DISTORTING EXCESSIVELY.
   ELEMENT 881 IS DISTORTING EXCESSIVELY.
   ELEMENT 881 IS DISTORTING EXCESSIVELY.
   ELEMENT 881 IS DISTORTING EXCESSIVELY.
   ELEMENT 881 IS DISTORTING EXCESSIVELY.
   ELEMENT 881 IS DISTORTING EXCESSIVELY.
   ELEMENT 881 IS DISTORTING EXCESSIVELY.
   ELEMENT 881 IS DISTORTING EXCESSIVELY.
   ELEMENT 881 IS DISTORTING EXCESSIVELY.
   ELEMENT 882 IS DISTORTING EXCESSIVELY.
   ELEMENT 882 IS DISTORTING EXCESSIVELY.
   ELEMENT 882 IS DISTORTING EXCESSIVELY.
   ELEMENT 882 IS DISTORTING EXCESSIVELY.
   ELEMENT 883 IS DISTORTING EXCESSIVELY.
   ELEMENT 883 IS DISTORTING EXCESSIVELY.
   ELEMENT 883 IS DISTORTING EXCESSIVELY.
   ELEMENT 883 IS DISTORTING EXCESSIVELY.
   ELEMENT 883 IS DISTORTING EXCESSIVELY.
   ELEMENT 883 IS DISTORTING EXCESSIVELY.
   ELEMENT 883 IS DISTORTING EXCESSIVELY.
   ELEMENT 883 IS DISTORTING EXCESSIVELY.
   ELEMENT 884 IS DISTORTING EXCESSIVELY.
   ELEMENT 884 IS DISTORTING EXCESSIVELY.
   ELEMENT 884 IS DISTORTING EXCESSIVELY.
   ELEMENT 884 IS DISTORTING EXCESSIVELY.
   ELEMENT 885 IS DISTORTING EXCESSIVELY.
   ELEMENT 885 IS DISTORTING EXCESSIVELY.
   ELEMENT 885 IS DISTORTING EXCESSIVELY.
   ELEMENT 885 IS DISTORTING EXCESSIVELY.
   ELEMENT 885 IS DISTORTING EXCESSIVELY.
   ELEMENT 885 IS DISTORTING EXCESSIVELY.
   ELEMENT 885 IS DISTORTING EXCESSIVELY.
   ELEMENT 885 IS DISTORTING EXCESSIVELY.
   ELEMENT 886 IS DISTORTING EXCESSIVELY.
   ELEMENT 886 IS DISTORTING EXCESSIVELY.
   ELEMENT 886 IS DISTORTING EXCESSIVELY.
   ELEMENT 886 IS DISTORTING EXCESSIVELY.
   ELEMENT 886 IS DISTORTING EXCESSIVELY.
   ELEMENT 886 IS DISTORTING EXCESSIVELY.
   ELEMENT 886 IS DISTORTING EXCESSIVELY.
   ELEMENT 886 IS DISTORTING EXCESSIVELY.
   ELEMENT 887 IS DISTORTING EXCESSIVELY.
   ELEMENT 887 IS DISTORTING EXCESSIVELY.
   ELEMENT 887 IS DISTORTING EXCESSIVELY.
   ELEMENT 887 IS DISTORTING EXCESSIVELY.
   ELEMENT 887 IS DISTORTING EXCESSIVELY.
   ELEMENT 887 IS DISTORTING EXCESSIVELY.
   ELEMENT 887 IS DISTORTING EXCESSIVELY.
   ELEMENT 887 IS DISTORTING EXCESSIVELY.
   ELEMENT 888 IS DISTORTING EXCESSIVELY.
   ELEMENT 888 IS DISTORTING EXCESSIVELY.
   ELEMENT 888 IS DISTORTING EXCESSIVELY.
   ELEMENT 888 IS DISTORTING EXCESSIVELY.
   ELEMENT 889 IS DISTORTING EXCESSIVELY.
   ELEMENT 889 IS DISTORTING EXCESSIVELY.
   ELEMENT 889 IS DISTORTING EXCESSIVELY.
   ELEMENT 889 IS DISTORTING EXCESSIVELY.
   ELEMENT 889 IS DISTORTING EXCESSIVELY.
   ELEMENT 889 IS DISTORTING EXCESSIVELY.
   ELEMENT 889 IS DISTORTING EXCESSIVELY.
   ELEMENT 889 IS DISTORTING EXCESSIVELY.
   ELEMENT 890 IS DISTORTING EXCESSIVELY.
   ELEMENT 890 IS DISTORTING EXCESSIVELY.
   ELEMENT 890 IS DISTORTING EXCESSIVELY.
   ELEMENT 890 IS DISTORTING EXCESSIVELY.
   ELEMENT 890 IS DISTORTING EXCESSIVELY.
   ELEMENT 890 IS DISTORTING EXCESSIVELY.
   ELEMENT 890 IS DISTORTING EXCESSIVELY.
   ELEMENT 890 IS DISTORTING EXCESSIVELY.
   ELEMENT 891 IS DISTORTING EXCESSIVELY.
   ELEMENT 891 IS DISTORTING EXCESSIVELY.
   ELEMENT 891 IS DISTORTING EXCESSIVELY.
   ELEMENT 891 IS DISTORTING EXCESSIVELY.
   ELEMENT 892 IS DISTORTING EXCESSIVELY.
   ELEMENT 892 IS DISTORTING EXCESSIVELY.
   ELEMENT 892 IS DISTORTING EXCESSIVELY.
   ELEMENT 892 IS DISTORTING EXCESSIVELY.
   ELEMENT 892 IS DISTORTING EXCESSIVELY.
   ELEMENT 892 IS DISTORTING EXCESSIVELY.
   ELEMENT 893 IS DISTORTING EXCESSIVELY.
   ELEMENT 893 IS DISTORTING EXCESSIVELY.
   ELEMENT 893 IS DISTORTING EXCESSIVELY.
   ELEMENT 893 IS DISTORTING EXCESSIVELY.
   ELEMENT 893 IS DISTORTING EXCESSIVELY.
   ELEMENT 893 IS DISTORTING EXCESSIVELY.
   ELEMENT 894 IS DISTORTING EXCESSIVELY.
   ELEMENT 894 IS DISTORTING EXCESSIVELY.
   ELEMENT 894 IS DISTORTING EXCESSIVELY.
   ELEMENT 894 IS DISTORTING EXCESSIVELY.
   ELEMENT 897 IS DISTORTING EXCESSIVELY.
   ELEMENT 897 IS DISTORTING EXCESSIVELY.
   ELEMENT 898 IS DISTORTING EXCESSIVELY.
   ELEMENT 898 IS DISTORTING EXCESSIVELY.
   ELEMENT 899 IS DISTORTING EXCESSIVELY.
   ELEMENT 899 IS DISTORTING EXCESSIVELY.
   ELEMENT 899 IS DISTORTING EXCESSIVELY.
   ELEMENT 899 IS DISTORTING EXCESSIVELY.
   ELEMENT 900 IS DISTORTING EXCESSIVELY.
   ELEMENT 900 IS DISTORTING EXCESSIVELY.
   ELEMENT 900 IS DISTORTING EXCESSIVELY.
   ELEMENT 900 IS DISTORTING EXCESSIVELY.
   ELEMENT 900 IS DISTORTING EXCESSIVELY.
   ELEMENT 900 IS DISTORTING EXCESSIVELY.
   ELEMENT 901 IS DISTORTING EXCESSIVELY.
   ELEMENT 901 IS DISTORTING EXCESSIVELY.
   ELEMENT 901 IS DISTORTING EXCESSIVELY.
   ELEMENT 901 IS DISTORTING EXCESSIVELY.
   ELEMENT 901 IS DISTORTING EXCESSIVELY.
   ELEMENT 901 IS DISTORTING EXCESSIVELY.
   ELEMENT 902 IS DISTORTING EXCESSIVELY.
   ELEMENT 902 IS DISTORTING EXCESSIVELY.
   ELEMENT 903 IS DISTORTING EXCESSIVELY.
   ELEMENT 903 IS DISTORTING EXCESSIVELY.
   ELEMENT 903 IS DISTORTING EXCESSIVELY.
   ELEMENT 903 IS DISTORTING EXCESSIVELY.
   ELEMENT 904 IS DISTORTING EXCESSIVELY.
   ELEMENT 904 IS DISTORTING EXCESSIVELY.
   ELEMENT 904 IS DISTORTING EXCESSIVELY.
   ELEMENT 904 IS DISTORTING EXCESSIVELY.
   ELEMENT 904 IS DISTORTING EXCESSIVELY.
   ELEMENT 904 IS DISTORTING EXCESSIVELY.
   ELEMENT 905 IS DISTORTING EXCESSIVELY.
   ELEMENT 905 IS DISTORTING EXCESSIVELY.
   ELEMENT 905 IS DISTORTING EXCESSIVELY.
   ELEMENT 905 IS DISTORTING EXCESSIVELY.
   ELEMENT 906 IS DISTORTING EXCESSIVELY.
   ELEMENT 906 IS DISTORTING EXCESSIVELY.
   ELEMENT 907 IS DISTORTING EXCESSIVELY.
   ELEMENT 907 IS DISTORTING EXCESSIVELY.
   ELEMENT 907 IS DISTORTING EXCESSIVELY.
   ELEMENT 907 IS DISTORTING EXCESSIVELY.
   ELEMENT 1164 IS DISTORTING EXCESSIVELY.
   ELEMENT 1164 IS DISTORTING EXCESSIVELY.
   ELEMENT 1164 IS DISTORTING EXCESSIVELY.
   ELEMENT 1164 IS DISTORTING EXCESSIVELY.
   ELEMENT 1165 IS DISTORTING EXCESSIVELY.
   ELEMENT 1165 IS DISTORTING EXCESSIVELY.
   ELEMENT 1165 IS DISTORTING EXCESSIVELY.
   ELEMENT 1165 IS DISTORTING EXCESSIVELY.
   ELEMENT 1166 IS DISTORTING EXCESSIVELY.
   ELEMENT 1166 IS DISTORTING EXCESSIVELY.
   ELEMENT 1166 IS DISTORTING EXCESSIVELY.
   ELEMENT 1166 IS DISTORTING EXCESSIVELY.
   ELEMENT 1167 IS DISTORTING EXCESSIVELY.
   ELEMENT 1167 IS DISTORTING EXCESSIVELY.
   ELEMENT 1167 IS DISTORTING EXCESSIVELY.
   ELEMENT 1167 IS DISTORTING EXCESSIVELY.
   ELEMENT 1167 IS DISTORTING EXCESSIVELY.
   ELEMENT 1167 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1169 IS DISTORTING EXCESSIVELY.
   ELEMENT 1169 IS DISTORTING EXCESSIVELY.
   ELEMENT 1169 IS DISTORTING EXCESSIVELY.
   ELEMENT 1169 IS DISTORTING EXCESSIVELY.
   ELEMENT 1170 IS DISTORTING EXCESSIVELY.
   ELEMENT 1170 IS DISTORTING EXCESSIVELY.
   ELEMENT 1170 IS DISTORTING EXCESSIVELY.
   ELEMENT 1170 IS DISTORTING EXCESSIVELY.
   ELEMENT 1170 IS DISTORTING EXCESSIVELY.
   ELEMENT 1170 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1172 IS DISTORTING EXCESSIVELY.
   ELEMENT 1172 IS DISTORTING EXCESSIVELY.
   ELEMENT 1173 IS DISTORTING EXCESSIVELY.
   ELEMENT 1173 IS DISTORTING EXCESSIVELY.
   ELEMENT 1174 IS DISTORTING EXCESSIVELY.
   ELEMENT 1174 IS DISTORTING EXCESSIVELY.
   ELEMENT 1175 IS DISTORTING EXCESSIVELY.
   ELEMENT 1175 IS DISTORTING EXCESSIVELY.
   ELEMENT 1176 IS DISTORTING EXCESSIVELY.
   ELEMENT 1176 IS DISTORTING EXCESSIVELY.
   ELEMENT 1176 IS DISTORTING EXCESSIVELY.
   ELEMENT 1176 IS DISTORTING EXCESSIVELY.
   ELEMENT 1177 IS DISTORTING EXCESSIVELY.
   ELEMENT 1177 IS DISTORTING EXCESSIVELY.
   ELEMENT 1178 IS DISTORTING EXCESSIVELY.
   ELEMENT 1178 IS DISTORTING EXCESSIVELY.
   ELEMENT 1178 IS DISTORTING EXCESSIVELY.
   ELEMENT 1178 IS DISTORTING EXCESSIVELY.
   ELEMENT 1179 IS DISTORTING EXCESSIVELY.
   ELEMENT 1179 IS DISTORTING EXCESSIVELY.
   ELEMENT 1179 IS DISTORTING EXCESSIVELY.
   ELEMENT 1179 IS DISTORTING EXCESSIVELY.
   ELEMENT 1179 IS DISTORTING EXCESSIVELY.
   ELEMENT 1179 IS DISTORTING EXCESSIVELY.
   ELEMENT 1180 IS DISTORTING EXCESSIVELY.
   ELEMENT 1180 IS DISTORTING EXCESSIVELY.
   ELEMENT 1181 IS DISTORTING EXCESSIVELY.
   ELEMENT 1181 IS DISTORTING EXCESSIVELY.
   ELEMENT 1181 IS DISTORTING EXCESSIVELY.
   ELEMENT 1181 IS DISTORTING EXCESSIVELY.
   ELEMENT 1182 IS DISTORTING EXCESSIVELY.
   ELEMENT 1182 IS DISTORTING EXCESSIVELY.
   ELEMENT 1182 IS DISTORTING EXCESSIVELY.
   ELEMENT 1182 IS DISTORTING EXCESSIVELY.
   ELEMENT 1183 IS DISTORTING EXCESSIVELY.
   ELEMENT 1183 IS DISTORTING EXCESSIVELY.
   ELEMENT 1184 IS DISTORTING EXCESSIVELY.
   ELEMENT 1184 IS DISTORTING EXCESSIVELY.
   ELEMENT 1184 IS DISTORTING EXCESSIVELY.
   ELEMENT 1184 IS DISTORTING EXCESSIVELY.
   ELEMENT 1185 IS DISTORTING EXCESSIVELY.
   ELEMENT 1185 IS DISTORTING EXCESSIVELY.
   ELEMENT 1185 IS DISTORTING EXCESSIVELY.
   ELEMENT 1185 IS DISTORTING EXCESSIVELY.
   ELEMENT 1186 IS DISTORTING EXCESSIVELY.
   ELEMENT 1186 IS DISTORTING EXCESSIVELY.
   ELEMENT 1186 IS DISTORTING EXCESSIVELY.
   ELEMENT 1186 IS DISTORTING EXCESSIVELY.
   ELEMENT 1188 IS DISTORTING EXCESSIVELY.
   ELEMENT 1188 IS DISTORTING EXCESSIVELY.
   ELEMENT 1188 IS DISTORTING EXCESSIVELY.
   ELEMENT 1188 IS DISTORTING EXCESSIVELY.
   ELEMENT 1189 IS DISTORTING EXCESSIVELY.
   ELEMENT 1189 IS DISTORTING EXCESSIVELY.
   ELEMENT 1189 IS DISTORTING EXCESSIVELY.
   ELEMENT 1189 IS DISTORTING EXCESSIVELY.
   ELEMENT 1190 IS DISTORTING EXCESSIVELY.
   ELEMENT 1190 IS DISTORTING EXCESSIVELY.
   ELEMENT 1191 IS DISTORTING EXCESSIVELY.
   ELEMENT 1191 IS DISTORTING EXCESSIVELY.
   ELEMENT 1192 IS DISTORTING EXCESSIVELY.
   ELEMENT 1192 IS DISTORTING EXCESSIVELY.
   ELEMENT 1192 IS DISTORTING EXCESSIVELY.
   ELEMENT 1192 IS DISTORTING EXCESSIVELY.
   ELEMENT 1193 IS DISTORTING EXCESSIVELY.
   ELEMENT 1193 IS DISTORTING EXCESSIVELY.
   ELEMENT 1449 IS DISTORTING EXCESSIVELY.
   ELEMENT 1449 IS DISTORTING EXCESSIVELY.
   ELEMENT 1449 IS DISTORTING EXCESSIVELY.
   ELEMENT 1449 IS DISTORTING EXCESSIVELY.
   ELEMENT 1450 IS DISTORTING EXCESSIVELY.
   ELEMENT 1450 IS DISTORTING EXCESSIVELY.
   ELEMENT 1450 IS DISTORTING EXCESSIVELY.
   ELEMENT 1450 IS DISTORTING EXCESSIVELY.
   ELEMENT 1451 IS DISTORTING EXCESSIVELY.
   ELEMENT 1451 IS DISTORTING EXCESSIVELY.
   ELEMENT 1451 IS DISTORTING EXCESSIVELY.
   ELEMENT 1451 IS DISTORTING EXCESSIVELY.
   ELEMENT 1452 IS DISTORTING EXCESSIVELY.
   ELEMENT 1452 IS DISTORTING EXCESSIVELY.
   ELEMENT 1452 IS DISTORTING EXCESSIVELY.
   ELEMENT 1452 IS DISTORTING EXCESSIVELY.
   ELEMENT 1453 IS DISTORTING EXCESSIVELY.
   ELEMENT 1453 IS DISTORTING EXCESSIVELY.
   ELEMENT 1453 IS DISTORTING EXCESSIVELY.
   ELEMENT 1453 IS DISTORTING EXCESSIVELY.
   ELEMENT 1454 IS DISTORTING EXCESSIVELY.
   ELEMENT 1454 IS DISTORTING EXCESSIVELY.
   ELEMENT 1454 IS DISTORTING EXCESSIVELY.
   ELEMENT 1454 IS DISTORTING EXCESSIVELY.
   ELEMENT 1455 IS DISTORTING EXCESSIVELY.
   ELEMENT 1455 IS DISTORTING EXCESSIVELY.
   ELEMENT 1455 IS DISTORTING EXCESSIVELY.
   ELEMENT 1455 IS DISTORTING EXCESSIVELY.
   ELEMENT 1455 IS DISTORTING EXCESSIVELY.
   ELEMENT 1455 IS DISTORTING EXCESSIVELY.
   ELEMENT 1456 IS DISTORTING EXCESSIVELY.
   ELEMENT 1456 IS DISTORTING EXCESSIVELY.
   ELEMENT 1456 IS DISTORTING EXCESSIVELY.
   ELEMENT 1456 IS DISTORTING EXCESSIVELY.
   ELEMENT 1458 IS DISTORTING EXCESSIVELY.
   ELEMENT 1458 IS DISTORTING EXCESSIVELY.
   ELEMENT 1459 IS DISTORTING EXCESSIVELY.
   ELEMENT 1459 IS DISTORTING EXCESSIVELY.
   ELEMENT 1460 IS DISTORTING EXCESSIVELY.
   ELEMENT 1460 IS DISTORTING EXCESSIVELY.
   ELEMENT 1461 IS DISTORTING EXCESSIVELY.
   ELEMENT 1461 IS DISTORTING EXCESSIVELY.
   ELEMENT 1461 IS DISTORTING EXCESSIVELY.
   ELEMENT 1461 IS DISTORTING EXCESSIVELY.
   ELEMENT 1462 IS DISTORTING EXCESSIVELY.
   ELEMENT 1462 IS DISTORTING EXCESSIVELY.
   ELEMENT 1463 IS DISTORTING EXCESSIVELY.
   ELEMENT 1463 IS DISTORTING EXCESSIVELY.
   ELEMENT 1463 IS DISTORTING EXCESSIVELY.
   ELEMENT 1463 IS DISTORTING EXCESSIVELY.
   ELEMENT 1464 IS DISTORTING EXCESSIVELY.
   ELEMENT 1464 IS DISTORTING EXCESSIVELY.
   ELEMENT 1464 IS DISTORTING EXCESSIVELY.
   ELEMENT 1464 IS DISTORTING EXCESSIVELY.
   ELEMENT 1464 IS DISTORTING EXCESSIVELY.
   ELEMENT 1464 IS DISTORTING EXCESSIVELY.
   ELEMENT 1465 IS DISTORTING EXCESSIVELY.
   ELEMENT 1465 IS DISTORTING EXCESSIVELY.
   ELEMENT 1466 IS DISTORTING EXCESSIVELY.
   ELEMENT 1466 IS DISTORTING EXCESSIVELY.
   ELEMENT 1466 IS DISTORTING EXCESSIVELY.
   ELEMENT 1466 IS DISTORTING EXCESSIVELY.
   ELEMENT 1467 IS DISTORTING EXCESSIVELY.
   ELEMENT 1467 IS DISTORTING EXCESSIVELY.
   ELEMENT 1467 IS DISTORTING EXCESSIVELY.
   ELEMENT 1467 IS DISTORTING EXCESSIVELY.
   ELEMENT 1468 IS DISTORTING EXCESSIVELY.
   ELEMENT 1468 IS DISTORTING EXCESSIVELY.
   ELEMENT 1469 IS DISTORTING EXCESSIVELY.
   ELEMENT 1469 IS DISTORTING EXCESSIVELY.
   ELEMENT 1469 IS DISTORTING EXCESSIVELY.
   ELEMENT 1469 IS DISTORTING EXCESSIVELY.
   ELEMENT 1470 IS DISTORTING EXCESSIVELY.
   ELEMENT 1470 IS DISTORTING EXCESSIVELY.
   ELEMENT 1470 IS DISTORTING EXCESSIVELY.
   ELEMENT 1470 IS DISTORTING EXCESSIVELY.
   ELEMENT 1471 IS DISTORTING EXCESSIVELY.
   ELEMENT 1471 IS DISTORTING EXCESSIVELY.
   ELEMENT 1471 IS DISTORTING EXCESSIVELY.
   ELEMENT 1471 IS DISTORTING EXCESSIVELY.
   ELEMENT 1473 IS DISTORTING EXCESSIVELY.
   ELEMENT 1473 IS DISTORTING EXCESSIVELY.
   ELEMENT 1473 IS DISTORTING EXCESSIVELY.
   ELEMENT 1473 IS DISTORTING EXCESSIVELY.
   ELEMENT 1474 IS DISTORTING EXCESSIVELY.
   ELEMENT 1474 IS DISTORTING EXCESSIVELY.
   ELEMENT 1474 IS DISTORTING EXCESSIVELY.
   ELEMENT 1474 IS DISTORTING EXCESSIVELY.
   ELEMENT 1475 IS DISTORTING EXCESSIVELY.
   ELEMENT 1475 IS DISTORTING EXCESSIVELY.
   ELEMENT 1476 IS DISTORTING EXCESSIVELY.
   ELEMENT 1476 IS DISTORTING EXCESSIVELY.
   ELEMENT 1477 IS DISTORTING EXCESSIVELY.
   ELEMENT 1477 IS DISTORTING EXCESSIVELY.
   ELEMENT 1477 IS DISTORTING EXCESSIVELY.
   ELEMENT 1477 IS DISTORTING EXCESSIVELY.
   ELEMENT 1478 IS DISTORTING EXCESSIVELY.
   ELEMENT 1478 IS DISTORTING EXCESSIVELY.
   ELEMENT 1735 IS DISTORTING EXCESSIVELY.
   ELEMENT 1735 IS DISTORTING EXCESSIVELY.
   ELEMENT 1735 IS DISTORTING EXCESSIVELY.
   ELEMENT 1735 IS DISTORTING EXCESSIVELY.
   ELEMENT 1735 IS DISTORTING EXCESSIVELY.
   ELEMENT 1735 IS DISTORTING EXCESSIVELY.
   ELEMENT 1736 IS DISTORTING EXCESSIVELY.
   ELEMENT 1736 IS DISTORTING EXCESSIVELY.
   ELEMENT 1736 IS DISTORTING EXCESSIVELY.
   ELEMENT 1736 IS DISTORTING EXCESSIVELY.
   ELEMENT 1736 IS DISTORTING EXCESSIVELY.
   ELEMENT 1736 IS DISTORTING EXCESSIVELY.
   ELEMENT 1736 IS DISTORTING EXCESSIVELY.
   ELEMENT 1736 IS DISTORTING EXCESSIVELY.
   ELEMENT 1737 IS DISTORTING EXCESSIVELY.
   ELEMENT 1737 IS DISTORTING EXCESSIVELY.
   ELEMENT 1737 IS DISTORTING EXCESSIVELY.
   ELEMENT 1737 IS DISTORTING EXCESSIVELY.
   ELEMENT 1738 IS DISTORTING EXCESSIVELY.
   ELEMENT 1738 IS DISTORTING EXCESSIVELY.
   ELEMENT 1738 IS DISTORTING EXCESSIVELY.
   ELEMENT 1738 IS DISTORTING EXCESSIVELY.
   ELEMENT 1738 IS DISTORTING EXCESSIVELY.
   ELEMENT 1738 IS DISTORTING EXCESSIVELY.
   ELEMENT 1738 IS DISTORTING EXCESSIVELY.
   ELEMENT 1738 IS DISTORTING EXCESSIVELY.
   ELEMENT 1739 IS DISTORTING EXCESSIVELY.
   ELEMENT 1739 IS DISTORTING EXCESSIVELY.
   ELEMENT 1739 IS DISTORTING EXCESSIVELY.
   ELEMENT 1739 IS DISTORTING EXCESSIVELY.
   ELEMENT 1740 IS DISTORTING EXCESSIVELY.
   ELEMENT 1740 IS DISTORTING EXCESSIVELY.
   ELEMENT 1740 IS DISTORTING EXCESSIVELY.
   ELEMENT 1740 IS DISTORTING EXCESSIVELY.
   ELEMENT 1740 IS DISTORTING EXCESSIVELY.
   ELEMENT 1740 IS DISTORTING EXCESSIVELY.
   ELEMENT 1740 IS DISTORTING EXCESSIVELY.
   ELEMENT 1740 IS DISTORTING EXCESSIVELY.
   ELEMENT 1741 IS DISTORTING EXCESSIVELY.
   ELEMENT 1741 IS DISTORTING EXCESSIVELY.
   ELEMENT 1741 IS DISTORTING EXCESSIVELY.
   ELEMENT 1741 IS DISTORTING EXCESSIVELY.
   ELEMENT 1741 IS DISTORTING EXCESSIVELY.
   ELEMENT 1741 IS DISTORTING EXCESSIVELY.
   ELEMENT 1741 IS DISTORTING EXCESSIVELY.
   ELEMENT 1741 IS DISTORTING EXCESSIVELY.
   ELEMENT 1742 IS DISTORTING EXCESSIVELY.
   ELEMENT 1742 IS DISTORTING EXCESSIVELY.
   ELEMENT 1742 IS DISTORTING EXCESSIVELY.
   ELEMENT 1742 IS DISTORTING EXCESSIVELY.
   ELEMENT 1742 IS DISTORTING EXCESSIVELY.
   ELEMENT 1742 IS DISTORTING EXCESSIVELY.
   ELEMENT 1742 IS DISTORTING EXCESSIVELY.
   ELEMENT 1742 IS DISTORTING EXCESSIVELY.
   ELEMENT 1743 IS DISTORTING EXCESSIVELY.
   ELEMENT 1743 IS DISTORTING EXCESSIVELY.
   ELEMENT 1743 IS DISTORTING EXCESSIVELY.
   ELEMENT 1743 IS DISTORTING EXCESSIVELY.
   ELEMENT 1744 IS DISTORTING EXCESSIVELY.
   ELEMENT 1744 IS DISTORTING EXCESSIVELY.
   ELEMENT 1744 IS DISTORTING EXCESSIVELY.
   ELEMENT 1744 IS DISTORTING EXCESSIVELY.
   ELEMENT 1744 IS DISTORTING EXCESSIVELY.
   ELEMENT 1744 IS DISTORTING EXCESSIVELY.
   ELEMENT 1744 IS DISTORTING EXCESSIVELY.
   ELEMENT 1744 IS DISTORTING EXCESSIVELY.
   ELEMENT 1745 IS DISTORTING EXCESSIVELY.
   ELEMENT 1745 IS DISTORTING EXCESSIVELY.
   ELEMENT 1745 IS DISTORTING EXCESSIVELY.
   ELEMENT 1745 IS DISTORTING EXCESSIVELY.
   ELEMENT 1745 IS DISTORTING EXCESSIVELY.
   ELEMENT 1745 IS DISTORTING EXCESSIVELY.
   ELEMENT 1745 IS DISTORTING EXCESSIVELY.
   ELEMENT 1745 IS DISTORTING EXCESSIVELY.
   ELEMENT 1746 IS DISTORTING EXCESSIVELY.
   ELEMENT 1746 IS DISTORTING EXCESSIVELY.
   ELEMENT 1746 IS DISTORTING EXCESSIVELY.
   ELEMENT 1746 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1748 IS DISTORTING EXCESSIVELY.
   ELEMENT 1748 IS DISTORTING EXCESSIVELY.
   ELEMENT 1748 IS DISTORTING EXCESSIVELY.
   ELEMENT 1748 IS DISTORTING EXCESSIVELY.
   ELEMENT 1748 IS DISTORTING EXCESSIVELY.
   ELEMENT 1748 IS DISTORTING EXCESSIVELY.
   ELEMENT 1749 IS DISTORTING EXCESSIVELY.
   ELEMENT 1749 IS DISTORTING EXCESSIVELY.
   ELEMENT 1749 IS DISTORTING EXCESSIVELY.
   ELEMENT 1749 IS DISTORTING EXCESSIVELY.
   ELEMENT 1752 IS DISTORTING EXCESSIVELY.
   ELEMENT 1752 IS DISTORTING EXCESSIVELY.
   ELEMENT 1753 IS DISTORTING EXCESSIVELY.
   ELEMENT 1753 IS DISTORTING EXCESSIVELY.
   ELEMENT 1754 IS DISTORTING EXCESSIVELY.
   ELEMENT 1754 IS DISTORTING EXCESSIVELY.
   ELEMENT 1754 IS DISTORTING EXCESSIVELY.
   ELEMENT 1754 IS DISTORTING EXCESSIVELY.
   ELEMENT 1755 IS DISTORTING EXCESSIVELY.
   ELEMENT 1755 IS DISTORTING EXCESSIVELY.
   ELEMENT 1755 IS DISTORTING EXCESSIVELY.
   ELEMENT 1755 IS DISTORTING EXCESSIVELY.
   ELEMENT 1755 IS DISTORTING EXCESSIVELY.
   ELEMENT 1755 IS DISTORTING EXCESSIVELY.
   ELEMENT 1756 IS DISTORTING EXCESSIVELY.
   ELEMENT 1756 IS DISTORTING EXCESSIVELY.
   ELEMENT 1756 IS DISTORTING EXCESSIVELY.
   ELEMENT 1756 IS DISTORTING EXCESSIVELY.
   ELEMENT 1756 IS DISTORTING EXCESSIVELY.
   ELEMENT 1756 IS DISTORTING EXCESSIVELY.
   ELEMENT 1757 IS DISTORTING EXCESSIVELY.
   ELEMENT 1757 IS DISTORTING EXCESSIVELY.
   ELEMENT 1758 IS DISTORTING EXCESSIVELY.
   ELEMENT 1758 IS DISTORTING EXCESSIVELY.
   ELEMENT 1758 IS DISTORTING EXCESSIVELY.
   ELEMENT 1758 IS DISTORTING EXCESSIVELY.
   ELEMENT 1759 IS DISTORTING EXCESSIVELY.
   ELEMENT 1759 IS DISTORTING EXCESSIVELY.
   ELEMENT 1759 IS DISTORTING EXCESSIVELY.
   ELEMENT 1759 IS DISTORTING EXCESSIVELY.
   ELEMENT 1759 IS DISTORTING EXCESSIVELY.
   ELEMENT 1759 IS DISTORTING EXCESSIVELY.
   ELEMENT 1760 IS DISTORTING EXCESSIVELY.
   ELEMENT 1760 IS DISTORTING EXCESSIVELY.
   ELEMENT 1760 IS DISTORTING EXCESSIVELY.
   ELEMENT 1760 IS DISTORTING EXCESSIVELY.
   ELEMENT 1761 IS DISTORTING EXCESSIVELY.
   ELEMENT 1761 IS DISTORTING EXCESSIVELY.
   ELEMENT 1762 IS DISTORTING EXCESSIVELY.
   ELEMENT 1762 IS DISTORTING EXCESSIVELY.
   ELEMENT 1762 IS DISTORTING EXCESSIVELY.
   ELEMENT 1762 IS DISTORTING EXCESSIVELY.
   ELEMENT 2024 IS DISTORTING EXCESSIVELY.
   ELEMENT 2024 IS DISTORTING EXCESSIVELY.
   ELEMENT 2031 IS DISTORTING EXCESSIVELY.
   ELEMENT 2031 IS DISTORTING EXCESSIVELY.
   ELEMENT 2038 IS DISTORTING EXCESSIVELY.
   ELEMENT 2038 IS DISTORTING EXCESSIVELY.
   ELEMENT 2040 IS DISTORTING EXCESSIVELY.
   ELEMENT 2040 IS DISTORTING EXCESSIVELY.
   ELEMENT 2041 IS DISTORTING EXCESSIVELY.
   ELEMENT 2041 IS DISTORTING EXCESSIVELY.
   ELEMENT 2041 IS DISTORTING EXCESSIVELY.
   ELEMENT 2041 IS DISTORTING EXCESSIVELY.
   ELEMENT 2041 IS DISTORTING EXCESSIVELY.
   ELEMENT 2041 IS DISTORTING EXCESSIVELY.
   ELEMENT 2041 IS DISTORTING EXCESSIVELY.
   ELEMENT 2041 IS DISTORTING EXCESSIVELY.
   ELEMENT 2042 IS DISTORTING EXCESSIVELY.
   ELEMENT 2042 IS DISTORTING EXCESSIVELY.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 510 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     1 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  5.000E-03
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13830
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.766e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   51 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   51 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 1.30741E-06 AT NODE 3433 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -1.75261E+06 AT NODE 3442 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                       59.7       TIME AVG. FORCE        59.7    
 LARGEST SCALED RESIDUAL FORCE     -1.452E+06   AT NODE       1440   DOF  2
  CORRESPONDING RESIDUAL FORCE     -1.452E+06
 LARGEST INCREMENT OF DISP.        -4.514E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -4.514E-02   AT NODE       1602   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 5158 
             IS 76.7261E+03  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13779
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.774e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 7 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   39 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   12 POINTS CHANGED FROM OPEN TO CLOSED
                   27 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR 80.6334E-15  AT NODE 5161 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -3.45176E-03 AT NODE 5151 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE CONTACT CONSTRAINT ERRORS ARE WITHIN THE TOLERANCES.

 AVERAGE FORCE                      2.193E-02   TIME AVG. FORCE       2.193E-02
 LARGEST SCALED RESIDUAL FORCE     -2.825E-03   AT NODE       1433   DOF  2
  CORRESPONDING RESIDUAL FORCE     -2.825E-03
 LARGEST INCREMENT OF DISP.        -4.514E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -3.538E-05   AT NODE        572   DOF  1
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13764
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.745e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   52 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   39 POINTS CHANGED FROM OPEN TO CLOSED
                   13 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     3

   MAX. PENETRATION ERROR 5.17077E-09 AT NODE 5160 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -154.343       AT NODE 5162 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      2.722E-02   TIME AVG. FORCE       2.722E-02
 LARGEST SCALED RESIDUAL FORCE       116.       AT NODE       3446   DOF  2
  CORRESPONDING RESIDUAL FORCE       116.    
 LARGEST INCREMENT OF DISP.        -4.514E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.        8.820E-07   AT NODE       3189   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13790
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.768e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   34 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    5 POINTS CHANGED FROM OPEN TO CLOSED
                   29 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     4

   MAX. PENETRATION ERROR 46.8375E-18  AT NODE 5160 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -711.447       AT NODE 3444 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      9.301E-02   TIME AVG. FORCE       9.301E-02
 LARGEST SCALED RESIDUAL FORCE      -584.       AT NODE       1442   DOF  2
  CORRESPONDING RESIDUAL FORCE      -584.    
 LARGEST INCREMENT OF DISP.        -4.514E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.        5.470E-09   AT NODE       3482   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13766
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.745e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   23 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   16 POINTS CHANGED FROM OPEN TO CLOSED
                    7 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     5

   MAX. PENETRATION ERROR 144.199E-18   AT NODE 3434 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -500.722       AT NODE 5169 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      3.064E-02   TIME AVG. FORCE       3.064E-02
 LARGEST SCALED RESIDUAL FORCE      -376.       AT NODE       1451   DOF  2
  CORRESPONDING RESIDUAL FORCE      -376.    
 LARGEST INCREMENT OF DISP.        -4.514E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -5.540E-07   AT NODE       1433   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13775
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.749e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   29 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   10 POINTS CHANGED FROM OPEN TO CLOSED
                   19 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     6

   MAX. PENETRATION ERROR 17.7622E-09  AT NODE 3439 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -3.95295E+03 AT NODE 3447 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.180       TIME AVG. FORCE       0.180    
 LARGEST SCALED RESIDUAL FORCE      3.131E+03   AT NODE       3447   DOF  2
  CORRESPONDING RESIDUAL FORCE      3.131E+03
 LARGEST INCREMENT OF DISP.        -4.514E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -5.544E-07   AT NODE       3151   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13766
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.737e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   24 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   13 POINTS CHANGED FROM OPEN TO CLOSED
                   11 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     7

   MAX. PENETRATION ERROR 81.5152E-12  AT NODE 3448 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -3.10672E+03 AT NODE 3465 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      8.920E-02   TIME AVG. FORCE       8.920E-02
 LARGEST SCALED RESIDUAL FORCE     -2.336E+03   AT NODE       1463   DOF  2
  CORRESPONDING RESIDUAL FORCE     -2.336E+03
 LARGEST INCREMENT OF DISP.        -4.514E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -1.487E-08   AT NODE       1154   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13768
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.740e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   26 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   10 POINTS CHANGED FROM OPEN TO CLOSED
                   16 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     8

   MAX. PENETRATION ERROR 111.022E-18   AT NODE 3466 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -297.086       AT NODE 5183 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      4.175E-02   TIME AVG. FORCE       4.175E-02
 LARGEST SCALED RESIDUAL FORCE      -236.       AT NODE       1465   DOF  2
  CORRESPONDING RESIDUAL FORCE      -236.    
 LARGEST INCREMENT OF DISP.        -4.514E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.        2.646E-09   AT NODE       5169   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13762
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.732e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   33 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   19 POINTS CHANGED FROM OPEN TO CLOSED
                   14 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     9

   MAX. PENETRATION ERROR 120.069E-09   AT NODE 5156 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -549.930       AT NODE 3466 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      5.632E-02   TIME AVG. FORCE       5.632E-02
 LARGEST SCALED RESIDUAL FORCE       454.       AT NODE       3464   DOF  2
  CORRESPONDING RESIDUAL FORCE       454.    
 LARGEST INCREMENT OF DISP.        -4.514E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -6.140E-07   AT NODE       1436   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13767
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.734e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   24 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   11 POINTS CHANGED FROM OPEN TO CLOSED
                   13 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    10

   MAX. PENETRATION ERROR 1.40677E-09 AT NODE 3445 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -8.20127E+06 AT NODE 5190 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                       143.       TIME AVG. FORCE        143.    
 LARGEST SCALED RESIDUAL FORCE      6.151E+06   AT NODE       3474   DOF  2
  CORRESPONDING RESIDUAL FORCE      6.151E+06
 LARGEST INCREMENT OF DISP.        -4.514E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.        7.298E-08   AT NODE       5156   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 3474 
             IS 184.791E+03   TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13765
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.732e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   22 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   12 POINTS CHANGED FROM OPEN TO CLOSED
                   10 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    11

   MAX. PENETRATION ERROR 78.8334E-12  AT NODE 3452 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -3.25488E+06 AT NODE 5189 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                       57.9       TIME AVG. FORCE        57.9    
 LARGEST SCALED RESIDUAL FORCE      2.450E+06   AT NODE       3473   DOF  2
  CORRESPONDING RESIDUAL FORCE      2.450E+06
 LARGEST INCREMENT OF DISP.        -4.514E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -6.288E-07   AT NODE       3153   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 3473 
             IS 74.6346E+03  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13767
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.727e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   23 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   10 POINTS CHANGED FROM OPEN TO CLOSED
                   13 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    12

   MAX. PENETRATION ERROR -352.690E-12   AT NODE 3450 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -8.37148E+09 AT NODE 5166 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.943E+05   TIME AVG. FORCE       1.943E+05
 LARGEST SCALED RESIDUAL FORCE     -6.628E+09   AT NODE       1448   DOF  2
  CORRESPONDING RESIDUAL FORCE     -6.628E+09
 LARGEST INCREMENT OF DISP.        -4.514E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -6.515E-07   AT NODE       1438   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 3450 
             IS 250.496E+06   TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13764
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.707e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   51 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   36 POINTS CHANGED FROM OPEN TO CLOSED
                   15 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    13

   MAX. PENETRATION ERROR -352.582E-06   AT NODE 5163 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -8.66878E+15 AT NODE 3447 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          PENETRATION ERROR TOO LARGE COMPARED TO DISPLACEMENT INCREMENT.

 AVERAGE FORCE                      1.508E+11   TIME AVG. FORCE       1.508E+11
 LARGEST SCALED RESIDUAL FORCE     -6.502E+15   AT NODE       1445   DOF  2
  CORRESPONDING RESIDUAL FORCE     -6.502E+15
 LARGEST INCREMENT OF DISP.        -4.658E-02   AT NODE       3650   DOF  2
 LARGEST CORRECTION TO DISP.        2.405E-02   AT NODE        876   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 5163 
             IS 119.184E+12   TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     1 STARTS. ATTEMPT NUMBER  3, TIME INCREMENT  1.250E-03
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13830
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.775e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   50 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   50 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 81.7131E-09  AT NODE 5149 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -114.460E+03   AT NODE 3473 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                       2.26       TIME AVG. FORCE        2.26    
 LARGEST SCALED RESIDUAL FORCE     -8.585E+04   AT NODE       1471   DOF  2
  CORRESPONDING RESIDUAL FORCE     -8.585E+04
 LARGEST INCREMENT OF DISP.        -1.129E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -1.129E-02   AT NODE       1602   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 5189 
             IS 11.6007E+03  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13780
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.771e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   48 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   15 POINTS CHANGED FROM OPEN TO CLOSED
                   33 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR 24.0953E-15  AT NODE 3472 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -865.211E-06   AT NODE 5151 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE CONTACT CONSTRAINT ERRORS ARE WITHIN THE TOLERANCES.

 AVERAGE FORCE                      5.482E-03   TIME AVG. FORCE       5.482E-03
 LARGEST SCALED RESIDUAL FORCE     -7.045E-04   AT NODE       1433   DOF  2
  CORRESPONDING RESIDUAL FORCE     -7.045E-04
 LARGEST INCREMENT OF DISP.        -1.129E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -2.213E-06   AT NODE        572   DOF  1
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13762
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.713e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   43 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   26 POINTS CHANGED FROM OPEN TO CLOSED
                   17 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     3

   MAX. PENETRATION ERROR 68.0126E-09  AT NODE 3451 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -239.729       AT NODE 3483 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.343E-02   TIME AVG. FORCE       1.343E-02
 LARGEST SCALED RESIDUAL FORCE       180.       AT NODE       3483   DOF  2
  CORRESPONDING RESIDUAL FORCE       180.    
 LARGEST INCREMENT OF DISP.        -1.129E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -1.189E-06   AT NODE       2581   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13771
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.740e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   28 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   14 POINTS CHANGED FROM OPEN TO CLOSED
                   14 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     4

   MAX. PENETRATION ERROR 50.3185E-12  AT NODE 3466 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -133.568       AT NODE 5167 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.312E-02   TIME AVG. FORCE       1.312E-02
 LARGEST SCALED RESIDUAL FORCE      -111.       AT NODE       1449   DOF  2
  CORRESPONDING RESIDUAL FORCE      -111.    
 LARGEST INCREMENT OF DISP.        -1.129E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.        3.551E-08   AT NODE       5166   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13771
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.749e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   25 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   13 POINTS CHANGED FROM OPEN TO CLOSED
                   12 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     5

   MAX. PENETRATION ERROR 6.93889E-18 AT NODE 5169 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -94.0015      AT NODE 5188 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.102E-02   TIME AVG. FORCE       1.102E-02
 LARGEST SCALED RESIDUAL FORCE      -71.1       AT NODE       1470   DOF  2
  CORRESPONDING RESIDUAL FORCE      -71.1    
 LARGEST INCREMENT OF DISP.        -1.129E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -1.933E-07   AT NODE       2586   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13772
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.749e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   16 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    4 POINTS CHANGED FROM OPEN TO CLOSED
                   12 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     6

   MAX. PENETRATION ERROR -12.1431E-18  AT NODE 3454 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -169.825       AT NODE 5170 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.439E-02   TIME AVG. FORCE       1.439E-02
 LARGEST SCALED RESIDUAL FORCE      -132.       AT NODE       1452   DOF  2
  CORRESPONDING RESIDUAL FORCE      -132.    
 LARGEST INCREMENT OF DISP.        -1.129E-02   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -6.238E-10   AT NODE       1480   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13764
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.739e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   38 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   24 POINTS CHANGED FROM OPEN TO CLOSED
                   14 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     7

   MAX. PENETRATION ERROR 61.3912E-09  AT NODE 3467 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -9.76647E+12 AT NODE 5183 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.699E+08   TIME AVG. FORCE       1.699E+08
 LARGEST SCALED RESIDUAL FORCE     -7.325E+12   AT NODE       1465   DOF  2
  CORRESPONDING RESIDUAL FORCE     -7.325E+12
 LARGEST INCREMENT OF DISP.        -1.130E-02   AT NODE       1600   DOF  2
 LARGEST CORRECTION TO DISP.       -1.236E-04   AT NODE       1466   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 3467 
             IS 874.406E+09   TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     1 STARTS. ATTEMPT NUMBER  4, TIME INCREMENT  3.125E-04
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13830
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.777e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   50 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   50 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 5.10707E-09 AT NODE 3433 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -27.8566E+03  AT NODE 3473 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.645       TIME AVG. FORCE       0.645    
 LARGEST SCALED RESIDUAL FORCE     -2.089E+04   AT NODE       1471   DOF  2
  CORRESPONDING RESIDUAL FORCE     -2.089E+04
 LARGEST INCREMENT OF DISP.        -2.821E-03   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -2.821E-03   AT NODE       1602   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 5189 
             IS 13.2831E+03  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13780
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.771e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   50 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   23 POINTS CHANGED FROM OPEN TO CLOSED
                   27 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR -2.78857E-15 AT NODE 5189 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -649.418       AT NODE 3471 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.266E-02   TIME AVG. FORCE       1.266E-02
 LARGEST SCALED RESIDUAL FORCE      -487.       AT NODE       1469   DOF  2
  CORRESPONDING RESIDUAL FORCE      -487.    
 LARGEST INCREMENT OF DISP.        -2.821E-03   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -1.383E-07   AT NODE       2288   DOF  1
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13776
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.752e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   47 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   20 POINTS CHANGED FROM OPEN TO CLOSED
                   27 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     3

   MAX. PENETRATION ERROR 24.4877E-12  AT NODE 5162 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -267.259       AT NODE 3470 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      8.675E-03   TIME AVG. FORCE       8.675E-03
 LARGEST SCALED RESIDUAL FORCE      -200.       AT NODE       1468   DOF  2
  CORRESPONDING RESIDUAL FORCE      -200.    
 LARGEST INCREMENT OF DISP.        -2.821E-03   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.        3.164E-09   AT NODE       3438   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13769
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.753e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   32 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   17 POINTS CHANGED FROM OPEN TO CLOSED
                   15 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     4

   MAX. PENETRATION ERROR 14.7621E-12  AT NODE 5166 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -15.2977      AT NODE 3476 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      2.509E-03   TIME AVG. FORCE       2.509E-03
 LARGEST SCALED RESIDUAL FORCE      -14.4       AT NODE       1474   DOF  2
  CORRESPONDING RESIDUAL FORCE      -14.4    
 LARGEST INCREMENT OF DISP.        -2.821E-03   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -1.928E-08   AT NODE       3149   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13771
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.753e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   68 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   47 POINTS CHANGED FROM OPEN TO CLOSED
                   21 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     5

   MAX. PENETRATION ERROR 1.24099E-06 AT NODE 3465 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -64.8220E+12  AT NODE 5180 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.158E+09   TIME AVG. FORCE       1.158E+09
 LARGEST SCALED RESIDUAL FORCE     -4.862E+13   AT NODE       1462   DOF  2
  CORRESPONDING RESIDUAL FORCE     -4.862E+13
 LARGEST INCREMENT OF DISP.        -2.949E-03   AT NODE       3562   DOF  2
 LARGEST CORRECTION TO DISP.       -4.462E-04   AT NODE       5484   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 3464 
             IS 21.0007E+12  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     1 STARTS. ATTEMPT NUMBER  5, TIME INCREMENT  7.813E-05
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13830
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.788e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   50 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   50 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 319.192E-12   AT NODE 5149 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -7.24385E+03 AT NODE 3473 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.141       TIME AVG. FORCE       0.141    
 LARGEST SCALED RESIDUAL FORCE     -5.433E+03   AT NODE       1471   DOF  2
  CORRESPONDING RESIDUAL FORCE     -5.433E+03
 LARGEST INCREMENT OF DISP.        -7.053E-04   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -7.053E-04   AT NODE       1602   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 5189 
             IS 11.5725E+03  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13780
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.781e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 3 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   60 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   29 POINTS CHANGED FROM OPEN TO CLOSED
                   31 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR -341.632E-18   AT NODE 3472 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -4.74983E+03 AT NODE 5187 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      8.293E-02   TIME AVG. FORCE       8.293E-02
 LARGEST SCALED RESIDUAL FORCE      3.562E+03   AT NODE       3471   DOF  2
  CORRESPONDING RESIDUAL FORCE      3.562E+03
 LARGEST INCREMENT OF DISP.        -7.053E-04   AT NODE       1602   DOF  2
 LARGEST CORRECTION TO DISP.       -8.642E-09   AT NODE       2288   DOF  1
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13778
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.762e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            1
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   60 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   45 POINTS CHANGED FROM OPEN TO CLOSED
                   15 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     3

   MAX. PENETRATION ERROR -21.2947E-06  AT NODE 5183 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -230.233E+12   AT NODE 3467 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          PENETRATION ERROR TOO LARGE COMPARED TO DISPLACEMENT INCREMENT.

 AVERAGE FORCE                      4.485E+09   TIME AVG. FORCE       4.485E+09
 LARGEST SCALED RESIDUAL FORCE     -1.727E+14   AT NODE       1465   DOF  2
  CORRESPONDING RESIDUAL FORCE     -1.727E+14
 LARGEST INCREMENT OF DISP.        -7.147E-04   AT NODE       1639   DOF  2
 LARGEST CORRECTION TO DISP.        3.116E-04   AT NODE        879   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 5183 
             IS 189.324E+12   TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13808
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.800e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            1
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   50 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    9 POINTS CHANGED FROM OPEN TO CLOSED
                   41 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     4

   MAX. PENETRATION ERROR 1.59117E-03 AT NODE 3458 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -11.3893E+15  AT NODE 5193 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          PENETRATION ERROR TOO LARGE COMPARED TO DISPLACEMENT INCREMENT.

 AVERAGE FORCE                      5.948E+11   TIME AVG. FORCE       5.948E+11
 LARGEST SCALED RESIDUAL FORCE     -9.466E+15   AT NODE       1475   DOF  2
  CORRESPONDING RESIDUAL FORCE     -9.466E+15
 LARGEST INCREMENT OF DISP.         4.780E-02   AT NODE       3478   DOF  2
 LARGEST CORRECTION TO DISP.        4.810E-02   AT NODE       3478   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 3477 
             IS 143.888E+12   TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     13776
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.771e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 33 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.
   ELEMENT 1 IS DISTORTING EXCESSIVELY.
   ELEMENT 1 IS DISTORTING EXCESSIVELY.
   ELEMENT 1 IS DISTORTING EXCESSIVELY.
   ELEMENT 1 IS DISTORTING EXCESSIVELY.
   ELEMENT 2 IS DISTORTING EXCESSIVELY.
   ELEMENT 2 IS DISTORTING EXCESSIVELY.
   ELEMENT 2 IS DISTORTING EXCESSIVELY.
   ELEMENT 2 IS DISTORTING EXCESSIVELY.
   ELEMENT 2 IS DISTORTING EXCESSIVELY.
   ELEMENT 2 IS DISTORTING EXCESSIVELY.
   ELEMENT 2 IS DISTORTING EXCESSIVELY.
   ELEMENT 2 IS DISTORTING EXCESSIVELY.
   ELEMENT 3 IS DISTORTING EXCESSIVELY.
   ELEMENT 3 IS DISTORTING EXCESSIVELY.
   ELEMENT 3 IS DISTORTING EXCESSIVELY.
   ELEMENT 3 IS DISTORTING EXCESSIVELY.
   ELEMENT 3 IS DISTORTING EXCESSIVELY.
   ELEMENT 3 IS DISTORTING EXCESSIVELY.
   ELEMENT 3 IS DISTORTING EXCESSIVELY.
   ELEMENT 3 IS DISTORTING EXCESSIVELY.
   ELEMENT 4 IS DISTORTING EXCESSIVELY.
   ELEMENT 4 IS DISTORTING EXCESSIVELY.
   ELEMENT 4 IS DISTORTING EXCESSIVELY.
   ELEMENT 4 IS DISTORTING EXCESSIVELY.
   ELEMENT 4 IS DISTORTING EXCESSIVELY.
   ELEMENT 4 IS DISTORTING EXCESSIVELY.
   ELEMENT 4 IS DISTORTING EXCESSIVELY.
   ELEMENT 4 IS DISTORTING EXCESSIVELY.
   ELEMENT 287 IS DISTORTING EXCESSIVELY.
   ELEMENT 287 IS DISTORTING EXCESSIVELY.
   ELEMENT 287 IS DISTORTING EXCESSIVELY.
   ELEMENT 287 IS DISTORTING EXCESSIVELY.
   ELEMENT 288 IS DISTORTING EXCESSIVELY.
   ELEMENT 288 IS DISTORTING EXCESSIVELY.
   ELEMENT 288 IS DISTORTING EXCESSIVELY.
   ELEMENT 288 IS DISTORTING EXCESSIVELY.
   ELEMENT 289 IS DISTORTING EXCESSIVELY.
   ELEMENT 289 IS DISTORTING EXCESSIVELY.
   ELEMENT 289 IS DISTORTING EXCESSIVELY.
   ELEMENT 289 IS DISTORTING EXCESSIVELY.
   ELEMENT 290 IS DISTORTING EXCESSIVELY.
   ELEMENT 290 IS DISTORTING EXCESSIVELY.
   ELEMENT 290 IS DISTORTING EXCESSIVELY.
   ELEMENT 290 IS DISTORTING EXCESSIVELY.
   ELEMENT 573 IS DISTORTING EXCESSIVELY.
   ELEMENT 573 IS DISTORTING EXCESSIVELY.
   ELEMENT 573 IS DISTORTING EXCESSIVELY.
   ELEMENT 573 IS DISTORTING EXCESSIVELY.
   ELEMENT 574 IS DISTORTING EXCESSIVELY.
   ELEMENT 574 IS DISTORTING EXCESSIVELY.
   ELEMENT 574 IS DISTORTING EXCESSIVELY.
   ELEMENT 574 IS DISTORTING EXCESSIVELY.
   ELEMENT 575 IS DISTORTING EXCESSIVELY.
   ELEMENT 575 IS DISTORTING EXCESSIVELY.
   ELEMENT 575 IS DISTORTING EXCESSIVELY.
   ELEMENT 575 IS DISTORTING EXCESSIVELY.
   ELEMENT 858 IS DISTORTING EXCESSIVELY.
   ELEMENT 858 IS DISTORTING EXCESSIVELY.
   ELEMENT 858 IS DISTORTING EXCESSIVELY.
   ELEMENT 858 IS DISTORTING EXCESSIVELY.
   ELEMENT 858 IS DISTORTING EXCESSIVELY.
   ELEMENT 858 IS DISTORTING EXCESSIVELY.
   ELEMENT 858 IS DISTORTING EXCESSIVELY.
   ELEMENT 858 IS DISTORTING EXCESSIVELY.
   ELEMENT 860 IS DISTORTING EXCESSIVELY.
   ELEMENT 860 IS DISTORTING EXCESSIVELY.
   ELEMENT 860 IS DISTORTING EXCESSIVELY.
   ELEMENT 860 IS DISTORTING EXCESSIVELY.
   ELEMENT 882 IS DISTORTING EXCESSIVELY.
   ELEMENT 882 IS DISTORTING EXCESSIVELY.
   ELEMENT 883 IS DISTORTING EXCESSIVELY.
   ELEMENT 883 IS DISTORTING EXCESSIVELY.
   ELEMENT 883 IS DISTORTING EXCESSIVELY.
   ELEMENT 883 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1145 IS DISTORTING EXCESSIVELY.
   ELEMENT 1145 IS DISTORTING EXCESSIVELY.
   ELEMENT 1145 IS DISTORTING EXCESSIVELY.
   ELEMENT 1145 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1169 IS DISTORTING EXCESSIVELY.
   ELEMENT 1169 IS DISTORTING EXCESSIVELY.
   ELEMENT 1170 IS DISTORTING EXCESSIVELY.
   ELEMENT 1170 IS DISTORTING EXCESSIVELY.
   ELEMENT 1170 IS DISTORTING EXCESSIVELY.
   ELEMENT 1170 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1181 IS DISTORTING EXCESSIVELY.
   ELEMENT 1181 IS DISTORTING EXCESSIVELY.
   ELEMENT 1181 IS DISTORTING EXCESSIVELY.
   ELEMENT 1181 IS DISTORTING EXCESSIVELY.
   ELEMENT 1182 IS DISTORTING EXCESSIVELY.
   ELEMENT 1182 IS DISTORTING EXCESSIVELY.
   ELEMENT 1182 IS DISTORTING EXCESSIVELY.
   ELEMENT 1182 IS DISTORTING EXCESSIVELY.
   ELEMENT 1183 IS DISTORTING EXCESSIVELY.
   ELEMENT 1183 IS DISTORTING EXCESSIVELY.
   ELEMENT 1183 IS DISTORTING EXCESSIVELY.
   ELEMENT 1183 IS DISTORTING EXCESSIVELY.
   ELEMENT 1184 IS DISTORTING EXCESSIVELY.
   ELEMENT 1184 IS DISTORTING EXCESSIVELY.
   ELEMENT 1185 IS DISTORTING EXCESSIVELY.
   ELEMENT 1185 IS DISTORTING EXCESSIVELY.
   ELEMENT 1185 IS DISTORTING EXCESSIVELY.
   ELEMENT 1185 IS DISTORTING EXCESSIVELY.
   ELEMENT 1186 IS DISTORTING EXCESSIVELY.
   ELEMENT 1186 IS DISTORTING EXCESSIVELY.
   ELEMENT 1186 IS DISTORTING EXCESSIVELY.
   ELEMENT 1186 IS DISTORTING EXCESSIVELY.
   ELEMENT 1456 IS DISTORTING EXCESSIVELY.
   ELEMENT 1456 IS DISTORTING EXCESSIVELY.
   ELEMENT 1462 IS DISTORTING EXCESSIVELY.
   ELEMENT 1462 IS DISTORTING EXCESSIVELY.
   ELEMENT 1463 IS DISTORTING EXCESSIVELY.
   ELEMENT 1463 IS DISTORTING EXCESSIVELY.
   ELEMENT 1463 IS DISTORTING EXCESSIVELY.
   ELEMENT 1463 IS DISTORTING EXCESSIVELY.
   ELEMENT 1464 IS DISTORTING EXCESSIVELY.
   ELEMENT 1464 IS DISTORTING EXCESSIVELY.
   ELEMENT 1464 IS DISTORTING EXCESSIVELY.
   ELEMENT 1464 IS DISTORTING EXCESSIVELY.
   ELEMENT 1464 IS DISTORTING EXCESSIVELY.
   ELEMENT 1464 IS DISTORTING EXCESSIVELY.
   ELEMENT 1465 IS DISTORTING EXCESSIVELY.
   ELEMENT 1465 IS DISTORTING EXCESSIVELY.
   ELEMENT 1465 IS DISTORTING EXCESSIVELY.
   ELEMENT 1465 IS DISTORTING EXCESSIVELY.
   ELEMENT 1465 IS DISTORTING EXCESSIVELY.
   ELEMENT 1465 IS DISTORTING EXCESSIVELY.
   ELEMENT 1465 IS DISTORTING EXCESSIVELY.
   ELEMENT 1465 IS DISTORTING EXCESSIVELY.
   ELEMENT 1466 IS DISTORTING EXCESSIVELY.
   ELEMENT 1466 IS DISTORTING EXCESSIVELY.
   ELEMENT 1466 IS DISTORTING EXCESSIVELY.
   ELEMENT 1466 IS DISTORTING EXCESSIVELY.
   ELEMENT 1467 IS DISTORTING EXCESSIVELY.
   ELEMENT 1467 IS DISTORTING EXCESSIVELY.
   ELEMENT 1467 IS DISTORTING EXCESSIVELY.
   ELEMENT 1467 IS DISTORTING EXCESSIVELY.
   ELEMENT 1467 IS DISTORTING EXCESSIVELY.
   ELEMENT 1467 IS DISTORTING EXCESSIVELY.
   ELEMENT 1468 IS DISTORTING EXCESSIVELY.
   ELEMENT 1468 IS DISTORTING EXCESSIVELY.
   ELEMENT 1468 IS DISTORTING EXCESSIVELY.
   ELEMENT 1468 IS DISTORTING EXCESSIVELY.
   ELEMENT 1469 IS DISTORTING EXCESSIVELY.
   ELEMENT 1469 IS DISTORTING EXCESSIVELY.
   ELEMENT 1470 IS DISTORTING EXCESSIVELY.
   ELEMENT 1470 IS DISTORTING EXCESSIVELY.
   ELEMENT 1470 IS DISTORTING EXCESSIVELY.
   ELEMENT 1470 IS DISTORTING EXCESSIVELY.
   ELEMENT 1471 IS DISTORTING EXCESSIVELY.
   ELEMENT 1471 IS DISTORTING EXCESSIVELY.
   ELEMENT 1471 IS DISTORTING EXCESSIVELY.
   ELEMENT 1471 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1747 IS DISTORTING EXCESSIVELY.
   ELEMENT 1748 IS DISTORTING EXCESSIVELY.
   ELEMENT 1748 IS DISTORTING EXCESSIVELY.
   ELEMENT 1748 IS DISTORTING EXCESSIVELY.
   ELEMENT 1748 IS DISTORTING EXCESSIVELY.
   ELEMENT 1748 IS DISTORTING EXCESSIVELY.
   ELEMENT 1748 IS DISTORTING EXCESSIVELY.
   ELEMENT 1749 IS DISTORTING EXCESSIVELY.
   ELEMENT 1749 IS DISTORTING EXCESSIVELY.
   ELEMENT 1749 IS DISTORTING EXCESSIVELY.
   ELEMENT 1749 IS DISTORTING EXCESSIVELY.
   ELEMENT 1750 IS DISTORTING EXCESSIVELY.
   ELEMENT 1750 IS DISTORTING EXCESSIVELY.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 3433 IS OVERCLOSED BY 485.454E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 3447 IS OVERCLOSED BY 923.379E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 3448 IS OVERCLOSED BY 564.613E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 5149 IS OVERCLOSED BY 485.454E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 5163 IS OVERCLOSED BY 923.379E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 5164 IS OVERCLOSED BY 564.613E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 188 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.
 

 ***NOTE: SEVERE CONTACT OVERCLOSURES EXIST. CONVERGENCE IS JUDGED UNLIKELY.
 

 ***ERROR: TOO MANY ATTEMPTS MADE FOR THIS INCREMENT



     ANALYSIS SUMMARY:
     TOTAL OF          1  INCREMENTS
                       5  CUTBACKS IN AUTOMATIC INCREMENTATION
                      32  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                      32  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                      32  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       2  WARNING MESSAGES DURING USER INPUT PROCESSING
                     520  WARNING MESSAGES DURING ANALYSIS
                     484  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                      19  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       1  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =      18.    
       SYSTEM TIME (SEC)    =      5.3    
       TOTAL CPU TIME (SEC) =      24.    
       WALLCLOCK TIME (SEC) =           33
       MEMORY PEAK (GB)     =            0

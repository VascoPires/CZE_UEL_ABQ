1

   Abaqus 2023                                  Date 25-Oct-2026   Time 18:25:13
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

 CONVERGENCE TOLERANCE PARAMETERS FOR MOMENT   
     CRITERION FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR ROTATION CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE MOMENT                              1.000E-02
     AVERAGE MOMENT    IS TIME AVERAGE MOMENT   
     ALTERNATE CRIT. FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO MOMENT    RELATIVE TO TIME AVRG. MOMENT        1.000E-05
     CRITERION FOR RESIDUAL MOMENT    WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR ROTATION CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL MOMENT    FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO MOMENT    REL. TO TIME AVRG. MAX. MOMENT       1.000E-05

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

     CHARACTERISTIC ELEMENT LENGTH     0.659    

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.000E-02
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10938
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.416e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2746.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2746. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2759.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2759. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2725.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2725. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2750.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2750. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2749.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2749. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2748.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2748. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2747.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2747. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2752.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2752. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2751.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2751. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2745.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2745. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2744.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2744. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2743.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2743. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2761.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2761. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2760.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2760. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2758.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2758. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2757.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2757. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2763.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2763. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2762.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2762. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2753.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2753. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2756.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2756. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2755.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2755. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2754.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2754. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2738.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2738. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2737.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2737. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2731.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2731. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2730.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2730. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2732.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2732. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2736.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2736. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2735.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2735. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2734.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2734. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2729.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2729. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2719.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2719. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2718.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2718. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2717.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2717. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2720.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2720. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2716.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2716. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2715.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2715. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2722.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2722. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2724.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2724. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2723.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2723. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2728.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2728. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2727.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2727. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2726.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2726. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2742.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2742. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2741.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2741. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2740.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2740. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2739.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2739. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2714.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2714. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: SOLVER PROBLEM.  ZERO PIVOT WHEN PROCESSING A (TIED) CONTACT 
             CONSTRAINT D.O.F. 1 FOR SECONDARY NODE 2713.  THIS MAY BE DUE TO 
             THE UNIT USED FOR THE DISPLACEMENT DIFFERS BY MORE THAN 10 ORDER 
             OF MAGNITUDE THAN THE UNIT USED FOR THE MATERIAL STIFFNESS.  IF 
             THIS IS THE CASE, PLEASE USE A DIFFERENT UNIT SYSTEM.
 
 OVERCONSTRAINT CHECKS: 
 An overconstraint was detected at one of the Lagrange multipliers associated 
 with contact secondary node 2713. There are multiple constraints applied directly 
 or chained constraints that are applied indirectly to this node. 
 A list of nodes and chained constraints between these nodes that most likely lead 
 to the detected overconstraint can be printed out to the message file by including 
 *constraint controls, print=yes in the input file.


 ***WARNING: Solver problem. Zero pivot when processing D.O.F. 1 of 49 nodes. 
             The nodes have been identified in node set 
             WarnNodeSolvProbZeroPiv_1_1_1_1_1.

                   48 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   48 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 6.13482E-06 AT NODE 4071 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -2.05189E+03 AT NODE 2730 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.409       TIME AVG. FORCE       0.409    
 LARGEST SCALED RESIDUAL FORCE      1.630E+03   AT NODE       2730   DOF  2
  CORRESPONDING RESIDUAL FORCE      1.630E+03
 LARGEST INCREMENT OF DISP.        -0.180       AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -0.180       AT NODE       5427   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.270    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     6.382E-04   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     6.382E-04
 LARGEST INCREMENT OF ROTATION     -2.620E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -2.620E-03   AT NODE       5425   DOF  6
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10890
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.472e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

 ***WARNING: OVERCONSTRAINT CHECKS: THE MAXIMUM NUMBER OF ZERO PIVOT-BASED 
             OVERCONSTRAINT CHECKS MESSAGES PRINTED FOR THIS STEP HAS BEEN 
             REACHED. THE OUTPUT OF THESE MESSAGES IS SUPRESSED UNTIL THE END 
             OF THE STEP TO AVOID POTENTIALLY LARGE INCREASES IN THE SYSTEM 
             TIME NEEDED TO COMPLETE THE ANALYSIS.

                   28 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    8 POINTS CHANGED FROM OPEN TO CLOSED
                   20 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR 122.876E-15   AT NODE 2713 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -859.192       AT NODE 4102 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      7.423E-02   TIME AVG. FORCE       7.423E-02
 LARGEST SCALED RESIDUAL FORCE       644.       AT NODE       2746   DOF  2
  CORRESPONDING RESIDUAL FORCE       644.    
 LARGEST INCREMENT OF DISP.        -0.180       AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -1.671E-04   AT NODE       2712   DOF  1
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      4.891E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -1.556E-04   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -1.556E-04
 LARGEST INCREMENT OF ROTATION     -2.620E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.197E-07   AT NODE       5425   DOF  6
          ALL ROTATION CORRECTIONS ARE ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10878
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.437e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   33 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   18 POINTS CHANGED FROM OPEN TO CLOSED
                   15 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     3

   MAX. PENETRATION ERROR 1.77519E-09 AT NODE 2743 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -1.79432E+03 AT NODE 4073 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.195       TIME AVG. FORCE       0.195    
 LARGEST SCALED RESIDUAL FORCE     -1.669E+03   AT NODE       1135   DOF  2
  CORRESPONDING RESIDUAL FORCE     -1.669E+03
 LARGEST INCREMENT OF DISP.        -0.180       AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.265E-06   AT NODE       1143   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.128    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     5.198E-04   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     5.198E-04
 LARGEST INCREMENT OF ROTATION     -2.620E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.168E-07   AT NODE       5425   DOF  6
          ALL ROTATION CORRECTIONS ARE ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10881
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.434e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   40 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   24 POINTS CHANGED FROM OPEN TO CLOSED
                   16 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     4

   MAX. PENETRATION ERROR 5.63307E-09 AT NODE 2748 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -2.35517E+03 AT NODE 4080 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.242       TIME AVG. FORCE       0.242    
 LARGEST SCALED RESIDUAL FORCE      2.000E+03   AT NODE       2724   DOF  2
  CORRESPONDING RESIDUAL FORCE      2.000E+03
 LARGEST INCREMENT OF DISP.        -0.180       AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.520E-07   AT NODE       2574   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.159    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     6.194E-07   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     6.194E-07
 LARGEST INCREMENT OF ROTATION     -2.620E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     5.467E-09   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     1 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  5.000E-03
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10938
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.490e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   44 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   44 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 383.426E-09   AT NODE 4071 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -2.80611E+03 AT NODE 2738 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.287       TIME AVG. FORCE       0.287    
 LARGEST SCALED RESIDUAL FORCE      2.233E+03   AT NODE       2738   DOF  2
  CORRESPONDING RESIDUAL FORCE      2.233E+03
 LARGEST INCREMENT OF DISP.        -4.500E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -4.500E-02   AT NODE       5427   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.189    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     1.990E-05   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     1.990E-05
 LARGEST INCREMENT OF ROTATION     -6.549E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -6.549E-04   AT NODE       5425   DOF  6
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10894
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.475e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   58 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   30 POINTS CHANGED FROM OPEN TO CLOSED
                   28 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR 284.448E-06   AT NODE 2756 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -10.3343E+15  AT NODE 4112 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          PENETRATION ERROR TOO LARGE COMPARED TO DISPLACEMENT INCREMENT.

 AVERAGE FORCE                      2.605E+11   TIME AVG. FORCE       2.605E+11
 LARGEST SCALED RESIDUAL FORCE      7.751E+15   AT NODE       2756   DOF  2
  CORRESPONDING RESIDUAL FORCE      7.751E+15
 LARGEST INCREMENT OF DISP.        -4.500E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        5.955E-02   AT NODE       1155   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 2756 
             IS 96.7587E+12  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.717E+11
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -3.011E-03   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -3.011E-03
 LARGEST INCREMENT OF ROTATION      2.992E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     3.647E-03   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10896
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.464e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.

     NEGATIVE EIGENVALUES MEAN THAT THE SYSTEM MATRIX IS NOT 
     POSITIVE DEFINITE:
     FOR EXAMPLE, A BIFURCATION (BUCKLING) LOAD MAY HAVE BEEN EXCEEDED.
   ELEMENT 7 IS DISTORTING EXCESSIVELY.
   ELEMENT 7 IS DISTORTING EXCESSIVELY.
   ELEMENT 7 IS DISTORTING EXCESSIVELY.
   ELEMENT 7 IS DISTORTING EXCESSIVELY.
   ELEMENT 8 IS DISTORTING EXCESSIVELY.
   ELEMENT 8 IS DISTORTING EXCESSIVELY.
   ELEMENT 8 IS DISTORTING EXCESSIVELY.
   ELEMENT 8 IS DISTORTING EXCESSIVELY.
   ELEMENT 8 IS DISTORTING EXCESSIVELY.
   ELEMENT 8 IS DISTORTING EXCESSIVELY.
   ELEMENT 8 IS DISTORTING EXCESSIVELY.
   ELEMENT 8 IS DISTORTING EXCESSIVELY.
   ELEMENT 9 IS DISTORTING EXCESSIVELY.
   ELEMENT 9 IS DISTORTING EXCESSIVELY.
   ELEMENT 9 IS DISTORTING EXCESSIVELY.
   ELEMENT 9 IS DISTORTING EXCESSIVELY.
   ELEMENT 29 IS DISTORTING EXCESSIVELY.
   ELEMENT 29 IS DISTORTING EXCESSIVELY.
   ELEMENT 29 IS DISTORTING EXCESSIVELY.
   ELEMENT 29 IS DISTORTING EXCESSIVELY.
   ELEMENT 29 IS DISTORTING EXCESSIVELY.
   ELEMENT 29 IS DISTORTING EXCESSIVELY.
   ELEMENT 29 IS DISTORTING EXCESSIVELY.
   ELEMENT 29 IS DISTORTING EXCESSIVELY.
   ELEMENT 30 IS DISTORTING EXCESSIVELY.
   ELEMENT 30 IS DISTORTING EXCESSIVELY.
   ELEMENT 30 IS DISTORTING EXCESSIVELY.
   ELEMENT 30 IS DISTORTING EXCESSIVELY.
   ELEMENT 30 IS DISTORTING EXCESSIVELY.
   ELEMENT 30 IS DISTORTING EXCESSIVELY.
   ELEMENT 44 IS DISTORTING EXCESSIVELY.
   ELEMENT 44 IS DISTORTING EXCESSIVELY.
   ELEMENT 45 IS DISTORTING EXCESSIVELY.
   ELEMENT 45 IS DISTORTING EXCESSIVELY.
   ELEMENT 45 IS DISTORTING EXCESSIVELY.
   ELEMENT 45 IS DISTORTING EXCESSIVELY.
   ELEMENT 45 IS DISTORTING EXCESSIVELY.
   ELEMENT 45 IS DISTORTING EXCESSIVELY.
   ELEMENT 45 IS DISTORTING EXCESSIVELY.
   ELEMENT 45 IS DISTORTING EXCESSIVELY.
   ELEMENT 46 IS DISTORTING EXCESSIVELY.
   ELEMENT 46 IS DISTORTING EXCESSIVELY.
   ELEMENT 46 IS DISTORTING EXCESSIVELY.
   ELEMENT 46 IS DISTORTING EXCESSIVELY.
   ELEMENT 46 IS DISTORTING EXCESSIVELY.
   ELEMENT 46 IS DISTORTING EXCESSIVELY.
   ELEMENT 46 IS DISTORTING EXCESSIVELY.
   ELEMENT 46 IS DISTORTING EXCESSIVELY.
   ELEMENT 47 IS DISTORTING EXCESSIVELY.
   ELEMENT 47 IS DISTORTING EXCESSIVELY.
   ELEMENT 47 IS DISTORTING EXCESSIVELY.
   ELEMENT 47 IS DISTORTING EXCESSIVELY.
   ELEMENT 47 IS DISTORTING EXCESSIVELY.
   ELEMENT 47 IS DISTORTING EXCESSIVELY.
   ELEMENT 47 IS DISTORTING EXCESSIVELY.
   ELEMENT 47 IS DISTORTING EXCESSIVELY.
   ELEMENT 48 IS DISTORTING EXCESSIVELY.
   ELEMENT 48 IS DISTORTING EXCESSIVELY.
   ELEMENT 48 IS DISTORTING EXCESSIVELY.
   ELEMENT 48 IS DISTORTING EXCESSIVELY.
   ELEMENT 48 IS DISTORTING EXCESSIVELY.
   ELEMENT 48 IS DISTORTING EXCESSIVELY.
   ELEMENT 48 IS DISTORTING EXCESSIVELY.
   ELEMENT 48 IS DISTORTING EXCESSIVELY.
   ELEMENT 232 IS DISTORTING EXCESSIVELY.
   ELEMENT 232 IS DISTORTING EXCESSIVELY.
   ELEMENT 233 IS DISTORTING EXCESSIVELY.
   ELEMENT 233 IS DISTORTING EXCESSIVELY.
   ELEMENT 233 IS DISTORTING EXCESSIVELY.
   ELEMENT 233 IS DISTORTING EXCESSIVELY.
   ELEMENT 233 IS DISTORTING EXCESSIVELY.
   ELEMENT 233 IS DISTORTING EXCESSIVELY.
   ELEMENT 233 IS DISTORTING EXCESSIVELY.
   ELEMENT 233 IS DISTORTING EXCESSIVELY.
   ELEMENT 234 IS DISTORTING EXCESSIVELY.
   ELEMENT 234 IS DISTORTING EXCESSIVELY.
   ELEMENT 234 IS DISTORTING EXCESSIVELY.
   ELEMENT 234 IS DISTORTING EXCESSIVELY.
   ELEMENT 251 IS DISTORTING EXCESSIVELY.
   ELEMENT 251 IS DISTORTING EXCESSIVELY.
   ELEMENT 251 IS DISTORTING EXCESSIVELY.
   ELEMENT 251 IS DISTORTING EXCESSIVELY.
   ELEMENT 252 IS DISTORTING EXCESSIVELY.
   ELEMENT 252 IS DISTORTING EXCESSIVELY.
   ELEMENT 252 IS DISTORTING EXCESSIVELY.
   ELEMENT 252 IS DISTORTING EXCESSIVELY.
   ELEMENT 252 IS DISTORTING EXCESSIVELY.
   ELEMENT 252 IS DISTORTING EXCESSIVELY.
   ELEMENT 252 IS DISTORTING EXCESSIVELY.
   ELEMENT 252 IS DISTORTING EXCESSIVELY.
   ELEMENT 253 IS DISTORTING EXCESSIVELY.
   ELEMENT 253 IS DISTORTING EXCESSIVELY.
   ELEMENT 253 IS DISTORTING EXCESSIVELY.
   ELEMENT 253 IS DISTORTING EXCESSIVELY.
   ELEMENT 253 IS DISTORTING EXCESSIVELY.
   ELEMENT 253 IS DISTORTING EXCESSIVELY.
   ELEMENT 253 IS DISTORTING EXCESSIVELY.
   ELEMENT 253 IS DISTORTING EXCESSIVELY.
   ELEMENT 254 IS DISTORTING EXCESSIVELY.
   ELEMENT 254 IS DISTORTING EXCESSIVELY.
   ELEMENT 254 IS DISTORTING EXCESSIVELY.
   ELEMENT 254 IS DISTORTING EXCESSIVELY.
   ELEMENT 254 IS DISTORTING EXCESSIVELY.
   ELEMENT 254 IS DISTORTING EXCESSIVELY.
   ELEMENT 254 IS DISTORTING EXCESSIVELY.
   ELEMENT 254 IS DISTORTING EXCESSIVELY.
   ELEMENT 255 IS DISTORTING EXCESSIVELY.
   ELEMENT 255 IS DISTORTING EXCESSIVELY.
   ELEMENT 255 IS DISTORTING EXCESSIVELY.
   ELEMENT 255 IS DISTORTING EXCESSIVELY.
   ELEMENT 255 IS DISTORTING EXCESSIVELY.
   ELEMENT 255 IS DISTORTING EXCESSIVELY.
   ELEMENT 255 IS DISTORTING EXCESSIVELY.
   ELEMENT 255 IS DISTORTING EXCESSIVELY.
   ELEMENT 269 IS DISTORTING EXCESSIVELY.
   ELEMENT 269 IS DISTORTING EXCESSIVELY.
   ELEMENT 269 IS DISTORTING EXCESSIVELY.
   ELEMENT 269 IS DISTORTING EXCESSIVELY.
   ELEMENT 269 IS DISTORTING EXCESSIVELY.
   ELEMENT 269 IS DISTORTING EXCESSIVELY.
   ELEMENT 270 IS DISTORTING EXCESSIVELY.
   ELEMENT 270 IS DISTORTING EXCESSIVELY.
   ELEMENT 270 IS DISTORTING EXCESSIVELY.
   ELEMENT 270 IS DISTORTING EXCESSIVELY.
   ELEMENT 270 IS DISTORTING EXCESSIVELY.
   ELEMENT 270 IS DISTORTING EXCESSIVELY.
   ELEMENT 270 IS DISTORTING EXCESSIVELY.
   ELEMENT 270 IS DISTORTING EXCESSIVELY.
   ELEMENT 271 IS DISTORTING EXCESSIVELY.
   ELEMENT 271 IS DISTORTING EXCESSIVELY.
   ELEMENT 271 IS DISTORTING EXCESSIVELY.
   ELEMENT 271 IS DISTORTING EXCESSIVELY.
   ELEMENT 271 IS DISTORTING EXCESSIVELY.
   ELEMENT 271 IS DISTORTING EXCESSIVELY.
   ELEMENT 271 IS DISTORTING EXCESSIVELY.
   ELEMENT 271 IS DISTORTING EXCESSIVELY.
   ELEMENT 272 IS DISTORTING EXCESSIVELY.
   ELEMENT 272 IS DISTORTING EXCESSIVELY.
   ELEMENT 272 IS DISTORTING EXCESSIVELY.
   ELEMENT 272 IS DISTORTING EXCESSIVELY.
   ELEMENT 272 IS DISTORTING EXCESSIVELY.
   ELEMENT 272 IS DISTORTING EXCESSIVELY.
   ELEMENT 272 IS DISTORTING EXCESSIVELY.
   ELEMENT 272 IS DISTORTING EXCESSIVELY.
   ELEMENT 273 IS DISTORTING EXCESSIVELY.
   ELEMENT 273 IS DISTORTING EXCESSIVELY.
   ELEMENT 273 IS DISTORTING EXCESSIVELY.
   ELEMENT 273 IS DISTORTING EXCESSIVELY.
   ELEMENT 273 IS DISTORTING EXCESSIVELY.
   ELEMENT 273 IS DISTORTING EXCESSIVELY.
   ELEMENT 273 IS DISTORTING EXCESSIVELY.
   ELEMENT 273 IS DISTORTING EXCESSIVELY.
   ELEMENT 458 IS DISTORTING EXCESSIVELY.
   ELEMENT 458 IS DISTORTING EXCESSIVELY.
   ELEMENT 458 IS DISTORTING EXCESSIVELY.
   ELEMENT 458 IS DISTORTING EXCESSIVELY.
   ELEMENT 458 IS DISTORTING EXCESSIVELY.
   ELEMENT 458 IS DISTORTING EXCESSIVELY.
   ELEMENT 458 IS DISTORTING EXCESSIVELY.
   ELEMENT 458 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 474 IS DISTORTING EXCESSIVELY.
   ELEMENT 474 IS DISTORTING EXCESSIVELY.
   ELEMENT 475 IS DISTORTING EXCESSIVELY.
   ELEMENT 475 IS DISTORTING EXCESSIVELY.
   ELEMENT 475 IS DISTORTING EXCESSIVELY.
   ELEMENT 475 IS DISTORTING EXCESSIVELY.
   ELEMENT 475 IS DISTORTING EXCESSIVELY.
   ELEMENT 475 IS DISTORTING EXCESSIVELY.
   ELEMENT 475 IS DISTORTING EXCESSIVELY.
   ELEMENT 475 IS DISTORTING EXCESSIVELY.
   ELEMENT 476 IS DISTORTING EXCESSIVELY.
   ELEMENT 476 IS DISTORTING EXCESSIVELY.
   ELEMENT 476 IS DISTORTING EXCESSIVELY.
   ELEMENT 476 IS DISTORTING EXCESSIVELY.
   ELEMENT 476 IS DISTORTING EXCESSIVELY.
   ELEMENT 476 IS DISTORTING EXCESSIVELY.
   ELEMENT 476 IS DISTORTING EXCESSIVELY.
   ELEMENT 476 IS DISTORTING EXCESSIVELY.
   ELEMENT 477 IS DISTORTING EXCESSIVELY.
   ELEMENT 477 IS DISTORTING EXCESSIVELY.
   ELEMENT 477 IS DISTORTING EXCESSIVELY.
   ELEMENT 477 IS DISTORTING EXCESSIVELY.
   ELEMENT 477 IS DISTORTING EXCESSIVELY.
   ELEMENT 477 IS DISTORTING EXCESSIVELY.
   ELEMENT 477 IS DISTORTING EXCESSIVELY.
   ELEMENT 477 IS DISTORTING EXCESSIVELY.
   ELEMENT 478 IS DISTORTING EXCESSIVELY.
   ELEMENT 478 IS DISTORTING EXCESSIVELY.
   ELEMENT 478 IS DISTORTING EXCESSIVELY.
   ELEMENT 478 IS DISTORTING EXCESSIVELY.
   ELEMENT 478 IS DISTORTING EXCESSIVELY.
   ELEMENT 478 IS DISTORTING EXCESSIVELY.
   ELEMENT 478 IS DISTORTING EXCESSIVELY.
   ELEMENT 478 IS DISTORTING EXCESSIVELY.
   ELEMENT 479 IS DISTORTING EXCESSIVELY.
   ELEMENT 479 IS DISTORTING EXCESSIVELY.
   ELEMENT 479 IS DISTORTING EXCESSIVELY.
   ELEMENT 479 IS DISTORTING EXCESSIVELY.
   ELEMENT 479 IS DISTORTING EXCESSIVELY.
   ELEMENT 479 IS DISTORTING EXCESSIVELY.
   ELEMENT 479 IS DISTORTING EXCESSIVELY.
   ELEMENT 479 IS DISTORTING EXCESSIVELY.
   ELEMENT 480 IS DISTORTING EXCESSIVELY.
   ELEMENT 480 IS DISTORTING EXCESSIVELY.
   ELEMENT 480 IS DISTORTING EXCESSIVELY.
   ELEMENT 480 IS DISTORTING EXCESSIVELY.
   ELEMENT 480 IS DISTORTING EXCESSIVELY.
   ELEMENT 480 IS DISTORTING EXCESSIVELY.
   ELEMENT 480 IS DISTORTING EXCESSIVELY.
   ELEMENT 480 IS DISTORTING EXCESSIVELY.
   ELEMENT 481 IS DISTORTING EXCESSIVELY.
   ELEMENT 481 IS DISTORTING EXCESSIVELY.
   ELEMENT 481 IS DISTORTING EXCESSIVELY.
   ELEMENT 481 IS DISTORTING EXCESSIVELY.
   ELEMENT 494 IS DISTORTING EXCESSIVELY.
   ELEMENT 494 IS DISTORTING EXCESSIVELY.
   ELEMENT 494 IS DISTORTING EXCESSIVELY.
   ELEMENT 494 IS DISTORTING EXCESSIVELY.
   ELEMENT 495 IS DISTORTING EXCESSIVELY.
   ELEMENT 495 IS DISTORTING EXCESSIVELY.
   ELEMENT 495 IS DISTORTING EXCESSIVELY.
   ELEMENT 495 IS DISTORTING EXCESSIVELY.
   ELEMENT 495 IS DISTORTING EXCESSIVELY.
   ELEMENT 495 IS DISTORTING EXCESSIVELY.
   ELEMENT 495 IS DISTORTING EXCESSIVELY.
   ELEMENT 495 IS DISTORTING EXCESSIVELY.
   ELEMENT 496 IS DISTORTING EXCESSIVELY.
   ELEMENT 496 IS DISTORTING EXCESSIVELY.
   ELEMENT 496 IS DISTORTING EXCESSIVELY.
   ELEMENT 496 IS DISTORTING EXCESSIVELY.
   ELEMENT 496 IS DISTORTING EXCESSIVELY.
   ELEMENT 496 IS DISTORTING EXCESSIVELY.
   ELEMENT 496 IS DISTORTING EXCESSIVELY.
   ELEMENT 496 IS DISTORTING EXCESSIVELY.
   ELEMENT 497 IS DISTORTING EXCESSIVELY.
   ELEMENT 497 IS DISTORTING EXCESSIVELY.
   ELEMENT 497 IS DISTORTING EXCESSIVELY.
   ELEMENT 497 IS DISTORTING EXCESSIVELY.
   ELEMENT 497 IS DISTORTING EXCESSIVELY.
   ELEMENT 497 IS DISTORTING EXCESSIVELY.
   ELEMENT 497 IS DISTORTING EXCESSIVELY.
   ELEMENT 497 IS DISTORTING EXCESSIVELY.
   ELEMENT 498 IS DISTORTING EXCESSIVELY.
   ELEMENT 498 IS DISTORTING EXCESSIVELY.
   ELEMENT 498 IS DISTORTING EXCESSIVELY.
   ELEMENT 498 IS DISTORTING EXCESSIVELY.
   ELEMENT 498 IS DISTORTING EXCESSIVELY.
   ELEMENT 498 IS DISTORTING EXCESSIVELY.
   ELEMENT 498 IS DISTORTING EXCESSIVELY.
   ELEMENT 498 IS DISTORTING EXCESSIVELY.
   ELEMENT 683 IS DISTORTING EXCESSIVELY.
   ELEMENT 683 IS DISTORTING EXCESSIVELY.
   ELEMENT 683 IS DISTORTING EXCESSIVELY.
   ELEMENT 683 IS DISTORTING EXCESSIVELY.
   ELEMENT 683 IS DISTORTING EXCESSIVELY.
   ELEMENT 683 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 699 IS DISTORTING EXCESSIVELY.
   ELEMENT 699 IS DISTORTING EXCESSIVELY.
   ELEMENT 699 IS DISTORTING EXCESSIVELY.
   ELEMENT 699 IS DISTORTING EXCESSIVELY.
   ELEMENT 699 IS DISTORTING EXCESSIVELY.
   ELEMENT 699 IS DISTORTING EXCESSIVELY.
   ELEMENT 700 IS DISTORTING EXCESSIVELY.
   ELEMENT 700 IS DISTORTING EXCESSIVELY.
   ELEMENT 700 IS DISTORTING EXCESSIVELY.
   ELEMENT 700 IS DISTORTING EXCESSIVELY.
   ELEMENT 700 IS DISTORTING EXCESSIVELY.
   ELEMENT 700 IS DISTORTING EXCESSIVELY.
   ELEMENT 700 IS DISTORTING EXCESSIVELY.
   ELEMENT 700 IS DISTORTING EXCESSIVELY.
   ELEMENT 701 IS DISTORTING EXCESSIVELY.
   ELEMENT 701 IS DISTORTING EXCESSIVELY.
   ELEMENT 701 IS DISTORTING EXCESSIVELY.
   ELEMENT 701 IS DISTORTING EXCESSIVELY.
   ELEMENT 701 IS DISTORTING EXCESSIVELY.
   ELEMENT 701 IS DISTORTING EXCESSIVELY.
   ELEMENT 701 IS DISTORTING EXCESSIVELY.
   ELEMENT 701 IS DISTORTING EXCESSIVELY.
   ELEMENT 702 IS DISTORTING EXCESSIVELY.
   ELEMENT 702 IS DISTORTING EXCESSIVELY.
   ELEMENT 702 IS DISTORTING EXCESSIVELY.
   ELEMENT 702 IS DISTORTING EXCESSIVELY.
   ELEMENT 702 IS DISTORTING EXCESSIVELY.
   ELEMENT 702 IS DISTORTING EXCESSIVELY.
   ELEMENT 702 IS DISTORTING EXCESSIVELY.
   ELEMENT 702 IS DISTORTING EXCESSIVELY.
   ELEMENT 703 IS DISTORTING EXCESSIVELY.
   ELEMENT 703 IS DISTORTING EXCESSIVELY.
   ELEMENT 703 IS DISTORTING EXCESSIVELY.
   ELEMENT 703 IS DISTORTING EXCESSIVELY.
   ELEMENT 703 IS DISTORTING EXCESSIVELY.
   ELEMENT 703 IS DISTORTING EXCESSIVELY.
   ELEMENT 703 IS DISTORTING EXCESSIVELY.
   ELEMENT 703 IS DISTORTING EXCESSIVELY.
   ELEMENT 704 IS DISTORTING EXCESSIVELY.
   ELEMENT 704 IS DISTORTING EXCESSIVELY.
   ELEMENT 704 IS DISTORTING EXCESSIVELY.
   ELEMENT 704 IS DISTORTING EXCESSIVELY.
   ELEMENT 704 IS DISTORTING EXCESSIVELY.
   ELEMENT 704 IS DISTORTING EXCESSIVELY.
   ELEMENT 704 IS DISTORTING EXCESSIVELY.
   ELEMENT 704 IS DISTORTING EXCESSIVELY.
   ELEMENT 705 IS DISTORTING EXCESSIVELY.
   ELEMENT 705 IS DISTORTING EXCESSIVELY.
   ELEMENT 705 IS DISTORTING EXCESSIVELY.
   ELEMENT 705 IS DISTORTING EXCESSIVELY.
   ELEMENT 705 IS DISTORTING EXCESSIVELY.
   ELEMENT 705 IS DISTORTING EXCESSIVELY.
   ELEMENT 705 IS DISTORTING EXCESSIVELY.
   ELEMENT 705 IS DISTORTING EXCESSIVELY.
   ELEMENT 706 IS DISTORTING EXCESSIVELY.
   ELEMENT 706 IS DISTORTING EXCESSIVELY.
   ELEMENT 706 IS DISTORTING EXCESSIVELY.
   ELEMENT 706 IS DISTORTING EXCESSIVELY.
   ELEMENT 706 IS DISTORTING EXCESSIVELY.
   ELEMENT 706 IS DISTORTING EXCESSIVELY.
   ELEMENT 707 IS DISTORTING EXCESSIVELY.
   ELEMENT 707 IS DISTORTING EXCESSIVELY.
   ELEMENT 707 IS DISTORTING EXCESSIVELY.
   ELEMENT 707 IS DISTORTING EXCESSIVELY.
   ELEMENT 716 IS DISTORTING EXCESSIVELY.
   ELEMENT 716 IS DISTORTING EXCESSIVELY.
   ELEMENT 717 IS DISTORTING EXCESSIVELY.
   ELEMENT 717 IS DISTORTING EXCESSIVELY.
   ELEMENT 717 IS DISTORTING EXCESSIVELY.
   ELEMENT 717 IS DISTORTING EXCESSIVELY.
   ELEMENT 717 IS DISTORTING EXCESSIVELY.
   ELEMENT 717 IS DISTORTING EXCESSIVELY.
   ELEMENT 718 IS DISTORTING EXCESSIVELY.
   ELEMENT 718 IS DISTORTING EXCESSIVELY.
   ELEMENT 718 IS DISTORTING EXCESSIVELY.
   ELEMENT 718 IS DISTORTING EXCESSIVELY.
   ELEMENT 718 IS DISTORTING EXCESSIVELY.
   ELEMENT 718 IS DISTORTING EXCESSIVELY.
   ELEMENT 718 IS DISTORTING EXCESSIVELY.
   ELEMENT 718 IS DISTORTING EXCESSIVELY.
   ELEMENT 719 IS DISTORTING EXCESSIVELY.
   ELEMENT 719 IS DISTORTING EXCESSIVELY.
   ELEMENT 719 IS DISTORTING EXCESSIVELY.
   ELEMENT 719 IS DISTORTING EXCESSIVELY.
   ELEMENT 719 IS DISTORTING EXCESSIVELY.
   ELEMENT 719 IS DISTORTING EXCESSIVELY.
   ELEMENT 719 IS DISTORTING EXCESSIVELY.
   ELEMENT 719 IS DISTORTING EXCESSIVELY.
   ELEMENT 720 IS DISTORTING EXCESSIVELY.
   ELEMENT 720 IS DISTORTING EXCESSIVELY.
   ELEMENT 720 IS DISTORTING EXCESSIVELY.
   ELEMENT 720 IS DISTORTING EXCESSIVELY.
   ELEMENT 720 IS DISTORTING EXCESSIVELY.
   ELEMENT 720 IS DISTORTING EXCESSIVELY.
   ELEMENT 720 IS DISTORTING EXCESSIVELY.
   ELEMENT 720 IS DISTORTING EXCESSIVELY.
   ELEMENT 721 IS DISTORTING EXCESSIVELY.
   ELEMENT 721 IS DISTORTING EXCESSIVELY.
   ELEMENT 721 IS DISTORTING EXCESSIVELY.
   ELEMENT 721 IS DISTORTING EXCESSIVELY.
   ELEMENT 721 IS DISTORTING EXCESSIVELY.
   ELEMENT 721 IS DISTORTING EXCESSIVELY.
   ELEMENT 721 IS DISTORTING EXCESSIVELY.
   ELEMENT 721 IS DISTORTING EXCESSIVELY.
   ELEMENT 722 IS DISTORTING EXCESSIVELY.
   ELEMENT 722 IS DISTORTING EXCESSIVELY.
   ELEMENT 722 IS DISTORTING EXCESSIVELY.
   ELEMENT 722 IS DISTORTING EXCESSIVELY.
   ELEMENT 722 IS DISTORTING EXCESSIVELY.
   ELEMENT 722 IS DISTORTING EXCESSIVELY.
   ELEMENT 723 IS DISTORTING EXCESSIVELY.
   ELEMENT 723 IS DISTORTING EXCESSIVELY.
   ELEMENT 723 IS DISTORTING EXCESSIVELY.
   ELEMENT 723 IS DISTORTING EXCESSIVELY.
   ELEMENT 723 IS DISTORTING EXCESSIVELY.
   ELEMENT 723 IS DISTORTING EXCESSIVELY.
   ELEMENT 723 IS DISTORTING EXCESSIVELY.
   ELEMENT 723 IS DISTORTING EXCESSIVELY.
   ELEMENT 724 IS DISTORTING EXCESSIVELY.
   ELEMENT 724 IS DISTORTING EXCESSIVELY.
   ELEMENT 724 IS DISTORTING EXCESSIVELY.
   ELEMENT 724 IS DISTORTING EXCESSIVELY.
   ELEMENT 724 IS DISTORTING EXCESSIVELY.
   ELEMENT 724 IS DISTORTING EXCESSIVELY.
   ELEMENT 724 IS DISTORTING EXCESSIVELY.
   ELEMENT 724 IS DISTORTING EXCESSIVELY.
   ELEMENT 725 IS DISTORTING EXCESSIVELY.
   ELEMENT 725 IS DISTORTING EXCESSIVELY.
   ELEMENT 725 IS DISTORTING EXCESSIVELY.
   ELEMENT 725 IS DISTORTING EXCESSIVELY.
   ELEMENT 725 IS DISTORTING EXCESSIVELY.
   ELEMENT 725 IS DISTORTING EXCESSIVELY.
   ELEMENT 725 IS DISTORTING EXCESSIVELY.
   ELEMENT 725 IS DISTORTING EXCESSIVELY.
   ELEMENT 726 IS DISTORTING EXCESSIVELY.
   ELEMENT 726 IS DISTORTING EXCESSIVELY.
   ELEMENT 726 IS DISTORTING EXCESSIVELY.
   ELEMENT 726 IS DISTORTING EXCESSIVELY.
   ELEMENT 726 IS DISTORTING EXCESSIVELY.
   ELEMENT 726 IS DISTORTING EXCESSIVELY.
   ELEMENT 726 IS DISTORTING EXCESSIVELY.
   ELEMENT 726 IS DISTORTING EXCESSIVELY.
   ELEMENT 727 IS DISTORTING EXCESSIVELY.
   ELEMENT 727 IS DISTORTING EXCESSIVELY.
   ELEMENT 727 IS DISTORTING EXCESSIVELY.
   ELEMENT 727 IS DISTORTING EXCESSIVELY.
   ELEMENT 727 IS DISTORTING EXCESSIVELY.
   ELEMENT 727 IS DISTORTING EXCESSIVELY.
   ELEMENT 727 IS DISTORTING EXCESSIVELY.
   ELEMENT 727 IS DISTORTING EXCESSIVELY.
   ELEMENT 728 IS DISTORTING EXCESSIVELY.
   ELEMENT 728 IS DISTORTING EXCESSIVELY.
   ELEMENT 728 IS DISTORTING EXCESSIVELY.
   ELEMENT 728 IS DISTORTING EXCESSIVELY.
   ELEMENT 728 IS DISTORTING EXCESSIVELY.
   ELEMENT 728 IS DISTORTING EXCESSIVELY.
   ELEMENT 728 IS DISTORTING EXCESSIVELY.
   ELEMENT 728 IS DISTORTING EXCESSIVELY.
   ELEMENT 729 IS DISTORTING EXCESSIVELY.
   ELEMENT 729 IS DISTORTING EXCESSIVELY.
   ELEMENT 729 IS DISTORTING EXCESSIVELY.
   ELEMENT 729 IS DISTORTING EXCESSIVELY.
   ELEMENT 908 IS DISTORTING EXCESSIVELY.
   ELEMENT 908 IS DISTORTING EXCESSIVELY.
   ELEMENT 924 IS DISTORTING EXCESSIVELY.
   ELEMENT 924 IS DISTORTING EXCESSIVELY.
   ELEMENT 925 IS DISTORTING EXCESSIVELY.
   ELEMENT 925 IS DISTORTING EXCESSIVELY.
   ELEMENT 925 IS DISTORTING EXCESSIVELY.
   ELEMENT 925 IS DISTORTING EXCESSIVELY.
   ELEMENT 925 IS DISTORTING EXCESSIVELY.
   ELEMENT 925 IS DISTORTING EXCESSIVELY.
   ELEMENT 926 IS DISTORTING EXCESSIVELY.
   ELEMENT 926 IS DISTORTING EXCESSIVELY.
   ELEMENT 926 IS DISTORTING EXCESSIVELY.
   ELEMENT 926 IS DISTORTING EXCESSIVELY.
   ELEMENT 926 IS DISTORTING EXCESSIVELY.
   ELEMENT 926 IS DISTORTING EXCESSIVELY.
   ELEMENT 926 IS DISTORTING EXCESSIVELY.
   ELEMENT 926 IS DISTORTING EXCESSIVELY.
   ELEMENT 927 IS DISTORTING EXCESSIVELY.
   ELEMENT 927 IS DISTORTING EXCESSIVELY.
   ELEMENT 927 IS DISTORTING EXCESSIVELY.
   ELEMENT 927 IS DISTORTING EXCESSIVELY.
   ELEMENT 927 IS DISTORTING EXCESSIVELY.
   ELEMENT 927 IS DISTORTING EXCESSIVELY.
   ELEMENT 927 IS DISTORTING EXCESSIVELY.
   ELEMENT 927 IS DISTORTING EXCESSIVELY.
   ELEMENT 928 IS DISTORTING EXCESSIVELY.
   ELEMENT 928 IS DISTORTING EXCESSIVELY.
   ELEMENT 928 IS DISTORTING EXCESSIVELY.
   ELEMENT 928 IS DISTORTING EXCESSIVELY.
   ELEMENT 928 IS DISTORTING EXCESSIVELY.
   ELEMENT 928 IS DISTORTING EXCESSIVELY.
   ELEMENT 928 IS DISTORTING EXCESSIVELY.
   ELEMENT 928 IS DISTORTING EXCESSIVELY.
   ELEMENT 929 IS DISTORTING EXCESSIVELY.
   ELEMENT 929 IS DISTORTING EXCESSIVELY.
   ELEMENT 929 IS DISTORTING EXCESSIVELY.
   ELEMENT 929 IS DISTORTING EXCESSIVELY.
   ELEMENT 929 IS DISTORTING EXCESSIVELY.
   ELEMENT 929 IS DISTORTING EXCESSIVELY.
   ELEMENT 929 IS DISTORTING EXCESSIVELY.
   ELEMENT 929 IS DISTORTING EXCESSIVELY.
   ELEMENT 930 IS DISTORTING EXCESSIVELY.
   ELEMENT 930 IS DISTORTING EXCESSIVELY.
   ELEMENT 930 IS DISTORTING EXCESSIVELY.
   ELEMENT 930 IS DISTORTING EXCESSIVELY.
   ELEMENT 930 IS DISTORTING EXCESSIVELY.
   ELEMENT 930 IS DISTORTING EXCESSIVELY.
   ELEMENT 930 IS DISTORTING EXCESSIVELY.
   ELEMENT 930 IS DISTORTING EXCESSIVELY.
   ELEMENT 931 IS DISTORTING EXCESSIVELY.
   ELEMENT 931 IS DISTORTING EXCESSIVELY.
   ELEMENT 931 IS DISTORTING EXCESSIVELY.
   ELEMENT 931 IS DISTORTING EXCESSIVELY.
   ELEMENT 931 IS DISTORTING EXCESSIVELY.
   ELEMENT 931 IS DISTORTING EXCESSIVELY.
   ELEMENT 931 IS DISTORTING EXCESSIVELY.
   ELEMENT 931 IS DISTORTING EXCESSIVELY.
   ELEMENT 932 IS DISTORTING EXCESSIVELY.
   ELEMENT 932 IS DISTORTING EXCESSIVELY.
   ELEMENT 932 IS DISTORTING EXCESSIVELY.
   ELEMENT 932 IS DISTORTING EXCESSIVELY.
   ELEMENT 932 IS DISTORTING EXCESSIVELY.
   ELEMENT 932 IS DISTORTING EXCESSIVELY.
   ELEMENT 932 IS DISTORTING EXCESSIVELY.
   ELEMENT 932 IS DISTORTING EXCESSIVELY.
   ELEMENT 933 IS DISTORTING EXCESSIVELY.
   ELEMENT 933 IS DISTORTING EXCESSIVELY.
   ELEMENT 933 IS DISTORTING EXCESSIVELY.
   ELEMENT 933 IS DISTORTING EXCESSIVELY.
   ELEMENT 933 IS DISTORTING EXCESSIVELY.
   ELEMENT 933 IS DISTORTING EXCESSIVELY.
   ELEMENT 934 IS DISTORTING EXCESSIVELY.
   ELEMENT 934 IS DISTORTING EXCESSIVELY.
   ELEMENT 936 IS DISTORTING EXCESSIVELY.
   ELEMENT 936 IS DISTORTING EXCESSIVELY.
   ELEMENT 939 IS DISTORTING EXCESSIVELY.
   ELEMENT 939 IS DISTORTING EXCESSIVELY.
   ELEMENT 941 IS DISTORTING EXCESSIVELY.
   ELEMENT 941 IS DISTORTING EXCESSIVELY.
   ELEMENT 941 IS DISTORTING EXCESSIVELY.
   ELEMENT 941 IS DISTORTING EXCESSIVELY.
   ELEMENT 943 IS DISTORTING EXCESSIVELY.
   ELEMENT 943 IS DISTORTING EXCESSIVELY.
   ELEMENT 943 IS DISTORTING EXCESSIVELY.
   ELEMENT 943 IS DISTORTING EXCESSIVELY.
   ELEMENT 944 IS DISTORTING EXCESSIVELY.
   ELEMENT 944 IS DISTORTING EXCESSIVELY.
   ELEMENT 944 IS DISTORTING EXCESSIVELY.
   ELEMENT 944 IS DISTORTING EXCESSIVELY.
   ELEMENT 945 IS DISTORTING EXCESSIVELY.
   ELEMENT 945 IS DISTORTING EXCESSIVELY.
   ELEMENT 946 IS DISTORTING EXCESSIVELY.
   ELEMENT 946 IS DISTORTING EXCESSIVELY.
   ELEMENT 946 IS DISTORTING EXCESSIVELY.
   ELEMENT 946 IS DISTORTING EXCESSIVELY.
   ELEMENT 947 IS DISTORTING EXCESSIVELY.
   ELEMENT 947 IS DISTORTING EXCESSIVELY.
   ELEMENT 947 IS DISTORTING EXCESSIVELY.
   ELEMENT 947 IS DISTORTING EXCESSIVELY.
   ELEMENT 948 IS DISTORTING EXCESSIVELY.
   ELEMENT 948 IS DISTORTING EXCESSIVELY.
   ELEMENT 948 IS DISTORTING EXCESSIVELY.
   ELEMENT 948 IS DISTORTING EXCESSIVELY.
   ELEMENT 948 IS DISTORTING EXCESSIVELY.
   ELEMENT 948 IS DISTORTING EXCESSIVELY.
   ELEMENT 948 IS DISTORTING EXCESSIVELY.
   ELEMENT 948 IS DISTORTING EXCESSIVELY.
   ELEMENT 949 IS DISTORTING EXCESSIVELY.
   ELEMENT 949 IS DISTORTING EXCESSIVELY.
   ELEMENT 949 IS DISTORTING EXCESSIVELY.
   ELEMENT 949 IS DISTORTING EXCESSIVELY.
   ELEMENT 949 IS DISTORTING EXCESSIVELY.
   ELEMENT 949 IS DISTORTING EXCESSIVELY.
   ELEMENT 949 IS DISTORTING EXCESSIVELY.
   ELEMENT 949 IS DISTORTING EXCESSIVELY.
   ELEMENT 950 IS DISTORTING EXCESSIVELY.
   ELEMENT 950 IS DISTORTING EXCESSIVELY.
   ELEMENT 950 IS DISTORTING EXCESSIVELY.
   ELEMENT 950 IS DISTORTING EXCESSIVELY.
   ELEMENT 950 IS DISTORTING EXCESSIVELY.
   ELEMENT 950 IS DISTORTING EXCESSIVELY.
   ELEMENT 950 IS DISTORTING EXCESSIVELY.
   ELEMENT 950 IS DISTORTING EXCESSIVELY.
   ELEMENT 951 IS DISTORTING EXCESSIVELY.
   ELEMENT 951 IS DISTORTING EXCESSIVELY.
   ELEMENT 951 IS DISTORTING EXCESSIVELY.
   ELEMENT 951 IS DISTORTING EXCESSIVELY.
   ELEMENT 951 IS DISTORTING EXCESSIVELY.
   ELEMENT 951 IS DISTORTING EXCESSIVELY.
   ELEMENT 951 IS DISTORTING EXCESSIVELY.
   ELEMENT 951 IS DISTORTING EXCESSIVELY.
   ELEMENT 952 IS DISTORTING EXCESSIVELY.
   ELEMENT 952 IS DISTORTING EXCESSIVELY.
   ELEMENT 952 IS DISTORTING EXCESSIVELY.
   ELEMENT 952 IS DISTORTING EXCESSIVELY.
   ELEMENT 952 IS DISTORTING EXCESSIVELY.
   ELEMENT 952 IS DISTORTING EXCESSIVELY.
   ELEMENT 1133 IS DISTORTING EXCESSIVELY.
   ELEMENT 1133 IS DISTORTING EXCESSIVELY.
   ELEMENT 1133 IS DISTORTING EXCESSIVELY.
   ELEMENT 1133 IS DISTORTING EXCESSIVELY.
   ELEMENT 1133 IS DISTORTING EXCESSIVELY.
   ELEMENT 1133 IS DISTORTING EXCESSIVELY.
   ELEMENT 1133 IS DISTORTING EXCESSIVELY.
   ELEMENT 1133 IS DISTORTING EXCESSIVELY.
   ELEMENT 1165 IS DISTORTING EXCESSIVELY.
   ELEMENT 1165 IS DISTORTING EXCESSIVELY.
   ELEMENT 1165 IS DISTORTING EXCESSIVELY.
   ELEMENT 1165 IS DISTORTING EXCESSIVELY.
   ELEMENT 1166 IS DISTORTING EXCESSIVELY.
   ELEMENT 1166 IS DISTORTING EXCESSIVELY.
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
   ELEMENT 1167 IS DISTORTING EXCESSIVELY.
   ELEMENT 1167 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1168 IS DISTORTING EXCESSIVELY.
   ELEMENT 1170 IS DISTORTING EXCESSIVELY.
   ELEMENT 1170 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1171 IS DISTORTING EXCESSIVELY.
   ELEMENT 1172 IS DISTORTING EXCESSIVELY.
   ELEMENT 1172 IS DISTORTING EXCESSIVELY.
   ELEMENT 1172 IS DISTORTING EXCESSIVELY.
   ELEMENT 1172 IS DISTORTING EXCESSIVELY.
   ELEMENT 1173 IS DISTORTING EXCESSIVELY.
   ELEMENT 1173 IS DISTORTING EXCESSIVELY.
   ELEMENT 1175 IS DISTORTING EXCESSIVELY.
   ELEMENT 1175 IS DISTORTING EXCESSIVELY.
   ELEMENT 1358 IS DISTORTING EXCESSIVELY.
   ELEMENT 1358 IS DISTORTING EXCESSIVELY.
   ELEMENT 1358 IS DISTORTING EXCESSIVELY.
   ELEMENT 1358 IS DISTORTING EXCESSIVELY.
   ELEMENT 1381 IS DISTORTING EXCESSIVELY.
   ELEMENT 1381 IS DISTORTING EXCESSIVELY.
   ELEMENT 1381 IS DISTORTING EXCESSIVELY.
   ELEMENT 1381 IS DISTORTING EXCESSIVELY.
   ELEMENT 1381 IS DISTORTING EXCESSIVELY.
   ELEMENT 1381 IS DISTORTING EXCESSIVELY.
   ELEMENT 1381 IS DISTORTING EXCESSIVELY.
   ELEMENT 1381 IS DISTORTING EXCESSIVELY.
   ELEMENT 1382 IS DISTORTING EXCESSIVELY.
   ELEMENT 1382 IS DISTORTING EXCESSIVELY.
   ELEMENT 1382 IS DISTORTING EXCESSIVELY.
   ELEMENT 1382 IS DISTORTING EXCESSIVELY.
   ELEMENT 1382 IS DISTORTING EXCESSIVELY.
   ELEMENT 1382 IS DISTORTING EXCESSIVELY.
   ELEMENT 1383 IS DISTORTING EXCESSIVELY.
   ELEMENT 1383 IS DISTORTING EXCESSIVELY.
   ELEMENT 1388 IS DISTORTING EXCESSIVELY.
   ELEMENT 1388 IS DISTORTING EXCESSIVELY.
   ELEMENT 1389 IS DISTORTING EXCESSIVELY.
   ELEMENT 1389 IS DISTORTING EXCESSIVELY.
   ELEMENT 1389 IS DISTORTING EXCESSIVELY.
   ELEMENT 1389 IS DISTORTING EXCESSIVELY.
   ELEMENT 1389 IS DISTORTING EXCESSIVELY.
   ELEMENT 1389 IS DISTORTING EXCESSIVELY.
   ELEMENT 1389 IS DISTORTING EXCESSIVELY.
   ELEMENT 1389 IS DISTORTING EXCESSIVELY.
   ELEMENT 1390 IS DISTORTING EXCESSIVELY.
   ELEMENT 1390 IS DISTORTING EXCESSIVELY.
   ELEMENT 1390 IS DISTORTING EXCESSIVELY.
   ELEMENT 1390 IS DISTORTING EXCESSIVELY.
   ELEMENT 1390 IS DISTORTING EXCESSIVELY.
   ELEMENT 1390 IS DISTORTING EXCESSIVELY.
   ELEMENT 1390 IS DISTORTING EXCESSIVELY.
   ELEMENT 1390 IS DISTORTING EXCESSIVELY.
   ELEMENT 1391 IS DISTORTING EXCESSIVELY.
   ELEMENT 1391 IS DISTORTING EXCESSIVELY.
   ELEMENT 1391 IS DISTORTING EXCESSIVELY.
   ELEMENT 1391 IS DISTORTING EXCESSIVELY.
   ELEMENT 1391 IS DISTORTING EXCESSIVELY.
   ELEMENT 1391 IS DISTORTING EXCESSIVELY.
   ELEMENT 1391 IS DISTORTING EXCESSIVELY.
   ELEMENT 1391 IS DISTORTING EXCESSIVELY.
   ELEMENT 1392 IS DISTORTING EXCESSIVELY.
   ELEMENT 1392 IS DISTORTING EXCESSIVELY.
   ELEMENT 1392 IS DISTORTING EXCESSIVELY.
   ELEMENT 1392 IS DISTORTING EXCESSIVELY.
   ELEMENT 1393 IS DISTORTING EXCESSIVELY.
   ELEMENT 1393 IS DISTORTING EXCESSIVELY.
   ELEMENT 1393 IS DISTORTING EXCESSIVELY.
   ELEMENT 1393 IS DISTORTING EXCESSIVELY.
   ELEMENT 1393 IS DISTORTING EXCESSIVELY.
   ELEMENT 1393 IS DISTORTING EXCESSIVELY.
   ELEMENT 1394 IS DISTORTING EXCESSIVELY.
   ELEMENT 1394 IS DISTORTING EXCESSIVELY.
   ELEMENT 1394 IS DISTORTING EXCESSIVELY.
   ELEMENT 1394 IS DISTORTING EXCESSIVELY.
   ELEMENT 1394 IS DISTORTING EXCESSIVELY.
   ELEMENT 1394 IS DISTORTING EXCESSIVELY.
   ELEMENT 1394 IS DISTORTING EXCESSIVELY.
   ELEMENT 1394 IS DISTORTING EXCESSIVELY.
   ELEMENT 1395 IS DISTORTING EXCESSIVELY.
   ELEMENT 1395 IS DISTORTING EXCESSIVELY.
   ELEMENT 1395 IS DISTORTING EXCESSIVELY.
   ELEMENT 1395 IS DISTORTING EXCESSIVELY.
   ELEMENT 1395 IS DISTORTING EXCESSIVELY.
   ELEMENT 1395 IS DISTORTING EXCESSIVELY.
   ELEMENT 1395 IS DISTORTING EXCESSIVELY.
   ELEMENT 1395 IS DISTORTING EXCESSIVELY.
   ELEMENT 1396 IS DISTORTING EXCESSIVELY.
   ELEMENT 1396 IS DISTORTING EXCESSIVELY.
   ELEMENT 1396 IS DISTORTING EXCESSIVELY.
   ELEMENT 1396 IS DISTORTING EXCESSIVELY.
   ELEMENT 1396 IS DISTORTING EXCESSIVELY.
   ELEMENT 1396 IS DISTORTING EXCESSIVELY.
   ELEMENT 1396 IS DISTORTING EXCESSIVELY.
   ELEMENT 1396 IS DISTORTING EXCESSIVELY.
   ELEMENT 1583 IS DISTORTING EXCESSIVELY.
   ELEMENT 1583 IS DISTORTING EXCESSIVELY.
   ELEMENT 1583 IS DISTORTING EXCESSIVELY.
   ELEMENT 1583 IS DISTORTING EXCESSIVELY.
   ELEMENT 1605 IS DISTORTING EXCESSIVELY.
   ELEMENT 1605 IS DISTORTING EXCESSIVELY.
   ELEMENT 1605 IS DISTORTING EXCESSIVELY.
   ELEMENT 1605 IS DISTORTING EXCESSIVELY.
   ELEMENT 1605 IS DISTORTING EXCESSIVELY.
   ELEMENT 1605 IS DISTORTING EXCESSIVELY.
   ELEMENT 1606 IS DISTORTING EXCESSIVELY.
   ELEMENT 1606 IS DISTORTING EXCESSIVELY.
   ELEMENT 1606 IS DISTORTING EXCESSIVELY.
   ELEMENT 1606 IS DISTORTING EXCESSIVELY.
   ELEMENT 1606 IS DISTORTING EXCESSIVELY.
   ELEMENT 1606 IS DISTORTING EXCESSIVELY.
   ELEMENT 1606 IS DISTORTING EXCESSIVELY.
   ELEMENT 1606 IS DISTORTING EXCESSIVELY.
   ELEMENT 1607 IS DISTORTING EXCESSIVELY.
   ELEMENT 1607 IS DISTORTING EXCESSIVELY.
   ELEMENT 1607 IS DISTORTING EXCESSIVELY.
   ELEMENT 1607 IS DISTORTING EXCESSIVELY.
   ELEMENT 1607 IS DISTORTING EXCESSIVELY.
   ELEMENT 1607 IS DISTORTING EXCESSIVELY.
   ELEMENT 1607 IS DISTORTING EXCESSIVELY.
   ELEMENT 1607 IS DISTORTING EXCESSIVELY.
   ELEMENT 1608 IS DISTORTING EXCESSIVELY.
   ELEMENT 1608 IS DISTORTING EXCESSIVELY.
   ELEMENT 1608 IS DISTORTING EXCESSIVELY.
   ELEMENT 1608 IS DISTORTING EXCESSIVELY.
   ELEMENT 1608 IS DISTORTING EXCESSIVELY.
   ELEMENT 1608 IS DISTORTING EXCESSIVELY.
   ELEMENT 1619 IS DISTORTING EXCESSIVELY.
   ELEMENT 1619 IS DISTORTING EXCESSIVELY.
   ELEMENT 1619 IS DISTORTING EXCESSIVELY.
   ELEMENT 1619 IS DISTORTING EXCESSIVELY.
   ELEMENT 1619 IS DISTORTING EXCESSIVELY.
   ELEMENT 1619 IS DISTORTING EXCESSIVELY.
   ELEMENT 1619 IS DISTORTING EXCESSIVELY.
   ELEMENT 1619 IS DISTORTING EXCESSIVELY.
   ELEMENT 1620 IS DISTORTING EXCESSIVELY.
   ELEMENT 1620 IS DISTORTING EXCESSIVELY.
   ELEMENT 1620 IS DISTORTING EXCESSIVELY.
   ELEMENT 1620 IS DISTORTING EXCESSIVELY.
   ELEMENT 1620 IS DISTORTING EXCESSIVELY.
   ELEMENT 1620 IS DISTORTING EXCESSIVELY.
   ELEMENT 1620 IS DISTORTING EXCESSIVELY.
   ELEMENT 1620 IS DISTORTING EXCESSIVELY.
   ELEMENT 1621 IS DISTORTING EXCESSIVELY.
   ELEMENT 1621 IS DISTORTING EXCESSIVELY.
   ELEMENT 1621 IS DISTORTING EXCESSIVELY.
   ELEMENT 1621 IS DISTORTING EXCESSIVELY.
   ELEMENT 1621 IS DISTORTING EXCESSIVELY.
   ELEMENT 1621 IS DISTORTING EXCESSIVELY.
   ELEMENT 1621 IS DISTORTING EXCESSIVELY.
   ELEMENT 1621 IS DISTORTING EXCESSIVELY.
   ELEMENT 1808 IS DISTORTING EXCESSIVELY.
   ELEMENT 1808 IS DISTORTING EXCESSIVELY.
   ELEMENT 1808 IS DISTORTING EXCESSIVELY.
   ELEMENT 1808 IS DISTORTING EXCESSIVELY.
   ELEMENT 1830 IS DISTORTING EXCESSIVELY.
   ELEMENT 1830 IS DISTORTING EXCESSIVELY.
   ELEMENT 1830 IS DISTORTING EXCESSIVELY.
   ELEMENT 1830 IS DISTORTING EXCESSIVELY.
   ELEMENT 1830 IS DISTORTING EXCESSIVELY.
   ELEMENT 1830 IS DISTORTING EXCESSIVELY.
   ELEMENT 1830 IS DISTORTING EXCESSIVELY.
   ELEMENT 1830 IS DISTORTING EXCESSIVELY.
   ELEMENT 1831 IS DISTORTING EXCESSIVELY.
   ELEMENT 1831 IS DISTORTING EXCESSIVELY.
   ELEMENT 1831 IS DISTORTING EXCESSIVELY.
   ELEMENT 1831 IS DISTORTING EXCESSIVELY.
   ELEMENT 1831 IS DISTORTING EXCESSIVELY.
   ELEMENT 1831 IS DISTORTING EXCESSIVELY.
   ELEMENT 1831 IS DISTORTING EXCESSIVELY.
   ELEMENT 1831 IS DISTORTING EXCESSIVELY.
   ELEMENT 1844 IS DISTORTING EXCESSIVELY.
   ELEMENT 1844 IS DISTORTING EXCESSIVELY.
   ELEMENT 1844 IS DISTORTING EXCESSIVELY.
   ELEMENT 1844 IS DISTORTING EXCESSIVELY.
   ELEMENT 1845 IS DISTORTING EXCESSIVELY.
   ELEMENT 1845 IS DISTORTING EXCESSIVELY.
   ELEMENT 1845 IS DISTORTING EXCESSIVELY.
   ELEMENT 1845 IS DISTORTING EXCESSIVELY.
   ELEMENT 1845 IS DISTORTING EXCESSIVELY.
   ELEMENT 1845 IS DISTORTING EXCESSIVELY.
   ELEMENT 1845 IS DISTORTING EXCESSIVELY.
   ELEMENT 1845 IS DISTORTING EXCESSIVELY.
   ELEMENT 1846 IS DISTORTING EXCESSIVELY.
   ELEMENT 1846 IS DISTORTING EXCESSIVELY.
   ELEMENT 1846 IS DISTORTING EXCESSIVELY.
   ELEMENT 1846 IS DISTORTING EXCESSIVELY.
   ELEMENT 1846 IS DISTORTING EXCESSIVELY.
   ELEMENT 1846 IS DISTORTING EXCESSIVELY.
   ELEMENT 1846 IS DISTORTING EXCESSIVELY.
   ELEMENT 1846 IS DISTORTING EXCESSIVELY.
   ELEMENT 2033 IS DISTORTING EXCESSIVELY.
   ELEMENT 2033 IS DISTORTING EXCESSIVELY.
   ELEMENT 2033 IS DISTORTING EXCESSIVELY.
   ELEMENT 2033 IS DISTORTING EXCESSIVELY.
   ELEMENT 2054 IS DISTORTING EXCESSIVELY.
   ELEMENT 2054 IS DISTORTING EXCESSIVELY.
   ELEMENT 2054 IS DISTORTING EXCESSIVELY.
   ELEMENT 2054 IS DISTORTING EXCESSIVELY.
   ELEMENT 2054 IS DISTORTING EXCESSIVELY.
   ELEMENT 2054 IS DISTORTING EXCESSIVELY.
   ELEMENT 2055 IS DISTORTING EXCESSIVELY.
   ELEMENT 2055 IS DISTORTING EXCESSIVELY.
   ELEMENT 2055 IS DISTORTING EXCESSIVELY.
   ELEMENT 2055 IS DISTORTING EXCESSIVELY.
   ELEMENT 2055 IS DISTORTING EXCESSIVELY.
   ELEMENT 2055 IS DISTORTING EXCESSIVELY.
   ELEMENT 2055 IS DISTORTING EXCESSIVELY.
   ELEMENT 2055 IS DISTORTING EXCESSIVELY.
   ELEMENT 2056 IS DISTORTING EXCESSIVELY.
   ELEMENT 2056 IS DISTORTING EXCESSIVELY.
   ELEMENT 2070 IS DISTORTING EXCESSIVELY.
   ELEMENT 2070 IS DISTORTING EXCESSIVELY.
   ELEMENT 2070 IS DISTORTING EXCESSIVELY.
   ELEMENT 2070 IS DISTORTING EXCESSIVELY.
   ELEMENT 2070 IS DISTORTING EXCESSIVELY.
   ELEMENT 2070 IS DISTORTING EXCESSIVELY.
   ELEMENT 2070 IS DISTORTING EXCESSIVELY.
   ELEMENT 2070 IS DISTORTING EXCESSIVELY.
   ELEMENT 2071 IS DISTORTING EXCESSIVELY.
   ELEMENT 2071 IS DISTORTING EXCESSIVELY.
   ELEMENT 2071 IS DISTORTING EXCESSIVELY.
   ELEMENT 2071 IS DISTORTING EXCESSIVELY.
   ELEMENT 2071 IS DISTORTING EXCESSIVELY.
   ELEMENT 2071 IS DISTORTING EXCESSIVELY.
   ELEMENT 2071 IS DISTORTING EXCESSIVELY.
   ELEMENT 2071 IS DISTORTING EXCESSIVELY.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2721 IS OVERCLOSED BY 179.117       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2722 IS OVERCLOSED BY 183.568       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2723 IS OVERCLOSED BY 185.583       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2724 IS OVERCLOSED BY 185.723       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2725 IS OVERCLOSED BY 184.026       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2726 IS OVERCLOSED BY 180.494       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2727 IS OVERCLOSED BY 175.110       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2728 IS OVERCLOSED BY 167.845       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2729 IS OVERCLOSED BY 158.661       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2730 IS OVERCLOSED BY 147.243       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2731 IS OVERCLOSED BY 134.358       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2732 IS OVERCLOSED BY 121.260       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2733 IS OVERCLOSED BY 105.919       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2734 IS OVERCLOSED BY 87.6759      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2735 IS OVERCLOSED BY 66.3978      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2736 IS OVERCLOSED BY 48.0127      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2737 IS OVERCLOSED BY 35.1959      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2738 IS OVERCLOSED BY 29.0860      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2739 IS OVERCLOSED BY 18.1609      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2740 IS OVERCLOSED BY 10.2517      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2741 IS OVERCLOSED BY 6.35974     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2742 IS OVERCLOSED BY 1.41090     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2746 IS OVERCLOSED BY 4.22567     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2747 IS OVERCLOSED BY 9.90151     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2748 IS OVERCLOSED BY 10.3751      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2749 IS OVERCLOSED BY 16.5688      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2750 IS OVERCLOSED BY 21.9666      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2751 IS OVERCLOSED BY 17.9938      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2752 IS OVERCLOSED BY 32.9885      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2753 IS OVERCLOSED BY 33.3289      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2754 IS OVERCLOSED BY 65.4524      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2755 IS OVERCLOSED BY 59.1092      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2757 IS OVERCLOSED BY 50.5133      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2758 IS OVERCLOSED BY 22.3837      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2759 IS OVERCLOSED BY 732.658E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2761 IS OVERCLOSED BY 4.32021     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2762 IS OVERCLOSED BY 1.69183     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4077 IS OVERCLOSED BY 179.117       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4078 IS OVERCLOSED BY 183.568       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4079 IS OVERCLOSED BY 185.583       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4080 IS OVERCLOSED BY 185.723       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4081 IS OVERCLOSED BY 184.026       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4082 IS OVERCLOSED BY 180.494       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4083 IS OVERCLOSED BY 175.110       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4084 IS OVERCLOSED BY 167.845       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4085 IS OVERCLOSED BY 158.661       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4086 IS OVERCLOSED BY 147.243       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4087 IS OVERCLOSED BY 134.358       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4088 IS OVERCLOSED BY 121.260       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4089 IS OVERCLOSED BY 105.919       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4090 IS OVERCLOSED BY 87.6759      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4091 IS OVERCLOSED BY 66.3978      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4092 IS OVERCLOSED BY 48.0127      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4093 IS OVERCLOSED BY 35.1959      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4094 IS OVERCLOSED BY 29.0860      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4095 IS OVERCLOSED BY 18.1609      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4096 IS OVERCLOSED BY 10.2517      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4097 IS OVERCLOSED BY 6.35974     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4098 IS OVERCLOSED BY 1.41090     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4102 IS OVERCLOSED BY 4.22567     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4103 IS OVERCLOSED BY 9.90151     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4104 IS OVERCLOSED BY 10.3751      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4105 IS OVERCLOSED BY 16.5688      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4106 IS OVERCLOSED BY 21.9666      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4107 IS OVERCLOSED BY 17.9938      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4108 IS OVERCLOSED BY 32.9885      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4109 IS OVERCLOSED BY 33.3289      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4110 IS OVERCLOSED BY 65.4524      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4111 IS OVERCLOSED BY 59.1092      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4113 IS OVERCLOSED BY 50.5133      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4114 IS OVERCLOSED BY 22.3837      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4115 IS OVERCLOSED BY 732.658E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4117 IS OVERCLOSED BY 4.32021     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4118 IS OVERCLOSED BY 1.69183     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 824 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.
 

 ***NOTE: SEVERE CONTACT OVERCLOSURES EXIST. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     1 STARTS. ATTEMPT NUMBER  3, TIME INCREMENT  1.250E-03
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10938
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.490e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   44 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   44 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 23.9641E-09  AT NODE 4071 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -701.528       AT NODE 2738 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      7.167E-02   TIME AVG. FORCE       7.167E-02
 LARGEST SCALED RESIDUAL FORCE       558.       AT NODE       2738   DOF  2
  CORRESPONDING RESIDUAL FORCE       558.    
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -1.125E-02   AT NODE       5427   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      4.722E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     9.728E-07   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     9.728E-07
 LARGEST INCREMENT OF ROTATION     -1.637E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.637E-04   AT NODE       5425   DOF  6
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10894
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.475e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   71 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   43 POINTS CHANGED FROM OPEN TO CLOSED
                   28 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR 14.0516E-06  AT NODE 2759 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -638.600E+12   AT NODE 4112 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      2.599E+10   TIME AVG. FORCE       2.599E+10
 LARGEST SCALED RESIDUAL FORCE      4.789E+14   AT NODE       2756   DOF  2
  CORRESPONDING RESIDUAL FORCE      4.789E+14
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.293E-02   AT NODE       1180   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 2756 
             IS 36.8978E+12  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.713E+10
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -5.254E-05   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -5.254E-05
 LARGEST INCREMENT OF ROTATION      3.980E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     5.618E-04   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10909
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.491e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 5 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.
   ELEMENT 719 IS DISTORTING EXCESSIVELY.
   ELEMENT 719 IS DISTORTING EXCESSIVELY.
   ELEMENT 719 IS DISTORTING EXCESSIVELY.
   ELEMENT 719 IS DISTORTING EXCESSIVELY.
   ELEMENT 720 IS DISTORTING EXCESSIVELY.
   ELEMENT 720 IS DISTORTING EXCESSIVELY.
   ELEMENT 943 IS DISTORTING EXCESSIVELY.
   ELEMENT 943 IS DISTORTING EXCESSIVELY.
   ELEMENT 943 IS DISTORTING EXCESSIVELY.
   ELEMENT 943 IS DISTORTING EXCESSIVELY.
   ELEMENT 944 IS DISTORTING EXCESSIVELY.
   ELEMENT 944 IS DISTORTING EXCESSIVELY.
   ELEMENT 944 IS DISTORTING EXCESSIVELY.
   ELEMENT 944 IS DISTORTING EXCESSIVELY.
   ELEMENT 1801 IS DISTORTING EXCESSIVELY.
   ELEMENT 1801 IS DISTORTING EXCESSIVELY.
   ELEMENT 2026 IS DISTORTING EXCESSIVELY.
   ELEMENT 2026 IS DISTORTING EXCESSIVELY.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2722 IS OVERCLOSED BY 1.49019     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2723 IS OVERCLOSED BY 3.26632     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2724 IS OVERCLOSED BY 4.40395     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2725 IS OVERCLOSED BY 5.08100     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2726 IS OVERCLOSED BY 5.45027     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2727 IS OVERCLOSED BY 5.60568     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2728 IS OVERCLOSED BY 5.60171     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2729 IS OVERCLOSED BY 5.47183     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2730 IS OVERCLOSED BY 5.22149     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2731 IS OVERCLOSED BY 4.87854     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2732 IS OVERCLOSED BY 4.47289     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2733 IS OVERCLOSED BY 3.97386     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2734 IS OVERCLOSED BY 3.37832     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2735 IS OVERCLOSED BY 2.70619     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2736 IS OVERCLOSED BY 2.18881     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2737 IS OVERCLOSED BY 1.79653     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2738 IS OVERCLOSED BY 1.60513     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2739 IS OVERCLOSED BY 1.41162     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2740 IS OVERCLOSED BY 1.22129     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2741 IS OVERCLOSED BY 1.03467     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2742 IS OVERCLOSED BY 853.107E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2743 IS OVERCLOSED BY 679.790E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2744 IS OVERCLOSED BY 517.211E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2755 IS OVERCLOSED BY 1.35549     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4078 IS OVERCLOSED BY 1.49019     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4079 IS OVERCLOSED BY 3.26632     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4080 IS OVERCLOSED BY 4.40395     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4081 IS OVERCLOSED BY 5.08100     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4082 IS OVERCLOSED BY 5.45027     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4083 IS OVERCLOSED BY 5.60568     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4084 IS OVERCLOSED BY 5.60171     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4085 IS OVERCLOSED BY 5.47183     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4086 IS OVERCLOSED BY 5.22149     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4087 IS OVERCLOSED BY 4.87854     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4088 IS OVERCLOSED BY 4.47289     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4089 IS OVERCLOSED BY 3.97386     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4090 IS OVERCLOSED BY 3.37832     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4091 IS OVERCLOSED BY 2.70619     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4092 IS OVERCLOSED BY 2.18881     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4093 IS OVERCLOSED BY 1.79653     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4094 IS OVERCLOSED BY 1.60513     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4095 IS OVERCLOSED BY 1.41162     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4096 IS OVERCLOSED BY 1.22129     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4097 IS OVERCLOSED BY 1.03467     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4098 IS OVERCLOSED BY 853.107E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4099 IS OVERCLOSED BY 679.790E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4100 IS OVERCLOSED BY 517.211E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4111 IS OVERCLOSED BY 1.35549     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 18 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.
 

 ***NOTE: SEVERE CONTACT OVERCLOSURES EXIST. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     1 STARTS. ATTEMPT NUMBER  4, TIME INCREMENT  3.125E-04
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10938
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.490e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   44 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   44 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 1.49776E-09 AT NODE 4071 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -175.382       AT NODE 2738 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.792E-02   TIME AVG. FORCE       1.792E-02
 LARGEST SCALED RESIDUAL FORCE       140.       AT NODE       2738   DOF  2
  CORRESPONDING RESIDUAL FORCE       140.    
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -2.812E-03   AT NODE       5427   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.181E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     5.656E-08   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     5.656E-08
 LARGEST INCREMENT OF ROTATION     -4.093E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -4.093E-05   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10894
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.475e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   37 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   25 POINTS CHANGED FROM OPEN TO CLOSED
                   12 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR 9.72563E-18 AT NODE 4069 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -22.0829      AT NODE 2731 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.387E-03   TIME AVG. FORCE       1.387E-03
 LARGEST SCALED RESIDUAL FORCE      -16.6       AT NODE       1149   DOF  2
  CORRESPONDING RESIDUAL FORCE      -16.6    
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -4.074E-08   AT NODE       1808   DOF  1
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT    -4.468E-10   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -4.468E-10
 LARGEST INCREMENT OF ROTATION     -4.093E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -7.408E-12   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10907
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.490e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   41 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   13 POINTS CHANGED FROM OPEN TO CLOSED
                   28 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     3

   MAX. PENETRATION ERROR 117.238E-18   AT NODE 2713 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -294.142       AT NODE 4104 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.742E-02   TIME AVG. FORCE       1.742E-02
 LARGEST SCALED RESIDUAL FORCE      -245.       AT NODE       1166   DOF  2
  CORRESPONDING RESIDUAL FORCE      -245.    
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        5.278E-11   AT NODE       1619   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.147E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     7.524E-08   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     7.524E-08
 LARGEST INCREMENT OF ROTATION     -4.093E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     2.098E-12   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10892
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.471e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   43 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   27 POINTS CHANGED FROM OPEN TO CLOSED
                   16 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     4

   MAX. PENETRATION ERROR 29.7071E-18  AT NODE 2749 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -142.130       AT NODE 2745 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      6.472E-03   TIME AVG. FORCE       6.472E-03
 LARGEST SCALED RESIDUAL FORCE      -111.       AT NODE       1163   DOF  2
  CORRESPONDING RESIDUAL FORCE      -111.    
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.760E-12   AT NODE       2735   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT     1.480E-10   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     1.480E-10
 LARGEST INCREMENT OF ROTATION     -4.093E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.926E-14   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10903
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.481e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   46 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   17 POINTS CHANGED FROM OPEN TO CLOSED
                   29 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     5

   MAX. PENETRATION ERROR 732.591E-09   AT NODE 4111 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -84.1037E+12  AT NODE 2755 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.841E+09   TIME AVG. FORCE       1.841E+09
 LARGEST SCALED RESIDUAL FORCE     -6.308E+13   AT NODE       1173   DOF  2
  CORRESPONDING RESIDUAL FORCE     -6.308E+13
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.678E-03   AT NODE        487   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 4111 
             IS 24.4116E+12  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.213E+09
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -2.401E-06   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -2.401E-06
 LARGEST INCREMENT OF ROTATION      3.938E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     8.031E-05   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     1 STARTS. ATTEMPT NUMBER  5, TIME INCREMENT  7.813E-05
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10938
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.490e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            1

                   44 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   44 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 93.6099E-12  AT NODE 4071 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -43.8455      AT NODE 2738 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      4.479E-03   TIME AVG. FORCE       4.479E-03
 LARGEST SCALED RESIDUAL FORCE       34.9       AT NODE       2738   DOF  2
  CORRESPONDING RESIDUAL FORCE       34.9    
 LARGEST INCREMENT OF DISP.        -7.031E-04   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -7.031E-04   AT NODE       5427   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT     3.469E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     3.469E-09
 LARGEST INCREMENT OF ROTATION     -1.023E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.023E-05   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10894
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.475e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   38 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   12 POINTS CHANGED FROM OPEN TO CLOSED
                   26 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR -26.5543E-06  AT NODE 4071 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -585.943E+12   AT NODE 4072 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          PENETRATION ERROR TOO LARGE COMPARED TO DISPLACEMENT INCREMENT.

 AVERAGE FORCE                      1.283E+10   TIME AVG. FORCE       1.283E+10
 LARGEST SCALED RESIDUAL FORCE     -4.395E+14   AT NODE       1134   DOF  2
  CORRESPONDING RESIDUAL FORCE     -4.395E+14
 LARGEST INCREMENT OF DISP.         1.232E-03   AT NODE        698   DOF  2
 LARGEST CORRECTION TO DISP.        1.453E-03   AT NODE        700   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 2716 
             IS 219.020E+12   TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      8.451E+09
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     5.562E-03   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     5.562E-03
 LARGEST INCREMENT OF ROTATION      1.033E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.135E-04   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10880
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.443e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.
   ELEMENT 17 IS DISTORTING EXCESSIVELY.
   ELEMENT 17 IS DISTORTING EXCESSIVELY.
   ELEMENT 17 IS DISTORTING EXCESSIVELY.
   ELEMENT 17 IS DISTORTING EXCESSIVELY.
   ELEMENT 18 IS DISTORTING EXCESSIVELY.
   ELEMENT 18 IS DISTORTING EXCESSIVELY.
   ELEMENT 18 IS DISTORTING EXCESSIVELY.
   ELEMENT 18 IS DISTORTING EXCESSIVELY.
   ELEMENT 18 IS DISTORTING EXCESSIVELY.
   ELEMENT 18 IS DISTORTING EXCESSIVELY.
   ELEMENT 18 IS DISTORTING EXCESSIVELY.
   ELEMENT 18 IS DISTORTING EXCESSIVELY.
   ELEMENT 212 IS DISTORTING EXCESSIVELY.
   ELEMENT 212 IS DISTORTING EXCESSIVELY.
   ELEMENT 212 IS DISTORTING EXCESSIVELY.
   ELEMENT 212 IS DISTORTING EXCESSIVELY.
   ELEMENT 242 IS DISTORTING EXCESSIVELY.
   ELEMENT 242 IS DISTORTING EXCESSIVELY.
   ELEMENT 243 IS DISTORTING EXCESSIVELY.
   ELEMENT 243 IS DISTORTING EXCESSIVELY.
   ELEMENT 243 IS DISTORTING EXCESSIVELY.
   ELEMENT 243 IS DISTORTING EXCESSIVELY.
   ELEMENT 243 IS DISTORTING EXCESSIVELY.
   ELEMENT 243 IS DISTORTING EXCESSIVELY.
   ELEMENT 243 IS DISTORTING EXCESSIVELY.
   ELEMENT 243 IS DISTORTING EXCESSIVELY.
   ELEMENT 437 IS DISTORTING EXCESSIVELY.
   ELEMENT 437 IS DISTORTING EXCESSIVELY.
   ELEMENT 468 IS DISTORTING EXCESSIVELY.
   ELEMENT 468 IS DISTORTING EXCESSIVELY.
   ELEMENT 468 IS DISTORTING EXCESSIVELY.
   ELEMENT 468 IS DISTORTING EXCESSIVELY.
   ELEMENT 468 IS DISTORTING EXCESSIVELY.
   ELEMENT 468 IS DISTORTING EXCESSIVELY.
   ELEMENT 468 IS DISTORTING EXCESSIVELY.
   ELEMENT 468 IS DISTORTING EXCESSIVELY.
   ELEMENT 679 IS DISTORTING EXCESSIVELY.
   ELEMENT 679 IS DISTORTING EXCESSIVELY.
   ELEMENT 693 IS DISTORTING EXCESSIVELY.
   ELEMENT 693 IS DISTORTING EXCESSIVELY.
   ELEMENT 693 IS DISTORTING EXCESSIVELY.
   ELEMENT 693 IS DISTORTING EXCESSIVELY.
   ELEMENT 693 IS DISTORTING EXCESSIVELY.
   ELEMENT 693 IS DISTORTING EXCESSIVELY.
   ELEMENT 903 IS DISTORTING EXCESSIVELY.
   ELEMENT 903 IS DISTORTING EXCESSIVELY.
   ELEMENT 903 IS DISTORTING EXCESSIVELY.
   ELEMENT 903 IS DISTORTING EXCESSIVELY.
   ELEMENT 904 IS DISTORTING EXCESSIVELY.
   ELEMENT 904 IS DISTORTING EXCESSIVELY.
   ELEMENT 904 IS DISTORTING EXCESSIVELY.
   ELEMENT 904 IS DISTORTING EXCESSIVELY.
   ELEMENT 918 IS DISTORTING EXCESSIVELY.
   ELEMENT 918 IS DISTORTING EXCESSIVELY.
   ELEMENT 1128 IS DISTORTING EXCESSIVELY.
   ELEMENT 1128 IS DISTORTING EXCESSIVELY.
   ELEMENT 1128 IS DISTORTING EXCESSIVELY.
   ELEMENT 1128 IS DISTORTING EXCESSIVELY.
   ELEMENT 1129 IS DISTORTING EXCESSIVELY.
   ELEMENT 1129 IS DISTORTING EXCESSIVELY.
   ELEMENT 1129 IS DISTORTING EXCESSIVELY.
   ELEMENT 1129 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1354 IS DISTORTING EXCESSIVELY.
   ELEMENT 1354 IS DISTORTING EXCESSIVELY.
   ELEMENT 1368 IS DISTORTING EXCESSIVELY.
   ELEMENT 1368 IS DISTORTING EXCESSIVELY.
   ELEMENT 1368 IS DISTORTING EXCESSIVELY.
   ELEMENT 1368 IS DISTORTING EXCESSIVELY.
   ELEMENT 1368 IS DISTORTING EXCESSIVELY.
   ELEMENT 1368 IS DISTORTING EXCESSIVELY.
   ELEMENT 1593 IS DISTORTING EXCESSIVELY.
   ELEMENT 1593 IS DISTORTING EXCESSIVELY.
   ELEMENT 1593 IS DISTORTING EXCESSIVELY.
   ELEMENT 1593 IS DISTORTING EXCESSIVELY.
   ELEMENT 1593 IS DISTORTING EXCESSIVELY.
   ELEMENT 1593 IS DISTORTING EXCESSIVELY.
   ELEMENT 1593 IS DISTORTING EXCESSIVELY.
   ELEMENT 1593 IS DISTORTING EXCESSIVELY.
   ELEMENT 1817 IS DISTORTING EXCESSIVELY.
   ELEMENT 1817 IS DISTORTING EXCESSIVELY.
   ELEMENT 1818 IS DISTORTING EXCESSIVELY.
   ELEMENT 1818 IS DISTORTING EXCESSIVELY.
   ELEMENT 1818 IS DISTORTING EXCESSIVELY.
   ELEMENT 1818 IS DISTORTING EXCESSIVELY.
   ELEMENT 1818 IS DISTORTING EXCESSIVELY.
   ELEMENT 1818 IS DISTORTING EXCESSIVELY.
   ELEMENT 1818 IS DISTORTING EXCESSIVELY.
   ELEMENT 1818 IS DISTORTING EXCESSIVELY.
   ELEMENT 2012 IS DISTORTING EXCESSIVELY.
   ELEMENT 2012 IS DISTORTING EXCESSIVELY.
   ELEMENT 2042 IS DISTORTING EXCESSIVELY.
   ELEMENT 2042 IS DISTORTING EXCESSIVELY.
   ELEMENT 2042 IS DISTORTING EXCESSIVELY.
   ELEMENT 2042 IS DISTORTING EXCESSIVELY.
   ELEMENT 2043 IS DISTORTING EXCESSIVELY.
   ELEMENT 2043 IS DISTORTING EXCESSIVELY.
   ELEMENT 2043 IS DISTORTING EXCESSIVELY.
   ELEMENT 2043 IS DISTORTING EXCESSIVELY.
   ELEMENT 2043 IS DISTORTING EXCESSIVELY.
   ELEMENT 2043 IS DISTORTING EXCESSIVELY.
   ELEMENT 2043 IS DISTORTING EXCESSIVELY.
   ELEMENT 2043 IS DISTORTING EXCESSIVELY.
   ELEMENT 2237 IS DISTORTING EXCESSIVELY.
   ELEMENT 2237 IS DISTORTING EXCESSIVELY.
   ELEMENT 2237 IS DISTORTING EXCESSIVELY.
   ELEMENT 2237 IS DISTORTING EXCESSIVELY.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2713 IS OVERCLOSED BY 22.4769E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2714 IS OVERCLOSED BY 49.1057E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2715 IS OVERCLOSED BY 29.4871E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2731 IS OVERCLOSED BY 18.0262E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2732 IS OVERCLOSED BY 17.8454E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2733 IS OVERCLOSED BY 17.4518E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2734 IS OVERCLOSED BY 16.8368E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2735 IS OVERCLOSED BY 15.9985E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2736 IS OVERCLOSED BY 15.1632E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2737 IS OVERCLOSED BY 14.6576E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2738 IS OVERCLOSED BY 14.3384E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2739 IS OVERCLOSED BY 14.0034E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2740 IS OVERCLOSED BY 13.6573E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2741 IS OVERCLOSED BY 13.3017E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2742 IS OVERCLOSED BY 12.9280E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2743 IS OVERCLOSED BY 12.5396E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2744 IS OVERCLOSED BY 12.1370E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2745 IS OVERCLOSED BY 11.7197E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2746 IS OVERCLOSED BY 11.2877E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2747 IS OVERCLOSED BY 10.8410E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2748 IS OVERCLOSED BY 10.3793E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2749 IS OVERCLOSED BY 9.90260E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2750 IS OVERCLOSED BY 9.41034E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2751 IS OVERCLOSED BY 8.90215E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2752 IS OVERCLOSED BY 8.37725E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2753 IS OVERCLOSED BY 7.83452E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2754 IS OVERCLOSED BY 7.27444E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2755 IS OVERCLOSED BY 6.69042E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2756 IS OVERCLOSED BY 6.07798E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2757 IS OVERCLOSED BY 5.45022E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2758 IS OVERCLOSED BY 4.78305E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2759 IS OVERCLOSED BY 4.06106E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2760 IS OVERCLOSED BY 3.26897E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2761 IS OVERCLOSED BY 2.37223E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2762 IS OVERCLOSED BY 1.32657E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2763 IS OVERCLOSED BY 451.151E+06   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4069 IS OVERCLOSED BY 22.4769E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4070 IS OVERCLOSED BY 49.1057E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4071 IS OVERCLOSED BY 29.4871E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4087 IS OVERCLOSED BY 18.0262E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4088 IS OVERCLOSED BY 17.8454E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4089 IS OVERCLOSED BY 17.4518E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4090 IS OVERCLOSED BY 16.8368E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4091 IS OVERCLOSED BY 15.9985E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4092 IS OVERCLOSED BY 15.1632E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4093 IS OVERCLOSED BY 14.6576E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4094 IS OVERCLOSED BY 14.3384E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4095 IS OVERCLOSED BY 14.0034E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4096 IS OVERCLOSED BY 13.6573E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4097 IS OVERCLOSED BY 13.3017E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4098 IS OVERCLOSED BY 12.9280E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4099 IS OVERCLOSED BY 12.5396E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4100 IS OVERCLOSED BY 12.1370E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4101 IS OVERCLOSED BY 11.7197E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4102 IS OVERCLOSED BY 11.2877E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4103 IS OVERCLOSED BY 10.8410E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4104 IS OVERCLOSED BY 10.3793E+09  
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4105 IS OVERCLOSED BY 9.90260E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4106 IS OVERCLOSED BY 9.41034E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4107 IS OVERCLOSED BY 8.90215E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4108 IS OVERCLOSED BY 8.37725E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4109 IS OVERCLOSED BY 7.83452E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4110 IS OVERCLOSED BY 7.27444E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4111 IS OVERCLOSED BY 6.69042E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4112 IS OVERCLOSED BY 6.07798E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4113 IS OVERCLOSED BY 5.45022E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4114 IS OVERCLOSED BY 4.78305E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4115 IS OVERCLOSED BY 4.06106E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4116 IS OVERCLOSED BY 3.26897E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4117 IS OVERCLOSED BY 2.37223E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4118 IS OVERCLOSED BY 1.32657E+09 
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4119 IS OVERCLOSED BY 451.151E+06   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 108 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS

 ***WARNING: INCOMPATIBLE MODES CONTRIBUTION TO THE ELEMENT STIFFNESS IS 
             SINGULAR IN 4 ELEMENTS.
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.
 

 ***NOTE: MATERIAL CALCULATIONS FAILED TO CONVERGE OR WERE NOT ATTEMPTED AT ONE 
          OR MORE POINTS. CONVERGENCE IS JUDGED UNLIKELY.
 

 ***NOTE: SEVERE CONTACT OVERCLOSURES EXIST. CONVERGENCE IS JUDGED UNLIKELY.
 

 ***ERROR: TOO MANY ATTEMPTS MADE FOR THIS INCREMENT



     ANALYSIS SUMMARY:
     TOTAL OF          1  INCREMENTS
                       5  CUTBACKS IN AUTOMATIC INCREMENTATION
                      18  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                      18  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                      18  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       7  WARNING MESSAGES DURING USER INPUT PROCESSING
                     403  WARNING MESSAGES DURING ANALYSIS
                     387  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                       6  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       1  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =      9.0    
       SYSTEM TIME (SEC)    =     0.70    
       TOTAL CPU TIME (SEC) =      9.7    
       WALLCLOCK TIME (SEC) =           10
       MEMORY PEAK (GB)     =            0

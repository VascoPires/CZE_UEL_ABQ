1

   Abaqus 2023                                  Date 25-Oct-2026   Time 18:23:56
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

   MAX. PENETRATION ERROR 2.33312E-06 AT NODE 4071 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -1.24259E+03 AT NODE 2730 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.249       TIME AVG. FORCE       0.249    
 LARGEST SCALED RESIDUAL FORCE       974.       AT NODE       2730   DOF  2
  CORRESPONDING RESIDUAL FORCE       974.    
 LARGEST INCREMENT OF DISP.        -0.180       AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -0.180       AT NODE       5427   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.164    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     1.937E-04   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     1.937E-04
 LARGEST INCREMENT OF ROTATION     -1.617E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.617E-03   AT NODE       5425   DOF  6
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10890
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.454e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

 ***WARNING: OVERCONSTRAINT CHECKS: THE MAXIMUM NUMBER OF ZERO PIVOT-BASED 
             OVERCONSTRAINT CHECKS MESSAGES PRINTED FOR THIS STEP HAS BEEN 
             REACHED. THE OUTPUT OF THESE MESSAGES IS SUPRESSED UNTIL THE END 
             OF THE STEP TO AVOID POTENTIALLY LARGE INCREASES IN THE SYSTEM 
             TIME NEEDED TO COMPLETE THE ANALYSIS.

                   26 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    7 POINTS CHANGED FROM OPEN TO CLOSED
                   19 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR 9.71657E-15 AT NODE 2713 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -273.281       AT NODE 2759 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      3.773E-02   TIME AVG. FORCE       3.773E-02
 LARGEST SCALED RESIDUAL FORCE       205.       AT NODE       2759   DOF  2
  CORRESPONDING RESIDUAL FORCE       205.    
 LARGEST INCREMENT OF DISP.        -0.180       AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -6.349E-05   AT NODE       2486   DOF  1
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      2.486E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -3.612E-05   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -3.612E-05
 LARGEST INCREMENT OF ROTATION     -1.617E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -3.688E-08   AT NODE       5425   DOF  6
          ALL ROTATION CORRECTIONS ARE ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10878
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.424e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   40 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   29 POINTS CHANGED FROM OPEN TO CLOSED
                   11 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     3

   MAX. PENETRATION ERROR 1.00241E-09 AT NODE 2751 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -130.394       AT NODE 2718 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      3.705E-02   TIME AVG. FORCE       3.705E-02
 LARGEST SCALED RESIDUAL FORCE       119.       AT NODE       2718   DOF  2
  CORRESPONDING RESIDUAL FORCE       119.    
 LARGEST INCREMENT OF DISP.        -0.180       AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.517E-07   AT NODE       4081   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      2.441E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     6.925E-06   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     6.925E-06
 LARGEST INCREMENT OF ROTATION     -1.617E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.393E-08   AT NODE       5425   DOF  6
          ALL ROTATION CORRECTIONS ARE ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10896
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.453e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.

     NEGATIVE EIGENVALUES MEAN THAT THE SYSTEM MATRIX IS NOT 
     POSITIVE DEFINITE:
     FOR EXAMPLE, A BIFURCATION (BUCKLING) LOAD MAY HAVE BEEN EXCEEDED.

                   34 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   10 POINTS CHANGED FROM OPEN TO CLOSED
                   24 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     4

   MAX. PENETRATION ERROR -13.4853E-09  AT NODE 2728 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -2.55637E+12 AT NODE 2726 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      6.721E+07   TIME AVG. FORCE       6.721E+07
 LARGEST SCALED RESIDUAL FORCE      1.917E+12   AT NODE       2726   DOF  2
  CORRESPONDING RESIDUAL FORCE      1.917E+12
 LARGEST INCREMENT OF DISP.        -0.180       AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -2.354E-05   AT NODE        708   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 4082 
             IS 52.5387E+09  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      4.428E+07
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     8.708E-06   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     8.708E-06
 LARGEST INCREMENT OF ROTATION     -1.617E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -7.362E-07   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10882
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.432e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 13 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   39 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   15 POINTS CHANGED FROM OPEN TO CLOSED
                   24 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     5

   MAX. PENETRATION ERROR -30.2043E-09  AT NODE 4072 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -456.515E+09   AT NODE 2716 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.992E+07   TIME AVG. FORCE       1.992E+07
 LARGEST SCALED RESIDUAL FORCE      3.899E+11   AT NODE       2716   DOF  2
  CORRESPONDING RESIDUAL FORCE      3.899E+11
 LARGEST INCREMENT OF DISP.        -0.180       AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        2.712E-05   AT NODE       2713   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 4072 
             IS 15.5676E+09  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.313E+07
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     1.474E-05   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     1.474E-05
 LARGEST INCREMENT OF ROTATION     -1.622E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -5.011E-06   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10873
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.417e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 6 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   51 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   40 POINTS CHANGED FROM OPEN TO CLOSED
                   11 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     6

   MAX. PENETRATION ERROR 32.1469E-09  AT NODE 4091 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -359.632       AT NODE 4098 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      5.080E-02   TIME AVG. FORCE       5.080E-02
 LARGEST SCALED RESIDUAL FORCE      -270.       AT NODE       1160   DOF  2
  CORRESPONDING RESIDUAL FORCE      -270.    
 LARGEST INCREMENT OF DISP.        -0.180       AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.280E-05   AT NODE       2525   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      3.347E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     5.361E-07   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     5.361E-07
 LARGEST INCREMENT OF ROTATION     -1.622E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     4.530E-08   AT NODE       5425   DOF  6
          ALL ROTATION CORRECTIONS ARE ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10902
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.453e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   44 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   15 POINTS CHANGED FROM OPEN TO CLOSED
                   29 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     7

   MAX. PENETRATION ERROR 395.146E-09   AT NODE 4076 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -1.04324E+03 AT NODE 4082 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.155       TIME AVG. FORCE       0.155    
 LARGEST SCALED RESIDUAL FORCE      -895.       AT NODE       1144   DOF  2
  CORRESPONDING RESIDUAL FORCE      -895.    
 LARGEST INCREMENT OF DISP.        -0.180       AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        8.321E-05   AT NODE       2713   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.102    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -1.010E-04   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -1.010E-04
 LARGEST INCREMENT OF ROTATION     -1.632E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -9.545E-06   AT NODE       5425   DOF  6
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10888
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.439e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   48 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   27 POINTS CHANGED FROM OPEN TO CLOSED
                   21 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     8

   MAX. PENETRATION ERROR 17.0471E-15  AT NODE 4077 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -1.14570E+03 AT NODE 4084 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.102       TIME AVG. FORCE       0.102    
 LARGEST SCALED RESIDUAL FORCE       859.       AT NODE       2728   DOF  2
  CORRESPONDING RESIDUAL FORCE       859.    
 LARGEST INCREMENT OF DISP.        -0.180       AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.072E-06   AT NODE       2528   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      6.736E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     2.458E-07   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     2.458E-07
 LARGEST INCREMENT OF ROTATION     -1.632E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     2.602E-08   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10894
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.442e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =            1

                   42 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   19 POINTS CHANGED FROM OPEN TO CLOSED
                   23 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     9

   MAX. PENETRATION ERROR 22.1540E-12  AT NODE 4100 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -18.6238E+03  AT NODE 4073 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                       1.32       TIME AVG. FORCE        1.32    
 LARGEST SCALED RESIDUAL FORCE      1.517E+04   AT NODE       2717   DOF  2
  CORRESPONDING RESIDUAL FORCE      1.517E+04
 LARGEST INCREMENT OF DISP.        -0.180       AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        7.768E-07   AT NODE       1170   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.867    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -1.139E-11   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -1.139E-11
 LARGEST INCREMENT OF ROTATION     -1.632E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     2.065E-08   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10890
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.439e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   28 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    3 POINTS CHANGED FROM OPEN TO CLOSED
                   25 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    10

   MAX. PENETRATION ERROR -230.106E-03   AT NODE 4072 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -4.18532E+18 AT NODE 2725 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          PENETRATION ERROR TOO LARGE COMPARED TO DISPLACEMENT INCREMENT.

 AVERAGE FORCE                      1.374E+14   TIME AVG. FORCE       1.374E+14
 LARGEST SCALED RESIDUAL FORCE     -3.118E+18   AT NODE       1143   DOF  2
  CORRESPONDING RESIDUAL FORCE     -3.118E+18
 LARGEST INCREMENT OF DISP.          1.63       AT NODE       1143   DOF  2
 LARGEST CORRECTION TO DISP.         1.65       AT NODE       1143   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 4081 
             IS 686.206E+12   TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      9.054E+13
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     -3.71       AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     -3.71    
 LARGEST INCREMENT OF ROTATION      5.007E-02   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     5.170E-02   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10868
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.408e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 10 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE 1138 
             D.O.F. 2 RATIO = 2.66356E+12.
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE 1139 
             D.O.F. 1 RATIO = 111.499E+09  .
 

 ***WARNING: SOLVER PROBLEM. NUMERICAL SINGULARITY WHEN PROCESSING NODE 1139 
             D.O.F. 2 RATIO = 21.9067E+09 .
   ELEMENT 1 IS DISTORTING EXCESSIVELY.
   ELEMENT 1 IS DISTORTING EXCESSIVELY.
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
   ELEMENT 11 IS DISTORTING EXCESSIVELY.
   ELEMENT 11 IS DISTORTING EXCESSIVELY.
   ELEMENT 11 IS DISTORTING EXCESSIVELY.
   ELEMENT 11 IS DISTORTING EXCESSIVELY.
   ELEMENT 12 IS DISTORTING EXCESSIVELY.
   ELEMENT 12 IS DISTORTING EXCESSIVELY.
   ELEMENT 12 IS DISTORTING EXCESSIVELY.
   ELEMENT 12 IS DISTORTING EXCESSIVELY.
   ELEMENT 228 IS DISTORTING EXCESSIVELY.
   ELEMENT 228 IS DISTORTING EXCESSIVELY.
   ELEMENT 228 IS DISTORTING EXCESSIVELY.
   ELEMENT 228 IS DISTORTING EXCESSIVELY.
   ELEMENT 228 IS DISTORTING EXCESSIVELY.
   ELEMENT 228 IS DISTORTING EXCESSIVELY.
   ELEMENT 229 IS DISTORTING EXCESSIVELY.
   ELEMENT 229 IS DISTORTING EXCESSIVELY.
   ELEMENT 229 IS DISTORTING EXCESSIVELY.
   ELEMENT 229 IS DISTORTING EXCESSIVELY.
   ELEMENT 229 IS DISTORTING EXCESSIVELY.
   ELEMENT 229 IS DISTORTING EXCESSIVELY.
   ELEMENT 229 IS DISTORTING EXCESSIVELY.
   ELEMENT 229 IS DISTORTING EXCESSIVELY.
   ELEMENT 230 IS DISTORTING EXCESSIVELY.
   ELEMENT 230 IS DISTORTING EXCESSIVELY.
   ELEMENT 230 IS DISTORTING EXCESSIVELY.
   ELEMENT 230 IS DISTORTING EXCESSIVELY.
   ELEMENT 230 IS DISTORTING EXCESSIVELY.
   ELEMENT 230 IS DISTORTING EXCESSIVELY.
   ELEMENT 235 IS DISTORTING EXCESSIVELY.
   ELEMENT 235 IS DISTORTING EXCESSIVELY.
   ELEMENT 235 IS DISTORTING EXCESSIVELY.
   ELEMENT 235 IS DISTORTING EXCESSIVELY.
   ELEMENT 235 IS DISTORTING EXCESSIVELY.
   ELEMENT 235 IS DISTORTING EXCESSIVELY.
   ELEMENT 235 IS DISTORTING EXCESSIVELY.
   ELEMENT 235 IS DISTORTING EXCESSIVELY.
   ELEMENT 236 IS DISTORTING EXCESSIVELY.
   ELEMENT 236 IS DISTORTING EXCESSIVELY.
   ELEMENT 236 IS DISTORTING EXCESSIVELY.
   ELEMENT 236 IS DISTORTING EXCESSIVELY.
   ELEMENT 236 IS DISTORTING EXCESSIVELY.
   ELEMENT 236 IS DISTORTING EXCESSIVELY.
   ELEMENT 237 IS DISTORTING EXCESSIVELY.
   ELEMENT 237 IS DISTORTING EXCESSIVELY.
   ELEMENT 237 IS DISTORTING EXCESSIVELY.
   ELEMENT 237 IS DISTORTING EXCESSIVELY.
   ELEMENT 237 IS DISTORTING EXCESSIVELY.
   ELEMENT 237 IS DISTORTING EXCESSIVELY.
   ELEMENT 237 IS DISTORTING EXCESSIVELY.
   ELEMENT 237 IS DISTORTING EXCESSIVELY.
   ELEMENT 238 IS DISTORTING EXCESSIVELY.
   ELEMENT 238 IS DISTORTING EXCESSIVELY.
   ELEMENT 239 IS DISTORTING EXCESSIVELY.
   ELEMENT 239 IS DISTORTING EXCESSIVELY.
   ELEMENT 239 IS DISTORTING EXCESSIVELY.
   ELEMENT 239 IS DISTORTING EXCESSIVELY.
   ELEMENT 240 IS DISTORTING EXCESSIVELY.
   ELEMENT 240 IS DISTORTING EXCESSIVELY.
   ELEMENT 240 IS DISTORTING EXCESSIVELY.
   ELEMENT 240 IS DISTORTING EXCESSIVELY.
   ELEMENT 244 IS DISTORTING EXCESSIVELY.
   ELEMENT 244 IS DISTORTING EXCESSIVELY.
   ELEMENT 244 IS DISTORTING EXCESSIVELY.
   ELEMENT 244 IS DISTORTING EXCESSIVELY.
   ELEMENT 244 IS DISTORTING EXCESSIVELY.
   ELEMENT 244 IS DISTORTING EXCESSIVELY.
   ELEMENT 244 IS DISTORTING EXCESSIVELY.
   ELEMENT 244 IS DISTORTING EXCESSIVELY.
   ELEMENT 245 IS DISTORTING EXCESSIVELY.
   ELEMENT 245 IS DISTORTING EXCESSIVELY.
   ELEMENT 245 IS DISTORTING EXCESSIVELY.
   ELEMENT 245 IS DISTORTING EXCESSIVELY.
   ELEMENT 245 IS DISTORTING EXCESSIVELY.
   ELEMENT 245 IS DISTORTING EXCESSIVELY.
   ELEMENT 245 IS DISTORTING EXCESSIVELY.
   ELEMENT 245 IS DISTORTING EXCESSIVELY.
   ELEMENT 453 IS DISTORTING EXCESSIVELY.
   ELEMENT 453 IS DISTORTING EXCESSIVELY.
   ELEMENT 453 IS DISTORTING EXCESSIVELY.
   ELEMENT 453 IS DISTORTING EXCESSIVELY.
   ELEMENT 453 IS DISTORTING EXCESSIVELY.
   ELEMENT 453 IS DISTORTING EXCESSIVELY.
   ELEMENT 453 IS DISTORTING EXCESSIVELY.
   ELEMENT 453 IS DISTORTING EXCESSIVELY.
   ELEMENT 454 IS DISTORTING EXCESSIVELY.
   ELEMENT 454 IS DISTORTING EXCESSIVELY.
   ELEMENT 454 IS DISTORTING EXCESSIVELY.
   ELEMENT 454 IS DISTORTING EXCESSIVELY.
   ELEMENT 454 IS DISTORTING EXCESSIVELY.
   ELEMENT 454 IS DISTORTING EXCESSIVELY.
   ELEMENT 454 IS DISTORTING EXCESSIVELY.
   ELEMENT 454 IS DISTORTING EXCESSIVELY.
   ELEMENT 455 IS DISTORTING EXCESSIVELY.
   ELEMENT 455 IS DISTORTING EXCESSIVELY.
   ELEMENT 455 IS DISTORTING EXCESSIVELY.
   ELEMENT 455 IS DISTORTING EXCESSIVELY.
   ELEMENT 455 IS DISTORTING EXCESSIVELY.
   ELEMENT 455 IS DISTORTING EXCESSIVELY.
   ELEMENT 455 IS DISTORTING EXCESSIVELY.
   ELEMENT 455 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 460 IS DISTORTING EXCESSIVELY.
   ELEMENT 460 IS DISTORTING EXCESSIVELY.
   ELEMENT 460 IS DISTORTING EXCESSIVELY.
   ELEMENT 460 IS DISTORTING EXCESSIVELY.
   ELEMENT 460 IS DISTORTING EXCESSIVELY.
   ELEMENT 460 IS DISTORTING EXCESSIVELY.
   ELEMENT 460 IS DISTORTING EXCESSIVELY.
   ELEMENT 460 IS DISTORTING EXCESSIVELY.
   ELEMENT 461 IS DISTORTING EXCESSIVELY.
   ELEMENT 461 IS DISTORTING EXCESSIVELY.
   ELEMENT 461 IS DISTORTING EXCESSIVELY.
   ELEMENT 461 IS DISTORTING EXCESSIVELY.
   ELEMENT 462 IS DISTORTING EXCESSIVELY.
   ELEMENT 462 IS DISTORTING EXCESSIVELY.
   ELEMENT 462 IS DISTORTING EXCESSIVELY.
   ELEMENT 462 IS DISTORTING EXCESSIVELY.
   ELEMENT 464 IS DISTORTING EXCESSIVELY.
   ELEMENT 464 IS DISTORTING EXCESSIVELY.
   ELEMENT 464 IS DISTORTING EXCESSIVELY.
   ELEMENT 464 IS DISTORTING EXCESSIVELY.
   ELEMENT 465 IS DISTORTING EXCESSIVELY.
   ELEMENT 465 IS DISTORTING EXCESSIVELY.
   ELEMENT 465 IS DISTORTING EXCESSIVELY.
   ELEMENT 465 IS DISTORTING EXCESSIVELY.
   ELEMENT 466 IS DISTORTING EXCESSIVELY.
   ELEMENT 466 IS DISTORTING EXCESSIVELY.
   ELEMENT 466 IS DISTORTING EXCESSIVELY.
   ELEMENT 466 IS DISTORTING EXCESSIVELY.
   ELEMENT 469 IS DISTORTING EXCESSIVELY.
   ELEMENT 469 IS DISTORTING EXCESSIVELY.
   ELEMENT 469 IS DISTORTING EXCESSIVELY.
   ELEMENT 469 IS DISTORTING EXCESSIVELY.
   ELEMENT 469 IS DISTORTING EXCESSIVELY.
   ELEMENT 469 IS DISTORTING EXCESSIVELY.
   ELEMENT 469 IS DISTORTING EXCESSIVELY.
   ELEMENT 469 IS DISTORTING EXCESSIVELY.
   ELEMENT 470 IS DISTORTING EXCESSIVELY.
   ELEMENT 470 IS DISTORTING EXCESSIVELY.
   ELEMENT 470 IS DISTORTING EXCESSIVELY.
   ELEMENT 470 IS DISTORTING EXCESSIVELY.
   ELEMENT 470 IS DISTORTING EXCESSIVELY.
   ELEMENT 470 IS DISTORTING EXCESSIVELY.
   ELEMENT 470 IS DISTORTING EXCESSIVELY.
   ELEMENT 470 IS DISTORTING EXCESSIVELY.
   ELEMENT 678 IS DISTORTING EXCESSIVELY.
   ELEMENT 678 IS DISTORTING EXCESSIVELY.
   ELEMENT 678 IS DISTORTING EXCESSIVELY.
   ELEMENT 678 IS DISTORTING EXCESSIVELY.
   ELEMENT 678 IS DISTORTING EXCESSIVELY.
   ELEMENT 678 IS DISTORTING EXCESSIVELY.
   ELEMENT 678 IS DISTORTING EXCESSIVELY.
   ELEMENT 678 IS DISTORTING EXCESSIVELY.
   ELEMENT 679 IS DISTORTING EXCESSIVELY.
   ELEMENT 679 IS DISTORTING EXCESSIVELY.
   ELEMENT 679 IS DISTORTING EXCESSIVELY.
   ELEMENT 679 IS DISTORTING EXCESSIVELY.
   ELEMENT 679 IS DISTORTING EXCESSIVELY.
   ELEMENT 679 IS DISTORTING EXCESSIVELY.
   ELEMENT 679 IS DISTORTING EXCESSIVELY.
   ELEMENT 679 IS DISTORTING EXCESSIVELY.
   ELEMENT 680 IS DISTORTING EXCESSIVELY.
   ELEMENT 680 IS DISTORTING EXCESSIVELY.
   ELEMENT 680 IS DISTORTING EXCESSIVELY.
   ELEMENT 680 IS DISTORTING EXCESSIVELY.
   ELEMENT 680 IS DISTORTING EXCESSIVELY.
   ELEMENT 680 IS DISTORTING EXCESSIVELY.
   ELEMENT 680 IS DISTORTING EXCESSIVELY.
   ELEMENT 680 IS DISTORTING EXCESSIVELY.
   ELEMENT 682 IS DISTORTING EXCESSIVELY.
   ELEMENT 682 IS DISTORTING EXCESSIVELY.
   ELEMENT 682 IS DISTORTING EXCESSIVELY.
   ELEMENT 682 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 685 IS DISTORTING EXCESSIVELY.
   ELEMENT 685 IS DISTORTING EXCESSIVELY.
   ELEMENT 685 IS DISTORTING EXCESSIVELY.
   ELEMENT 685 IS DISTORTING EXCESSIVELY.
   ELEMENT 685 IS DISTORTING EXCESSIVELY.
   ELEMENT 685 IS DISTORTING EXCESSIVELY.
   ELEMENT 686 IS DISTORTING EXCESSIVELY.
   ELEMENT 686 IS DISTORTING EXCESSIVELY.
   ELEMENT 687 IS DISTORTING EXCESSIVELY.
   ELEMENT 687 IS DISTORTING EXCESSIVELY.
   ELEMENT 687 IS DISTORTING EXCESSIVELY.
   ELEMENT 687 IS DISTORTING EXCESSIVELY.
   ELEMENT 688 IS DISTORTING EXCESSIVELY.
   ELEMENT 688 IS DISTORTING EXCESSIVELY.
   ELEMENT 689 IS DISTORTING EXCESSIVELY.
   ELEMENT 689 IS DISTORTING EXCESSIVELY.
   ELEMENT 691 IS DISTORTING EXCESSIVELY.
   ELEMENT 691 IS DISTORTING EXCESSIVELY.
   ELEMENT 691 IS DISTORTING EXCESSIVELY.
   ELEMENT 691 IS DISTORTING EXCESSIVELY.
   ELEMENT 693 IS DISTORTING EXCESSIVELY.
   ELEMENT 693 IS DISTORTING EXCESSIVELY.
   ELEMENT 693 IS DISTORTING EXCESSIVELY.
   ELEMENT 693 IS DISTORTING EXCESSIVELY.
   ELEMENT 694 IS DISTORTING EXCESSIVELY.
   ELEMENT 694 IS DISTORTING EXCESSIVELY.
   ELEMENT 694 IS DISTORTING EXCESSIVELY.
   ELEMENT 694 IS DISTORTING EXCESSIVELY.
   ELEMENT 694 IS DISTORTING EXCESSIVELY.
   ELEMENT 694 IS DISTORTING EXCESSIVELY.
   ELEMENT 694 IS DISTORTING EXCESSIVELY.
   ELEMENT 694 IS DISTORTING EXCESSIVELY.
   ELEMENT 695 IS DISTORTING EXCESSIVELY.
   ELEMENT 695 IS DISTORTING EXCESSIVELY.
   ELEMENT 695 IS DISTORTING EXCESSIVELY.
   ELEMENT 695 IS DISTORTING EXCESSIVELY.
   ELEMENT 695 IS DISTORTING EXCESSIVELY.
   ELEMENT 695 IS DISTORTING EXCESSIVELY.
   ELEMENT 695 IS DISTORTING EXCESSIVELY.
   ELEMENT 695 IS DISTORTING EXCESSIVELY.
   ELEMENT 903 IS DISTORTING EXCESSIVELY.
   ELEMENT 903 IS DISTORTING EXCESSIVELY.
   ELEMENT 905 IS DISTORTING EXCESSIVELY.
   ELEMENT 905 IS DISTORTING EXCESSIVELY.
   ELEMENT 905 IS DISTORTING EXCESSIVELY.
   ELEMENT 905 IS DISTORTING EXCESSIVELY.
   ELEMENT 907 IS DISTORTING EXCESSIVELY.
   ELEMENT 907 IS DISTORTING EXCESSIVELY.
   ELEMENT 907 IS DISTORTING EXCESSIVELY.
   ELEMENT 907 IS DISTORTING EXCESSIVELY.
   ELEMENT 907 IS DISTORTING EXCESSIVELY.
   ELEMENT 907 IS DISTORTING EXCESSIVELY.
   ELEMENT 907 IS DISTORTING EXCESSIVELY.
   ELEMENT 907 IS DISTORTING EXCESSIVELY.
   ELEMENT 910 IS DISTORTING EXCESSIVELY.
   ELEMENT 910 IS DISTORTING EXCESSIVELY.
   ELEMENT 910 IS DISTORTING EXCESSIVELY.
   ELEMENT 910 IS DISTORTING EXCESSIVELY.
   ELEMENT 910 IS DISTORTING EXCESSIVELY.
   ELEMENT 910 IS DISTORTING EXCESSIVELY.
   ELEMENT 911 IS DISTORTING EXCESSIVELY.
   ELEMENT 911 IS DISTORTING EXCESSIVELY.
   ELEMENT 911 IS DISTORTING EXCESSIVELY.
   ELEMENT 911 IS DISTORTING EXCESSIVELY.
   ELEMENT 913 IS DISTORTING EXCESSIVELY.
   ELEMENT 913 IS DISTORTING EXCESSIVELY.
   ELEMENT 916 IS DISTORTING EXCESSIVELY.
   ELEMENT 916 IS DISTORTING EXCESSIVELY.
   ELEMENT 917 IS DISTORTING EXCESSIVELY.
   ELEMENT 917 IS DISTORTING EXCESSIVELY.
   ELEMENT 917 IS DISTORTING EXCESSIVELY.
   ELEMENT 917 IS DISTORTING EXCESSIVELY.
   ELEMENT 918 IS DISTORTING EXCESSIVELY.
   ELEMENT 918 IS DISTORTING EXCESSIVELY.
   ELEMENT 918 IS DISTORTING EXCESSIVELY.
   ELEMENT 918 IS DISTORTING EXCESSIVELY.
   ELEMENT 918 IS DISTORTING EXCESSIVELY.
   ELEMENT 918 IS DISTORTING EXCESSIVELY.
   ELEMENT 918 IS DISTORTING EXCESSIVELY.
   ELEMENT 918 IS DISTORTING EXCESSIVELY.
   ELEMENT 919 IS DISTORTING EXCESSIVELY.
   ELEMENT 919 IS DISTORTING EXCESSIVELY.
   ELEMENT 919 IS DISTORTING EXCESSIVELY.
   ELEMENT 919 IS DISTORTING EXCESSIVELY.
   ELEMENT 921 IS DISTORTING EXCESSIVELY.
   ELEMENT 921 IS DISTORTING EXCESSIVELY.
   ELEMENT 921 IS DISTORTING EXCESSIVELY.
   ELEMENT 921 IS DISTORTING EXCESSIVELY.
   ELEMENT 1130 IS DISTORTING EXCESSIVELY.
   ELEMENT 1130 IS DISTORTING EXCESSIVELY.
   ELEMENT 1132 IS DISTORTING EXCESSIVELY.
   ELEMENT 1132 IS DISTORTING EXCESSIVELY.
   ELEMENT 1132 IS DISTORTING EXCESSIVELY.
   ELEMENT 1132 IS DISTORTING EXCESSIVELY.
   ELEMENT 1132 IS DISTORTING EXCESSIVELY.
   ELEMENT 1132 IS DISTORTING EXCESSIVELY.
   ELEMENT 1132 IS DISTORTING EXCESSIVELY.
   ELEMENT 1132 IS DISTORTING EXCESSIVELY.
   ELEMENT 1136 IS DISTORTING EXCESSIVELY.
   ELEMENT 1136 IS DISTORTING EXCESSIVELY.
   ELEMENT 1136 IS DISTORTING EXCESSIVELY.
   ELEMENT 1136 IS DISTORTING EXCESSIVELY.
   ELEMENT 1136 IS DISTORTING EXCESSIVELY.
   ELEMENT 1136 IS DISTORTING EXCESSIVELY.
   ELEMENT 1136 IS DISTORTING EXCESSIVELY.
   ELEMENT 1136 IS DISTORTING EXCESSIVELY.
   ELEMENT 1137 IS DISTORTING EXCESSIVELY.
   ELEMENT 1137 IS DISTORTING EXCESSIVELY.
   ELEMENT 1137 IS DISTORTING EXCESSIVELY.
   ELEMENT 1137 IS DISTORTING EXCESSIVELY.
   ELEMENT 1138 IS DISTORTING EXCESSIVELY.
   ELEMENT 1138 IS DISTORTING EXCESSIVELY.
   ELEMENT 1138 IS DISTORTING EXCESSIVELY.
   ELEMENT 1138 IS DISTORTING EXCESSIVELY.
   ELEMENT 1138 IS DISTORTING EXCESSIVELY.
   ELEMENT 1138 IS DISTORTING EXCESSIVELY.
   ELEMENT 1138 IS DISTORTING EXCESSIVELY.
   ELEMENT 1138 IS DISTORTING EXCESSIVELY.
   ELEMENT 1139 IS DISTORTING EXCESSIVELY.
   ELEMENT 1139 IS DISTORTING EXCESSIVELY.
   ELEMENT 1139 IS DISTORTING EXCESSIVELY.
   ELEMENT 1139 IS DISTORTING EXCESSIVELY.
   ELEMENT 1139 IS DISTORTING EXCESSIVELY.
   ELEMENT 1139 IS DISTORTING EXCESSIVELY.
   ELEMENT 1139 IS DISTORTING EXCESSIVELY.
   ELEMENT 1139 IS DISTORTING EXCESSIVELY.
   ELEMENT 1140 IS DISTORTING EXCESSIVELY.
   ELEMENT 1140 IS DISTORTING EXCESSIVELY.
   ELEMENT 1140 IS DISTORTING EXCESSIVELY.
   ELEMENT 1140 IS DISTORTING EXCESSIVELY.
   ELEMENT 1141 IS DISTORTING EXCESSIVELY.
   ELEMENT 1141 IS DISTORTING EXCESSIVELY.
   ELEMENT 1141 IS DISTORTING EXCESSIVELY.
   ELEMENT 1141 IS DISTORTING EXCESSIVELY.
   ELEMENT 1142 IS DISTORTING EXCESSIVELY.
   ELEMENT 1142 IS DISTORTING EXCESSIVELY.
   ELEMENT 1142 IS DISTORTING EXCESSIVELY.
   ELEMENT 1142 IS DISTORTING EXCESSIVELY.
   ELEMENT 1142 IS DISTORTING EXCESSIVELY.
   ELEMENT 1142 IS DISTORTING EXCESSIVELY.
   ELEMENT 1142 IS DISTORTING EXCESSIVELY.
   ELEMENT 1142 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1143 IS DISTORTING EXCESSIVELY.
   ELEMENT 1146 IS DISTORTING EXCESSIVELY.
   ELEMENT 1146 IS DISTORTING EXCESSIVELY.
   ELEMENT 1357 IS DISTORTING EXCESSIVELY.
   ELEMENT 1357 IS DISTORTING EXCESSIVELY.
   ELEMENT 1357 IS DISTORTING EXCESSIVELY.
   ELEMENT 1357 IS DISTORTING EXCESSIVELY.
   ELEMENT 1361 IS DISTORTING EXCESSIVELY.
   ELEMENT 1361 IS DISTORTING EXCESSIVELY.
   ELEMENT 1361 IS DISTORTING EXCESSIVELY.
   ELEMENT 1361 IS DISTORTING EXCESSIVELY.
   ELEMENT 1363 IS DISTORTING EXCESSIVELY.
   ELEMENT 1363 IS DISTORTING EXCESSIVELY.
   ELEMENT 1363 IS DISTORTING EXCESSIVELY.
   ELEMENT 1363 IS DISTORTING EXCESSIVELY.
   ELEMENT 1363 IS DISTORTING EXCESSIVELY.
   ELEMENT 1363 IS DISTORTING EXCESSIVELY.
   ELEMENT 1363 IS DISTORTING EXCESSIVELY.
   ELEMENT 1363 IS DISTORTING EXCESSIVELY.
   ELEMENT 1366 IS DISTORTING EXCESSIVELY.
   ELEMENT 1366 IS DISTORTING EXCESSIVELY.
   ELEMENT 1366 IS DISTORTING EXCESSIVELY.
   ELEMENT 1366 IS DISTORTING EXCESSIVELY.
   ELEMENT 1367 IS DISTORTING EXCESSIVELY.
   ELEMENT 1367 IS DISTORTING EXCESSIVELY.
   ELEMENT 1367 IS DISTORTING EXCESSIVELY.
   ELEMENT 1367 IS DISTORTING EXCESSIVELY.
   ELEMENT 1367 IS DISTORTING EXCESSIVELY.
   ELEMENT 1367 IS DISTORTING EXCESSIVELY.
   ELEMENT 1367 IS DISTORTING EXCESSIVELY.
   ELEMENT 1367 IS DISTORTING EXCESSIVELY.
   ELEMENT 1588 IS DISTORTING EXCESSIVELY.
   ELEMENT 1588 IS DISTORTING EXCESSIVELY.
   ELEMENT 1588 IS DISTORTING EXCESSIVELY.
   ELEMENT 1588 IS DISTORTING EXCESSIVELY.
   ELEMENT 1592 IS DISTORTING EXCESSIVELY.
   ELEMENT 1592 IS DISTORTING EXCESSIVELY.
   ELEMENT 1592 IS DISTORTING EXCESSIVELY.
   ELEMENT 1592 IS DISTORTING EXCESSIVELY.
   ELEMENT 1592 IS DISTORTING EXCESSIVELY.
   ELEMENT 1592 IS DISTORTING EXCESSIVELY.
   ELEMENT 1592 IS DISTORTING EXCESSIVELY.
   ELEMENT 1592 IS DISTORTING EXCESSIVELY.
   ELEMENT 1593 IS DISTORTING EXCESSIVELY.
   ELEMENT 1593 IS DISTORTING EXCESSIVELY.
   ELEMENT 1817 IS DISTORTING EXCESSIVELY.
   ELEMENT 1817 IS DISTORTING EXCESSIVELY.
   ELEMENT 1817 IS DISTORTING EXCESSIVELY.
   ELEMENT 1817 IS DISTORTING EXCESSIVELY.
   ELEMENT 1817 IS DISTORTING EXCESSIVELY.
   ELEMENT 1817 IS DISTORTING EXCESSIVELY.
   ELEMENT 1817 IS DISTORTING EXCESSIVELY.
   ELEMENT 1817 IS DISTORTING EXCESSIVELY.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2715 IS OVERCLOSED BY 1.18491     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2716 IS OVERCLOSED BY 1.18291     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2717 IS OVERCLOSED BY 1.63276     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2726 IS OVERCLOSED BY 4.39832     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2727 IS OVERCLOSED BY 3.19240     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2729 IS OVERCLOSED BY 9.44078     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2730 IS OVERCLOSED BY 1.83256     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4071 IS OVERCLOSED BY 1.18491     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4072 IS OVERCLOSED BY 1.18291     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4073 IS OVERCLOSED BY 1.63276     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4082 IS OVERCLOSED BY 4.39832     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4083 IS OVERCLOSED BY 3.19240     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4085 IS OVERCLOSED BY 9.44078     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4086 IS OVERCLOSED BY 1.83256     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 400 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.
 

 ***NOTE: SEVERE CONTACT OVERCLOSURES EXIST. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     1 STARTS. ATTEMPT NUMBER  2, TIME INCREMENT  5.000E-03
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10938
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.473e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   47 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   47 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 145.820E-09   AT NODE 2715 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -9.56360E+03 AT NODE 2759 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                       1.12       TIME AVG. FORCE        1.12    
 LARGEST SCALED RESIDUAL FORCE     -8.218E+03   AT NODE       1177   DOF  2
  CORRESPONDING RESIDUAL FORCE     -8.218E+03
 LARGEST INCREMENT OF DISP.        -4.500E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -4.500E-02   AT NODE       5427   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.737    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     1.354E-05   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     1.354E-05
 LARGEST INCREMENT OF ROTATION     -4.041E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -4.041E-04   AT NODE       5425   DOF  6
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10891
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.454e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 3 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   38 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   25 POINTS CHANGED FROM OPEN TO CLOSED
                   13 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR -1.16053E-15 AT NODE 4115 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -103.057       AT NODE 2732 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      9.610E-03   TIME AVG. FORCE       9.610E-03
 LARGEST SCALED RESIDUAL FORCE       77.3       AT NODE       2732   DOF  2
  CORRESPONDING RESIDUAL FORCE       77.3    
 LARGEST INCREMENT OF DISP.        -4.500E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -3.966E-06   AT NODE        904   DOF  1
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT    -5.391E-07   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -5.391E-07
 LARGEST INCREMENT OF ROTATION     -4.041E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     4.224E-10   AT NODE       5425   DOF  6
          ALL ROTATION CORRECTIONS ARE ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10903
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.464e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   45 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   17 POINTS CHANGED FROM OPEN TO CLOSED
                   28 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     3

   MAX. PENETRATION ERROR 57.8501E-12  AT NODE 4111 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -2.57735E+03 AT NODE 2745 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.201       TIME AVG. FORCE       0.201    
 LARGEST SCALED RESIDUAL FORCE      1.933E+03   AT NODE       2745   DOF  2
  CORRESPONDING RESIDUAL FORCE      1.933E+03
 LARGEST INCREMENT OF DISP.        -4.500E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -2.902E-08   AT NODE        700   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.132    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     3.102E-06   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     3.102E-06
 LARGEST INCREMENT OF ROTATION     -4.041E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.798E-09   AT NODE       5425   DOF  6
          ALL ROTATION CORRECTIONS ARE ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10892
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.449e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   24 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    6 POINTS CHANGED FROM OPEN TO CLOSED
                   18 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     4

   MAX. PENETRATION ERROR 133.947E-12   AT NODE 2737 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -5.41601E+03 AT NODE 2746 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.318       TIME AVG. FORCE       0.318    
 LARGEST SCALED RESIDUAL FORCE     -4.348E+03   AT NODE       1159   DOF  2
  CORRESPONDING RESIDUAL FORCE     -4.348E+03
 LARGEST INCREMENT OF DISP.        -4.500E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        6.773E-06   AT NODE       2713   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.209    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     1.130E-05   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     1.130E-05
 LARGEST INCREMENT OF ROTATION     -4.054E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.261E-06   AT NODE       5425   DOF  6
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10880
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.422e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   36 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   24 POINTS CHANGED FROM OPEN TO CLOSED
                   12 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     5

   MAX. PENETRATION ERROR 1.13088E-09 AT NODE 4103 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -2.96975E+03 AT NODE 2751 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.112       TIME AVG. FORCE       0.112    
 LARGEST SCALED RESIDUAL FORCE      2.227E+03   AT NODE       2751   DOF  2
  CORRESPONDING RESIDUAL FORCE      2.227E+03
 LARGEST INCREMENT OF DISP.        -4.500E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -1.086E-07   AT NODE       4147   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      7.401E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     2.718E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     2.718E-09
 LARGEST INCREMENT OF ROTATION     -4.054E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.365E-08   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10892
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.437e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            1

                   42 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   16 POINTS CHANGED FROM OPEN TO CLOSED
                   26 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     6

   MAX. PENETRATION ERROR 36.8944E-15  AT NODE 4071 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -1.03815E+03 AT NODE 2747 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      8.260E-02   TIME AVG. FORCE       8.260E-02
 LARGEST SCALED RESIDUAL FORCE       787.       AT NODE       2722   DOF  2
  CORRESPONDING RESIDUAL FORCE       787.    
 LARGEST INCREMENT OF DISP.        -4.500E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        2.176E-06   AT NODE       1168   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      5.442E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -8.173E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -8.173E-09
 LARGEST INCREMENT OF ROTATION     -4.053E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     9.582E-08   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10882
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.427e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   50 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   33 POINTS CHANGED FROM OPEN TO CLOSED
                   17 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     7

   MAX. PENETRATION ERROR 3.91324E-09 AT NODE 2745 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -62.5468E+06  AT NODE 2734 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.649E+03   TIME AVG. FORCE       1.649E+03
 LARGEST SCALED RESIDUAL FORCE      4.691E+07   AT NODE       2734   DOF  2
  CORRESPONDING RESIDUAL FORCE      4.691E+07
 LARGEST INCREMENT OF DISP.        -4.500E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -5.907E-07   AT NODE       2516   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 4090 
             IS 5.15532E+06 TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES IN 
             THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.086E+03
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -5.360E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -5.360E-09
 LARGEST INCREMENT OF ROTATION     -4.053E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -2.019E-08   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     1 STARTS. ATTEMPT NUMBER  3, TIME INCREMENT  1.250E-03
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10938
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.473e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            1

                   47 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   47 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 9.11375E-09 AT NODE 2715 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -2.39090E+03 AT NODE 2759 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.280       TIME AVG. FORCE       0.280    
 LARGEST SCALED RESIDUAL FORCE     -2.054E+03   AT NODE       1177   DOF  2
  CORRESPONDING RESIDUAL FORCE     -2.054E+03
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -1.125E-02   AT NODE       5427   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.184    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     7.821E-07   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     7.821E-07
 LARGEST INCREMENT OF ROTATION     -1.010E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.010E-04   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10891
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.454e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   41 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   26 POINTS CHANGED FROM OPEN TO CLOSED
                   15 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR 180.845E-18   AT NODE 4118 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -69.3737      AT NODE 4102 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      4.166E-03   TIME AVG. FORCE       4.166E-03
 LARGEST SCALED RESIDUAL FORCE       52.0       AT NODE       2746   DOF  2
  CORRESPONDING RESIDUAL FORCE       52.0    
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -2.480E-07   AT NODE        452   DOF  1
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT    -6.844E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -6.844E-09
 LARGEST INCREMENT OF ROTATION     -1.010E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -3.758E-10   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10902
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.464e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   41 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   18 POINTS CHANGED FROM OPEN TO CLOSED
                   23 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     3

   MAX. PENETRATION ERROR 2.42036E-12 AT NODE 2748 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -6.92770E+03 AT NODE 2725 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.268       TIME AVG. FORCE       0.268    
 LARGEST SCALED RESIDUAL FORCE      5.636E+03   AT NODE       2725   DOF  2
  CORRESPONDING RESIDUAL FORCE      5.636E+03
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        5.618E-09   AT NODE       2739   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.176    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     1.675E-06   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     1.675E-06
 LARGEST INCREMENT OF ROTATION     -1.010E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     2.891E-10   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10897
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.458e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   34 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   15 POINTS CHANGED FROM OPEN TO CLOSED
                   19 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     4

   MAX. PENETRATION ERROR -549.691E-18   AT NODE 4081 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -7.14856E+03 AT NODE 4079 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.331       TIME AVG. FORCE       0.331    
 LARGEST SCALED RESIDUAL FORCE      6.176E+03   AT NODE       2723   DOF  2
  CORRESPONDING RESIDUAL FORCE      6.176E+03
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        3.179E-11   AT NODE       2762   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.218    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     2.108E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     2.108E-09
 LARGEST INCREMENT OF ROTATION     -1.010E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -8.949E-13   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10893
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.452e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   47 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   24 POINTS CHANGED FROM OPEN TO CLOSED
                   23 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     5

   MAX. PENETRATION ERROR 1.01944E-12 AT NODE 2734 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -3.09522E+03 AT NODE 2720 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.196       TIME AVG. FORCE       0.196    
 LARGEST SCALED RESIDUAL FORCE      2.460E+03   AT NODE       2720   DOF  2
  CORRESPONDING RESIDUAL FORCE      2.460E+03
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.693E-06   AT NODE       2713   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.129    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     2.554E-07   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     2.554E-07
 LARGEST INCREMENT OF ROTATION     -1.014E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -3.156E-07   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10894
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.455e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   36 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   11 POINTS CHANGED FROM OPEN TO CLOSED
                   25 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     6

   MAX. PENETRATION ERROR 4.57310E-12 AT NODE 2738 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -602.999       AT NODE 4073 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      2.196E-02   TIME AVG. FORCE       2.196E-02
 LARGEST SCALED RESIDUAL FORCE      -452.       AT NODE       1135   DOF  2
  CORRESPONDING RESIDUAL FORCE      -452.    
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        5.855E-07   AT NODE       2779   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.447E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -1.022E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -1.022E-09
 LARGEST INCREMENT OF ROTATION     -1.013E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     3.971E-08   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10880
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.426e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   49 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   36 POINTS CHANGED FROM OPEN TO CLOSED
                   13 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     7

   MAX. PENETRATION ERROR 42.4946E-09  AT NODE 4075 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -22.4348      AT NODE 2742 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      3.043E-03   TIME AVG. FORCE       3.043E-03
 LARGEST SCALED RESIDUAL FORCE      -18.1       AT NODE       1160   DOF  2
  CORRESPONDING RESIDUAL FORCE      -18.1    
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        5.309E-06   AT NODE       2713   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT    -3.937E-07   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -3.937E-07
 LARGEST INCREMENT OF ROTATION     -1.020E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -6.925E-07   AT NODE       5425   DOF  6
          THE MOMENT    IS ZERO EVERYWHERE BUT THE MOMENT    RESIDUAL OR
          THE ROTATION CORRECTION IS NON-ZERO
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10903
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.458e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   32 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    6 POINTS CHANGED FROM OPEN TO CLOSED
                   26 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     8

   MAX. PENETRATION ERROR 2.84266E-12 AT NODE 4091 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -20.9445E+03  AT NODE 2733 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                       1.06       TIME AVG. FORCE        1.06    
 LARGEST SCALED RESIDUAL FORCE      1.584E+04   AT NODE       2733   DOF  2
  CORRESPONDING RESIDUAL FORCE      1.584E+04
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        3.101E-07   AT NODE       1831   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 4089 
             IS 13.2753E+03  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.701    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     7.047E-11   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     7.047E-11
 LARGEST INCREMENT OF ROTATION     -1.020E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.966E-08   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10883
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.434e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            1
 

 ***WARNING: THE SYSTEM MATRIX HAS 3 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   26 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   17 POINTS CHANGED FROM OPEN TO CLOSED
                    9 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     9

   MAX. PENETRATION ERROR 1.36776E-12 AT NODE 4099 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -24.3950E+03  AT NODE 2734 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.790       TIME AVG. FORCE       0.790    
 LARGEST SCALED RESIDUAL FORCE     -1.890E+04   AT NODE       1152   DOF  2
  CORRESPONDING RESIDUAL FORCE     -1.890E+04
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        2.855E-10   AT NODE       2757   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.521    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -2.556E-13   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -2.556E-13
 LARGEST INCREMENT OF ROTATION     -1.020E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     8.039E-12   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10891
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.443e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   25 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    9 POINTS CHANGED FROM OPEN TO CLOSED
                   16 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    10

   MAX. PENETRATION ERROR -995.731E-18   AT NODE 4089 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -2.22814E+03 AT NODE 2718 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      5.286E-02   TIME AVG. FORCE       5.286E-02
 LARGEST SCALED RESIDUAL FORCE     -1.671E+03   AT NODE       1136   DOF  2
  CORRESPONDING RESIDUAL FORCE     -1.671E+03
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -3.493E-10   AT NODE       1150   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      3.483E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     3.551E-13   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     3.551E-13
 LARGEST INCREMENT OF ROTATION     -1.020E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.748E-11   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10884
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.434e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   28 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   15 POINTS CHANGED FROM OPEN TO CLOSED
                   13 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    11

   MAX. PENETRATION ERROR 165.883E-18   AT NODE 4075 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -1.51815E+03 AT NODE 4078 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      7.844E-02   TIME AVG. FORCE       7.844E-02
 LARGEST SCALED RESIDUAL FORCE      1.183E+03   AT NODE       2722   DOF  2
  CORRESPONDING RESIDUAL FORCE      1.183E+03
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        4.408E-10   AT NODE       2558   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      5.168E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -9.007E-14   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -9.007E-14
 LARGEST INCREMENT OF ROTATION     -1.020E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.187E-11   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10886
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.435e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   21 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    5 POINTS CHANGED FROM OPEN TO CLOSED
                   16 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    12

   MAX. PENETRATION ERROR 4.23039E-12 AT NODE 2735 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -3.00619E+03 AT NODE 4073 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.197       TIME AVG. FORCE       0.197    
 LARGEST SCALED RESIDUAL FORCE     -2.255E+03   AT NODE       1135   DOF  2
  CORRESPONDING RESIDUAL FORCE     -2.255E+03
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        9.235E-06   AT NODE       2713   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.130    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     2.700E-08   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     2.700E-08
 LARGEST INCREMENT OF ROTATION     -1.030E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.016E-06   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10875
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.420e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   23 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    6 POINTS CHANGED FROM OPEN TO CLOSED
                   17 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    13

   MAX. PENETRATION ERROR 2.45706E-09 AT NODE 2730 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -2.32201E+03 AT NODE 4077 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.115       TIME AVG. FORCE       0.115    
 LARGEST SCALED RESIDUAL FORCE     -1.742E+03   AT NODE       1139   DOF  2
  CORRESPONDING RESIDUAL FORCE     -1.742E+03
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.625E-05   AT NODE       4155   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      7.604E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     1.795E-08   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     1.795E-08
 LARGEST INCREMENT OF ROTATION     -1.039E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -9.447E-07   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10864
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.397e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            1

                   34 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   23 POINTS CHANGED FROM OPEN TO CLOSED
                   11 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    14

   MAX. PENETRATION ERROR 675.765E-09   AT NODE 2719 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -31.2160      AT NODE 2733 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      2.717E-03   TIME AVG. FORCE       2.717E-03
 LARGEST SCALED RESIDUAL FORCE       23.4       AT NODE       2733   DOF  2
  CORRESPONDING RESIDUAL FORCE       23.4    
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.670E-05   AT NODE       4069   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT     9.555E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     9.555E-09
 LARGEST INCREMENT OF ROTATION     -1.059E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.971E-06   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10876
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.409e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   43 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   32 POINTS CHANGED FROM OPEN TO CLOSED
                   11 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    15

   MAX. PENETRATION ERROR 107.686E-09   AT NODE 2722 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -33.4130      AT NODE 2734 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      4.049E-03   TIME AVG. FORCE       4.049E-03
 LARGEST SCALED RESIDUAL FORCE      -26.9       AT NODE       1152   DOF  2
  CORRESPONDING RESIDUAL FORCE      -26.9    
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.608E-05   AT NODE       2510   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT    -4.600E-11   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -4.600E-11
 LARGEST INCREMENT OF ROTATION     -1.050E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     9.589E-07   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10897
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.443e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   31 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   11 POINTS CHANGED FROM OPEN TO CLOSED
                   20 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    16

   MAX. PENETRATION ERROR 96.2772E-18  AT NODE 4079 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -30.2756      AT NODE 2748 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      6.073E-03   TIME AVG. FORCE       6.073E-03
 LARGEST SCALED RESIDUAL FORCE       24.7       AT NODE       2730   DOF  2
  CORRESPONDING RESIDUAL FORCE       24.7    
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -7.749E-08   AT NODE       2496   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT     1.540E-12   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     1.540E-12
 LARGEST INCREMENT OF ROTATION     -1.050E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.964E-09   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10888
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.437e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   25 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   10 POINTS CHANGED FROM OPEN TO CLOSED
                   15 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    17

   MAX. PENETRATION ERROR -4.33681E-18 AT NODE 2735 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -98.2918      AT NODE 2738 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.069E-02   TIME AVG. FORCE       1.069E-02
 LARGEST SCALED RESIDUAL FORCE       79.8       AT NODE       2738   DOF  2
  CORRESPONDING RESIDUAL FORCE       79.8    
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        6.919E-10   AT NODE       4786   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT    -2.263E-15   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -2.263E-15
 LARGEST INCREMENT OF ROTATION     -1.050E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.657E-11   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10883
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.429e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   40 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   26 POINTS CHANGED FROM OPEN TO CLOSED
                   14 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    18

   MAX. PENETRATION ERROR 465.083E-15   AT NODE 4075 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -36.9214      AT NODE 2751 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      3.782E-03   TIME AVG. FORCE       3.782E-03
 LARGEST SCALED RESIDUAL FORCE      -27.7       AT NODE       1169   DOF  2
  CORRESPONDING RESIDUAL FORCE      -27.7    
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        2.579E-05   AT NODE       2713   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT     1.705E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     1.705E-09
 LARGEST INCREMENT OF ROTATION     -1.063E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.301E-06   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10895
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.442e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   36 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   13 POINTS CHANGED FROM OPEN TO CLOSED
                   23 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    19

   MAX. PENETRATION ERROR 20.8401E-15  AT NODE 4090 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -682.456E+03   AT NODE 2734 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                       16.6       TIME AVG. FORCE        16.6    
 LARGEST SCALED RESIDUAL FORCE      5.118E+05   AT NODE       2734   DOF  2
  CORRESPONDING RESIDUAL FORCE      5.118E+05
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.138E-07   AT NODE       3434   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 4090 
             IS 208.094E+03   TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       10.9    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -2.649E-12   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -2.649E-12
 LARGEST INCREMENT OF ROTATION     -1.063E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     2.065E-09   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10885
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.432e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 3 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   23 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   10 POINTS CHANGED FROM OPEN TO CLOSED
                   13 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    20

   MAX. PENETRATION ERROR -8.49581E-15 AT NODE 4089 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -19.6646E+03  AT NODE 2739 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.987       TIME AVG. FORCE       0.987    
 LARGEST SCALED RESIDUAL FORCE     -1.475E+04   AT NODE       1157   DOF  2
  CORRESPONDING RESIDUAL FORCE     -1.475E+04
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        3.234E-10   AT NODE       4133   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 4095 
             IS 12.3549E+03  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.650    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     6.955E-14   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     6.955E-14
 LARGEST INCREMENT OF ROTATION     -1.063E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -4.510E-12   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10882
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.429e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 3 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   25 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   17 POINTS CHANGED FROM OPEN TO CLOSED
                    8 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    21

   MAX. PENETRATION ERROR 23.8290E-15  AT NODE 2734 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -594.640E+03   AT NODE 4090 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                       14.8       TIME AVG. FORCE        14.8    
 LARGEST SCALED RESIDUAL FORCE     -4.460E+05   AT NODE       1152   DOF  2
  CORRESPONDING RESIDUAL FORCE     -4.460E+05
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        2.459E-10   AT NODE       2765   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 2734 
             IS 185.102E+03   TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       9.72    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -5.531E-14   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -5.531E-14
 LARGEST INCREMENT OF ROTATION     -1.063E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -4.500E-12   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10891
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.438e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   26 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    9 POINTS CHANGED FROM OPEN TO CLOSED
                   17 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    22

   MAX. PENETRATION ERROR -35.0306E-15  AT NODE 2734 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -22.1921E+03  AT NODE 4095 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                       1.28       TIME AVG. FORCE        1.28    
 LARGEST SCALED RESIDUAL FORCE      1.751E+04   AT NODE       2739   DOF  2
  CORRESPONDING RESIDUAL FORCE      1.751E+04
 LARGEST INCREMENT OF DISP.        -1.125E-02   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        9.649E-11   AT NODE       4764   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 2739 
             IS 16.0729E+03  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.845    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     1.200E-13   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     1.200E-13
 LARGEST INCREMENT OF ROTATION     -1.063E-04   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     5.696E-12   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10883
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.429e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 3 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   32 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    7 POINTS CHANGED FROM OPEN TO CLOSED
                   25 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    23

   MAX. PENETRATION ERROR -1.17701E-03 AT NODE 4082 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -67.9504E+15  AT NODE 2726 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.727E+12   TIME AVG. FORCE       1.727E+12
 LARGEST SCALED RESIDUAL FORCE     -5.093E+16   AT NODE       1144   DOF  2
  CORRESPONDING RESIDUAL FORCE     -5.093E+16
 LARGEST INCREMENT OF DISP.         0.276       AT NODE        463   DOF  2
 LARGEST CORRECTION TO DISP.        0.277       AT NODE        463   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 4082 
             IS 70.3566E+12  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.138E+12
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -0.262       AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -0.262    
 LARGEST INCREMENT OF ROTATION      1.222E-02   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.233E-02   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 

 ***NOTE: THE SOLUTION APPEARS TO BE DIVERGING. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     1 STARTS. ATTEMPT NUMBER  4, TIME INCREMENT  3.125E-04
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10938
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.473e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   47 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   47 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 569.609E-12   AT NODE 2715 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -597.725       AT NODE 2759 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      6.991E-02   TIME AVG. FORCE       6.991E-02
 LARGEST SCALED RESIDUAL FORCE      -514.       AT NODE       1177   DOF  2
  CORRESPONDING RESIDUAL FORCE      -514.    
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -2.812E-03   AT NODE       5427   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      4.606E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     4.788E-08   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     4.788E-08
 LARGEST INCREMENT OF ROTATION     -2.526E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -2.526E-05   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10891
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.454e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   42 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   28 POINTS CHANGED FROM OPEN TO CLOSED
                   14 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR 60.0648E-18  AT NODE 2758 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -6.11960     AT NODE 2746 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      7.295E-04   TIME AVG. FORCE       7.295E-04
 LARGEST SCALED RESIDUAL FORCE       4.59       AT NODE       2746   DOF  2
  CORRESPONDING RESIDUAL FORCE       4.59    
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -1.549E-08   AT NODE        904   DOF  1
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT    -8.075E-12   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -8.075E-12
 LARGEST INCREMENT OF ROTATION     -2.526E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -6.642E-12   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10905
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.466e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   36 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    9 POINTS CHANGED FROM OPEN TO CLOSED
                   27 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     3

   MAX. PENETRATION ERROR -16.5510E-18  AT NODE 2714 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -22.4746      AT NODE 2756 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      2.164E-03   TIME AVG. FORCE       2.164E-03
 LARGEST SCALED RESIDUAL FORCE      -16.9       AT NODE       1174   DOF  2
  CORRESPONDING RESIDUAL FORCE      -16.9    
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        2.927E-11   AT NODE       4115   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT     6.408E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     6.408E-09
 LARGEST INCREMENT OF ROTATION     -2.526E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.153E-12   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10887
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.445e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.20000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   49 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   32 POINTS CHANGED FROM OPEN TO CLOSED
                   17 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     4

   MAX. PENETRATION ERROR 2.16912E-15 AT NODE 4070 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -21.6280      AT NODE 2727 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      2.149E-03   TIME AVG. FORCE       2.149E-03
 LARGEST SCALED RESIDUAL FORCE       18.6       AT NODE       2727   DOF  2
  CORRESPONDING RESIDUAL FORCE       18.6    
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        4.234E-07   AT NODE       2713   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT     3.074E-08   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     3.074E-08
 LARGEST INCREMENT OF ROTATION     -2.533E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -7.575E-08   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10902
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.459e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   35 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    8 POINTS CHANGED FROM OPEN TO CLOSED
                   27 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     5

   MAX. PENETRATION ERROR 51.4542E-15  AT NODE 2739 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -26.0588      AT NODE 2750 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      3.064E-03   TIME AVG. FORCE       3.064E-03
 LARGEST SCALED RESIDUAL FORCE      -21.7       AT NODE       1168   DOF  2
  CORRESPONDING RESIDUAL FORCE      -21.7    
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.194E-09   AT NODE       1838   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT    -2.563E-11   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -2.563E-11
 LARGEST INCREMENT OF ROTATION     -2.533E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     7.259E-11   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10883
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.439e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   20 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   11 POINTS CHANGED FROM OPEN TO CLOSED
                    9 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     6

   MAX. PENETRATION ERROR -1.08409E-15 AT NODE 4079 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -49.7528E+03  AT NODE 4081 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                       1.11       TIME AVG. FORCE        1.11    
 LARGEST SCALED RESIDUAL FORCE      3.731E+04   AT NODE       2725   DOF  2
  CORRESPONDING RESIDUAL FORCE      3.731E+04
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        2.638E-11   AT NODE       4114   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 2725 
             IS 55.5035E+03  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.731    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -2.705E-11   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -2.705E-11
 LARGEST INCREMENT OF ROTATION     -2.533E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.124E-12   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10885
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.438e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   35 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   23 POINTS CHANGED FROM OPEN TO CLOSED
                   12 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     7

   MAX. PENETRATION ERROR 971.225E-15   AT NODE 4091 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -3.90235E+03 AT NODE 2724 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      8.619E-02   TIME AVG. FORCE       8.619E-02
 LARGEST SCALED RESIDUAL FORCE     -2.927E+03   AT NODE       1142   DOF  2
  CORRESPONDING RESIDUAL FORCE     -2.927E+03
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.326E-06   AT NODE       4069   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      5.679E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -2.463E-08   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -2.463E-08
 LARGEST INCREMENT OF ROTATION     -2.550E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.653E-07   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10896
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.445e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   27 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    6 POINTS CHANGED FROM OPEN TO CLOSED
                   21 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     8

   MAX. PENETRATION ERROR 153.941E-12   AT NODE 4082 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -11.0355      AT NODE 2730 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.399E-03   TIME AVG. FORCE       1.399E-03
 LARGEST SCALED RESIDUAL FORCE      -8.96       AT NODE       1148   DOF  2
  CORRESPONDING RESIDUAL FORCE      -8.96    
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        5.726E-08   AT NODE       2768   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT    -4.287E-12   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -4.287E-12
 LARGEST INCREMENT OF ROTATION     -2.550E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     7.103E-10   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10881
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.426e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   49 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   34 POINTS CHANGED FROM OPEN TO CLOSED
                   15 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     9

   MAX. PENETRATION ERROR 763.014E-15   AT NODE 4093 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -10.2772      AT NODE 4111 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      9.910E-04   TIME AVG. FORCE       9.910E-04
 LARGEST SCALED RESIDUAL FORCE      -7.71       AT NODE       1173   DOF  2
  CORRESPONDING RESIDUAL FORCE      -7.71    
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        2.268E-06   AT NODE       2713   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT     2.010E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     2.010E-09
 LARGEST INCREMENT OF ROTATION     -2.570E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -2.033E-07   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10900
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.446e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   33 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    7 POINTS CHANGED FROM OPEN TO CLOSED
                   26 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    10

   MAX. PENETRATION ERROR 104.633E-12   AT NODE 4076 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -4.60982E+09 AT NODE 4081 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      2.621E+05   TIME AVG. FORCE       2.621E+05
 LARGEST SCALED RESIDUAL FORCE      3.525E+09   AT NODE       2725   DOF  2
  CORRESPONDING RESIDUAL FORCE      3.525E+09
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        5.568E-08   AT NODE       1829   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 2725 
             IS 13.1117E+09  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.727E+05
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -6.356E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -6.356E-09
 LARGEST INCREMENT OF ROTATION     -2.570E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     3.489E-09   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10881
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.430e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 15 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   44 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   21 POINTS CHANGED FROM OPEN TO CLOSED
                   23 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    11

   MAX. PENETRATION ERROR -349.032E-09   AT NODE 2754 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -24.7396E+12  AT NODE 4110 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      5.415E+08   TIME AVG. FORCE       5.415E+08
 LARGEST SCALED RESIDUAL FORCE      1.855E+13   AT NODE       2754   DOF  2
  CORRESPONDING RESIDUAL FORCE      1.855E+13
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        9.064E-05   AT NODE       1173   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 2754 
             IS 15.0736E+12  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      3.568E+08
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -1.034E-07   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -1.034E-07
 LARGEST INCREMENT OF ROTATION     -2.287E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     2.830E-06   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10879
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.403e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 4 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   48 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   26 POINTS CHANGED FROM OPEN TO CLOSED
                   22 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    12

   MAX. PENETRATION ERROR 557.828E-09   AT NODE 2755 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -4.58087E+12 AT NODE 4111 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.251E+08   TIME AVG. FORCE       1.251E+08
 LARGEST SCALED RESIDUAL FORCE     -3.436E+12   AT NODE       1173   DOF  2
  CORRESPONDING RESIDUAL FORCE     -3.436E+12
 LARGEST INCREMENT OF DISP.        -2.812E-03   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -4.108E-05   AT NODE       1160   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 2755 
             IS 4.37235E+12 TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES IN 
             THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      8.243E+07
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     3.718E-08   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     3.718E-08
 LARGEST INCREMENT OF ROTATION     -2.342E-05   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -5.489E-07   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10883
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.417e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 12 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   39 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   16 POINTS CHANGED FROM OPEN TO CLOSED
                   23 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    13

   MAX. PENETRATION ERROR 43.7334E-03  AT NODE 2750 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -6.15953E+09 AT NODE 4096 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          PENETRATION ERROR TOO LARGE COMPARED TO DISPLACEMENT INCREMENT.

 AVERAGE FORCE                      2.448E+05   TIME AVG. FORCE       2.448E+05
 LARGEST SCALED RESIDUAL FORCE      4.948E+09   AT NODE       2740   DOF  2
  CORRESPONDING RESIDUAL FORCE      4.948E+09
 LARGEST INCREMENT OF DISP.         0.172       AT NODE       1166   DOF  2
 LARGEST CORRECTION TO DISP.        0.172       AT NODE       1166   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 2740 
             IS 4.36201E+06 TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES IN 
             THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.613E+05
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -2.311E-02   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -2.311E-02
 LARGEST INCREMENT OF ROTATION      1.061E-02   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.064E-02   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10876
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.410e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 13 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   43 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   23 POINTS CHANGED FROM OPEN TO CLOSED
                   20 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    14

   MAX. PENETRATION ERROR 985.485E-06   AT NODE 2736 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -118.371E+06   AT NODE 2719 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          PENETRATION ERROR TOO LARGE COMPARED TO DISPLACEMENT INCREMENT.

 AVERAGE FORCE                      4.557E+03   TIME AVG. FORCE       4.557E+03
 LARGEST SCALED RESIDUAL FORCE     -8.992E+07   AT NODE       1137   DOF  2
  CORRESPONDING RESIDUAL FORCE     -8.992E+07
 LARGEST INCREMENT OF DISP.         8.722E-02   AT NODE       1151   DOF  2
 LARGEST CORRECTION TO DISP.        0.114       AT NODE       2804   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 4075 
             IS 396.663E+03   TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      3.003E+03
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -4.731E-03   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -4.731E-03
 LARGEST INCREMENT OF ROTATION      3.419E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -7.195E-03   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10879
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.407e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   41 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   21 POINTS CHANGED FROM OPEN TO CLOSED
                   20 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    15

   MAX. PENETRATION ERROR 191.536E-06   AT NODE 2720 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -288.377E+06   AT NODE 2750 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.196E+04   TIME AVG. FORCE       1.196E+04
 LARGEST SCALED RESIDUAL FORCE     -2.163E+08   AT NODE       1168   DOF  2
  CORRESPONDING RESIDUAL FORCE     -2.163E+08
 LARGEST INCREMENT OF DISP.         6.006E-02   AT NODE        475   DOF  2
 LARGEST CORRECTION TO DISP.       -2.925E-02   AT NODE       3183   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 4106 
             IS 2.41207E+06 TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES IN 
             THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      7.878E+03
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     3.118E-04   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     3.118E-04
 LARGEST INCREMENT OF ROTATION      2.357E-03   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.062E-03   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10880
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.420e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 12 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.
   ELEMENT 700 IS DISTORTING EXCESSIVELY.
   ELEMENT 700 IS DISTORTING EXCESSIVELY.
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
   ELEMENT 703 IS DISTORTING EXCESSIVELY.
   ELEMENT 703 IS DISTORTING EXCESSIVELY.
   ELEMENT 926 IS DISTORTING EXCESSIVELY.
   ELEMENT 926 IS DISTORTING EXCESSIVELY.
   ELEMENT 926 IS DISTORTING EXCESSIVELY.
   ELEMENT 926 IS DISTORTING EXCESSIVELY.
   ELEMENT 927 IS DISTORTING EXCESSIVELY.
   ELEMENT 927 IS DISTORTING EXCESSIVELY.
   ELEMENT 927 IS DISTORTING EXCESSIVELY.
   ELEMENT 927 IS DISTORTING EXCESSIVELY.
   ELEMENT 930 IS DISTORTING EXCESSIVELY.
   ELEMENT 930 IS DISTORTING EXCESSIVELY.
   ELEMENT 1150 IS DISTORTING EXCESSIVELY.
   ELEMENT 1150 IS DISTORTING EXCESSIVELY.
   ELEMENT 1151 IS DISTORTING EXCESSIVELY.
   ELEMENT 1151 IS DISTORTING EXCESSIVELY.
   ELEMENT 1151 IS DISTORTING EXCESSIVELY.
   ELEMENT 1151 IS DISTORTING EXCESSIVELY.
   ELEMENT 1152 IS DISTORTING EXCESSIVELY.
   ELEMENT 1152 IS DISTORTING EXCESSIVELY.
   ELEMENT 1152 IS DISTORTING EXCESSIVELY.
   ELEMENT 1152 IS DISTORTING EXCESSIVELY.
   ELEMENT 1153 IS DISTORTING EXCESSIVELY.
   ELEMENT 1153 IS DISTORTING EXCESSIVELY.
   ELEMENT 1153 IS DISTORTING EXCESSIVELY.
   ELEMENT 1153 IS DISTORTING EXCESSIVELY.
   ELEMENT 1155 IS DISTORTING EXCESSIVELY.
   ELEMENT 1155 IS DISTORTING EXCESSIVELY.
   ELEMENT 1155 IS DISTORTING EXCESSIVELY.
   ELEMENT 1155 IS DISTORTING EXCESSIVELY.
   ELEMENT 1156 IS DISTORTING EXCESSIVELY.
   ELEMENT 1156 IS DISTORTING EXCESSIVELY.
   ELEMENT 1156 IS DISTORTING EXCESSIVELY.
   ELEMENT 1156 IS DISTORTING EXCESSIVELY.
   ELEMENT 1378 IS DISTORTING EXCESSIVELY.
   ELEMENT 1378 IS DISTORTING EXCESSIVELY.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2713 IS OVERCLOSED BY 1.49805     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2714 IS OVERCLOSED BY 1.48351     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2715 IS OVERCLOSED BY 1.20784     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2716 IS OVERCLOSED BY 1.09052     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2717 IS OVERCLOSED BY 979.161E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2718 IS OVERCLOSED BY 812.334E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2719 IS OVERCLOSED BY 616.863E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2720 IS OVERCLOSED BY 413.136E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2741 IS OVERCLOSED BY 898.649E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4069 IS OVERCLOSED BY 1.49805     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4070 IS OVERCLOSED BY 1.48351     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4071 IS OVERCLOSED BY 1.20784     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4072 IS OVERCLOSED BY 1.09052     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4073 IS OVERCLOSED BY 979.161E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4074 IS OVERCLOSED BY 812.334E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4075 IS OVERCLOSED BY 616.863E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4076 IS OVERCLOSED BY 413.136E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4097 IS OVERCLOSED BY 898.649E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 48 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.
 

 ***NOTE: SEVERE CONTACT OVERCLOSURES EXIST. CONVERGENCE IS JUDGED UNLIKELY.


  INCREMENT     1 STARTS. ATTEMPT NUMBER  5, TIME INCREMENT  7.813E-05
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10938
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.473e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            1

                   47 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   47 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     1

   MAX. PENETRATION ERROR 35.6006E-12  AT NODE 2715 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -149.431       AT NODE 2759 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.748E-02   TIME AVG. FORCE       1.748E-02
 LARGEST SCALED RESIDUAL FORCE      -128.       AT NODE       1177   DOF  2
  CORRESPONDING RESIDUAL FORCE      -128.    
 LARGEST INCREMENT OF DISP.        -7.031E-04   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -7.031E-04   AT NODE       5427   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.152E-02
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     2.977E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     2.977E-09
 LARGEST INCREMENT OF ROTATION     -6.315E-06   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -6.315E-06   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10891
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.454e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   36 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   25 POINTS CHANGED FROM OPEN TO CLOSED
                   11 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     2

   MAX. PENETRATION ERROR 14.6638E-18  AT NODE 2754 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -1.97697     AT NODE 2736 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      1.834E-04   TIME AVG. FORCE       1.834E-04
 LARGEST SCALED RESIDUAL FORCE       1.48       AT NODE       2736   DOF  2
  CORRESPONDING RESIDUAL FORCE       1.48    
 LARGEST INCREMENT OF DISP.        -7.031E-04   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -9.684E-10   AT NODE        904   DOF  1
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT     6.050E-12   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     6.050E-12
 LARGEST INCREMENT OF ROTATION     -6.315E-06   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -3.562E-13   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10905
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.468e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =  0.10000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   37 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                    9 POINTS CHANGED FROM OPEN TO CLOSED
                   28 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     3

   MAX. PENETRATION ERROR -7.02021E-18 AT NODE 2740 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -245.323       AT NODE 4096 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      9.846E-03   TIME AVG. FORCE       9.846E-03
 LARGEST SCALED RESIDUAL FORCE      -199.       AT NODE       1158   DOF  2
  CORRESPONDING RESIDUAL FORCE      -199.    
 LARGEST INCREMENT OF DISP.        -7.031E-04   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        4.476E-13   AT NODE       4118   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT     1.078E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     1.078E-09
 LARGEST INCREMENT OF ROTATION     -6.315E-06   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     1.451E-14   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10886
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.444e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   51 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   33 POINTS CHANGED FROM OPEN TO CLOSED
                   18 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     4

   MAX. PENETRATION ERROR 246.207E-15   AT NODE 4093 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -4.46529     AT NODE 4089 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      3.223E-04   TIME AVG. FORCE       3.223E-04
 LARGEST SCALED RESIDUAL FORCE       3.35       AT NODE       2733   DOF  2
  CORRESPONDING RESIDUAL FORCE       3.35    
 LARGEST INCREMENT OF DISP.        -7.031E-04   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        1.058E-07   AT NODE       2713   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT     3.516E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     3.516E-09
 LARGEST INCREMENT OF ROTATION     -6.334E-06   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.895E-08   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10901
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.458e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   43 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   16 POINTS CHANGED FROM OPEN TO CLOSED
                   27 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     5

   MAX. PENETRATION ERROR 16.6677E-15  AT NODE 2738 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -18.3978E+03  AT NODE 4090 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      0.950       TIME AVG. FORCE       0.950    
 LARGEST SCALED RESIDUAL FORCE     -1.495E+04   AT NODE       1152   DOF  2
  CORRESPONDING RESIDUAL FORCE     -1.495E+04
 LARGEST INCREMENT OF DISP.        -7.031E-04   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        7.445E-10   AT NODE        492   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 2734 
             IS 190.083E+03   TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.626    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -1.744E-12   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -1.744E-12
 LARGEST INCREMENT OF ROTATION     -6.334E-06   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     2.910E-11   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10890
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.443e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 2 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   36 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   16 POINTS CHANGED FROM OPEN TO CLOSED
                   20 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     6

   MAX. PENETRATION ERROR -310.241E-15   AT NODE 4111 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -7.92762E+06 AT NODE 2755 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                       261.       TIME AVG. FORCE        261.    
 LARGEST SCALED RESIDUAL FORCE     -6.441E+06   AT NODE       1173   DOF  2
  CORRESPONDING RESIDUAL FORCE     -6.441E+06
 LARGEST INCREMENT OF DISP.        -7.031E-04   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        3.315E-07   AT NODE       2713   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 4111 
             IS 52.1992E+06  TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       172.    
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -1.539E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -1.539E-09
 LARGEST INCREMENT OF ROTATION     -6.373E-06   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -3.972E-08   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10886
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.424e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 3 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   55 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   29 POINTS CHANGED FROM OPEN TO CLOSED
                   26 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     7

   MAX. PENETRATION ERROR 1.08796E-09 AT NODE 4098 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -215.953       AT NODE 2737 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      5.183E-03   TIME AVG. FORCE       5.183E-03
 LARGEST SCALED RESIDUAL FORCE       162.       AT NODE       2737   DOF  2
  CORRESPONDING RESIDUAL FORCE       162.    
 LARGEST INCREMENT OF DISP.        -7.031E-04   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -1.993E-08   AT NODE       4765   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT    -1.022E-13   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -1.022E-13
 LARGEST INCREMENT OF ROTATION     -6.375E-06   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.404E-09   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10889
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.426e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

                   53 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   35 POINTS CHANGED FROM OPEN TO CLOSED
                   18 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     8

   MAX. PENETRATION ERROR 137.218E-15   AT NODE 2752 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -11.4034      AT NODE 4119 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      6.991E-04   TIME AVG. FORCE       6.991E-04
 LARGEST SCALED RESIDUAL FORCE       8.63       AT NODE       2763   DOF  2
  CORRESPONDING RESIDUAL FORCE       8.63    
 LARGEST INCREMENT OF DISP.        -7.031E-04   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        2.167E-08   AT NODE       2514   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 LARGEST SCALED RESIDUAL MOMENT     1.022E-14   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     1.022E-14
 LARGEST INCREMENT OF ROTATION     -6.374E-06   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     7.450E-10   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10906
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.453e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 1 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   46 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   17 POINTS CHANGED FROM OPEN TO CLOSED
                   29 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION     9

   MAX. PENETRATION ERROR 48.2407E-12  AT NODE 2725 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -1.18179E+09 AT NODE 4081 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      4.657E+04   TIME AVG. FORCE       4.657E+04
 LARGEST SCALED RESIDUAL FORCE      9.052E+08   AT NODE       2725   DOF  2
  CORRESPONDING RESIDUAL FORCE      9.052E+08
 LARGEST INCREMENT OF DISP.        -7.031E-04   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -1.968E-08   AT NODE       4768   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 2725 
             IS 9.31824E+09 TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES IN 
             THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      3.069E+04
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT     2.138E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT     2.138E-09
 LARGEST INCREMENT OF ROTATION     -6.375E-06   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -1.276E-09   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10894
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.453e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 7 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   49 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   26 POINTS CHANGED FROM OPEN TO CLOSED
                   23 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    10

   MAX. PENETRATION ERROR 27.4360E-12  AT NODE 4081 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -165.818E+06   AT NODE 2717 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      3.865E+03   TIME AVG. FORCE       3.865E+03
 LARGEST SCALED RESIDUAL FORCE      1.244E+08   AT NODE       2717   DOF  2
  CORRESPONDING RESIDUAL FORCE      1.244E+08
 LARGEST INCREMENT OF DISP.        -7.031E-04   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.        4.729E-08   AT NODE        459   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 4073 
             IS 773.343E+06   TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES 
             IN THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      2.547E+03
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -1.095E-08   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -1.095E-08
 LARGEST INCREMENT OF ROTATION     -6.370E-06   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION     4.963E-09   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10897
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.456e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.

                   49 SEVERE DISCONTINUITIES OCCURRED DURING THIS ITERATION.
                   22 POINTS CHANGED FROM OPEN TO CLOSED
                   27 POINTS CHANGED FROM CLOSED TO OPEN

               CONVERGENCE CHECKS FOR SEVERE DISCONTINUITY ITERATION    11

   MAX. PENETRATION ERROR 172.948E-12   AT NODE 2724 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
   MAX. CONTACT FORCE ERROR -543.033E+06   AT NODE 4076 OF CONTACT PAIR 
   (CRACKHIGH,CRACKLOW)
          THE ESTIMATED CONTACT FORCE ERROR IS OUTSIDE OF CONVERGENCE TOLERANCES.

 AVERAGE FORCE                      2.400E+04   TIME AVG. FORCE       2.400E+04
 LARGEST SCALED RESIDUAL FORCE     -4.671E+08   AT NODE       1138   DOF  2
  CORRESPONDING RESIDUAL FORCE     -4.671E+08
 LARGEST INCREMENT OF DISP.        -7.031E-04   AT NODE       5427   DOF  2
 LARGEST CORRECTION TO DISP.       -1.891E-08   AT NODE       1813   DOF  2
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.
 

 ***WARNING: THE AVERAGE FORCE USED TO ENFORCE CONTACT CONSTRAINTS AT NODE 2720 
             IS 4.80143E+09 TIMES LARGER THAN THE AVERAGE AT THE OTHER NODES IN 
             THE MODEL.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.581E+04
                                                (FROM FORCE     FIELD)
 LARGEST SCALED RESIDUAL MOMENT    -4.098E-09   AT NODE       5425   DOF  6
  CORRESPONDING RESIDUAL MOMENT    -4.098E-09
 LARGEST INCREMENT OF ROTATION     -6.371E-06   AT NODE       5425   DOF  6
 LARGEST CORRECTION TO ROTATION    -6.752E-10   AT NODE       5425   DOF  6
          THERE IS ZERO MOMENT    EVERYWHERE
 
	UNSYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     10892
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         1.451e+08

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0
 

 ***WARNING: THE SYSTEM MATRIX HAS 8 NEGATIVE EIGENVALUES.
             EXPLANATIONS ARE SUGGESTED AFTER THE FIRST OCCURRENCE OF THIS MESSAGE.
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
   ELEMENT 232 IS DISTORTING EXCESSIVELY.
   ELEMENT 232 IS DISTORTING EXCESSIVELY.
   ELEMENT 232 IS DISTORTING EXCESSIVELY.
   ELEMENT 232 IS DISTORTING EXCESSIVELY.
   ELEMENT 233 IS DISTORTING EXCESSIVELY.
   ELEMENT 233 IS DISTORTING EXCESSIVELY.
   ELEMENT 233 IS DISTORTING EXCESSIVELY.
   ELEMENT 233 IS DISTORTING EXCESSIVELY.
   ELEMENT 234 IS DISTORTING EXCESSIVELY.
   ELEMENT 234 IS DISTORTING EXCESSIVELY.
   ELEMENT 234 IS DISTORTING EXCESSIVELY.
   ELEMENT 234 IS DISTORTING EXCESSIVELY.
   ELEMENT 457 IS DISTORTING EXCESSIVELY.
   ELEMENT 457 IS DISTORTING EXCESSIVELY.
   ELEMENT 457 IS DISTORTING EXCESSIVELY.
   ELEMENT 457 IS DISTORTING EXCESSIVELY.
   ELEMENT 458 IS DISTORTING EXCESSIVELY.
   ELEMENT 458 IS DISTORTING EXCESSIVELY.
   ELEMENT 458 IS DISTORTING EXCESSIVELY.
   ELEMENT 458 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 459 IS DISTORTING EXCESSIVELY.
   ELEMENT 682 IS DISTORTING EXCESSIVELY.
   ELEMENT 682 IS DISTORTING EXCESSIVELY.
   ELEMENT 682 IS DISTORTING EXCESSIVELY.
   ELEMENT 682 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 684 IS DISTORTING EXCESSIVELY.
   ELEMENT 909 IS DISTORTING EXCESSIVELY.
   ELEMENT 909 IS DISTORTING EXCESSIVELY.
   ELEMENT 909 IS DISTORTING EXCESSIVELY.
   ELEMENT 909 IS DISTORTING EXCESSIVELY.
   ELEMENT 909 IS DISTORTING EXCESSIVELY.
   ELEMENT 909 IS DISTORTING EXCESSIVELY.
   ELEMENT 909 IS DISTORTING EXCESSIVELY.
   ELEMENT 909 IS DISTORTING EXCESSIVELY.
   ELEMENT 912 IS DISTORTING EXCESSIVELY.
   ELEMENT 912 IS DISTORTING EXCESSIVELY.
   ELEMENT 912 IS DISTORTING EXCESSIVELY.
   ELEMENT 912 IS DISTORTING EXCESSIVELY.
   ELEMENT 913 IS DISTORTING EXCESSIVELY.
   ELEMENT 913 IS DISTORTING EXCESSIVELY.
   ELEMENT 913 IS DISTORTING EXCESSIVELY.
   ELEMENT 913 IS DISTORTING EXCESSIVELY.
   ELEMENT 1132 IS DISTORTING EXCESSIVELY.
   ELEMENT 1132 IS DISTORTING EXCESSIVELY.
   ELEMENT 1134 IS DISTORTING EXCESSIVELY.
   ELEMENT 1134 IS DISTORTING EXCESSIVELY.
   ELEMENT 1134 IS DISTORTING EXCESSIVELY.
   ELEMENT 1134 IS DISTORTING EXCESSIVELY.
   ELEMENT 1134 IS DISTORTING EXCESSIVELY.
   ELEMENT 1134 IS DISTORTING EXCESSIVELY.
   ELEMENT 1137 IS DISTORTING EXCESSIVELY.
   ELEMENT 1137 IS DISTORTING EXCESSIVELY.
   ELEMENT 1137 IS DISTORTING EXCESSIVELY.
   ELEMENT 1137 IS DISTORTING EXCESSIVELY.
   ELEMENT 1138 IS DISTORTING EXCESSIVELY.
   ELEMENT 1138 IS DISTORTING EXCESSIVELY.
   ELEMENT 1138 IS DISTORTING EXCESSIVELY.
   ELEMENT 1138 IS DISTORTING EXCESSIVELY.
   ELEMENT 1357 IS DISTORTING EXCESSIVELY.
   ELEMENT 1357 IS DISTORTING EXCESSIVELY.
   ELEMENT 1357 IS DISTORTING EXCESSIVELY.
   ELEMENT 1357 IS DISTORTING EXCESSIVELY.
   ELEMENT 1359 IS DISTORTING EXCESSIVELY.
   ELEMENT 1359 IS DISTORTING EXCESSIVELY.
   ELEMENT 1359 IS DISTORTING EXCESSIVELY.
   ELEMENT 1359 IS DISTORTING EXCESSIVELY.
   ELEMENT 1359 IS DISTORTING EXCESSIVELY.
   ELEMENT 1359 IS DISTORTING EXCESSIVELY.
   ELEMENT 1359 IS DISTORTING EXCESSIVELY.
   ELEMENT 1359 IS DISTORTING EXCESSIVELY.
   ELEMENT 1582 IS DISTORTING EXCESSIVELY.
   ELEMENT 1582 IS DISTORTING EXCESSIVELY.
   ELEMENT 1582 IS DISTORTING EXCESSIVELY.
   ELEMENT 1582 IS DISTORTING EXCESSIVELY.
   ELEMENT 1583 IS DISTORTING EXCESSIVELY.
   ELEMENT 1583 IS DISTORTING EXCESSIVELY.
   ELEMENT 1583 IS DISTORTING EXCESSIVELY.
   ELEMENT 1583 IS DISTORTING EXCESSIVELY.
   ELEMENT 1584 IS DISTORTING EXCESSIVELY.
   ELEMENT 1584 IS DISTORTING EXCESSIVELY.
   ELEMENT 1584 IS DISTORTING EXCESSIVELY.
   ELEMENT 1584 IS DISTORTING EXCESSIVELY.
   ELEMENT 1807 IS DISTORTING EXCESSIVELY.
   ELEMENT 1807 IS DISTORTING EXCESSIVELY.
   ELEMENT 1807 IS DISTORTING EXCESSIVELY.
   ELEMENT 1807 IS DISTORTING EXCESSIVELY.
   ELEMENT 1808 IS DISTORTING EXCESSIVELY.
   ELEMENT 1808 IS DISTORTING EXCESSIVELY.
   ELEMENT 1808 IS DISTORTING EXCESSIVELY.
   ELEMENT 1808 IS DISTORTING EXCESSIVELY.
   ELEMENT 1809 IS DISTORTING EXCESSIVELY.
   ELEMENT 1809 IS DISTORTING EXCESSIVELY.
   ELEMENT 1809 IS DISTORTING EXCESSIVELY.
   ELEMENT 1809 IS DISTORTING EXCESSIVELY.
   ELEMENT 2032 IS DISTORTING EXCESSIVELY.
   ELEMENT 2032 IS DISTORTING EXCESSIVELY.
   ELEMENT 2032 IS DISTORTING EXCESSIVELY.
   ELEMENT 2032 IS DISTORTING EXCESSIVELY.
   ELEMENT 2033 IS DISTORTING EXCESSIVELY.
   ELEMENT 2033 IS DISTORTING EXCESSIVELY.
   ELEMENT 2033 IS DISTORTING EXCESSIVELY.
   ELEMENT 2033 IS DISTORTING EXCESSIVELY.
   ELEMENT 2033 IS DISTORTING EXCESSIVELY.
   ELEMENT 2033 IS DISTORTING EXCESSIVELY.
   ELEMENT 2033 IS DISTORTING EXCESSIVELY.
   ELEMENT 2033 IS DISTORTING EXCESSIVELY.
   ELEMENT 2034 IS DISTORTING EXCESSIVELY.
   ELEMENT 2034 IS DISTORTING EXCESSIVELY.
   ELEMENT 2034 IS DISTORTING EXCESSIVELY.
   ELEMENT 2034 IS DISTORTING EXCESSIVELY.
   ELEMENT 2034 IS DISTORTING EXCESSIVELY.
   ELEMENT 2034 IS DISTORTING EXCESSIVELY.
   ELEMENT 2034 IS DISTORTING EXCESSIVELY.
   ELEMENT 2034 IS DISTORTING EXCESSIVELY.
   ELEMENT 2045 IS DISTORTING EXCESSIVELY.
   ELEMENT 2045 IS DISTORTING EXCESSIVELY.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2713 IS OVERCLOSED BY 89.1215      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2714 IS OVERCLOSED BY 123.422       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2715 IS OVERCLOSED BY 123.099       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2716 IS OVERCLOSED BY 103.694       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2717 IS OVERCLOSED BY 76.9515      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2718 IS OVERCLOSED BY 35.3626      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2720 IS OVERCLOSED BY 1.46025     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2735 IS OVERCLOSED BY 419.916E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 2736 IS OVERCLOSED BY 428.248E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4069 IS OVERCLOSED BY 89.1215      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4070 IS OVERCLOSED BY 123.422       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4071 IS OVERCLOSED BY 123.099       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4072 IS OVERCLOSED BY 103.694       
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4073 IS OVERCLOSED BY 76.9515      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4074 IS OVERCLOSED BY 35.3626      
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4076 IS OVERCLOSED BY 1.46025     
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4091 IS OVERCLOSED BY 419.916E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.
   CONTACT PAIR (CRACKHIGH,CRACKLOW) NODE 4092 IS OVERCLOSED BY 428.248E-03   
   WHICH IS TOO SEVERE. SPECIFY A LARGE VALUE FOR THE HCRIT PARAMETER TO IGNORE 
   SEVERE OVERCLOSURES.

 ***WARNING: EXCESSIVE DISTORTION AT A TOTAL OF 140 INTEGRATION POINTS IN SOLID 
             (CONTINUUM) ELEMENTS
 

 ***NOTE: ELEMENTS ARE DISTORTING EXCESSIVELY. CONVERGENCE IS JUDGED UNLIKELY.
 

 ***NOTE: SEVERE CONTACT OVERCLOSURES EXIST. CONVERGENCE IS JUDGED UNLIKELY.
 

 ***ERROR: TOO MANY ATTEMPTS MADE FOR THIS INCREMENT



     ANALYSIS SUMMARY:
     TOTAL OF          1  INCREMENTS
                       5  CUTBACKS IN AUTOMATIC INCREMENTATION
                      69  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                      69  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                      69  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       7  WARNING MESSAGES DURING USER INPUT PROCESSING
                    1100  WARNING MESSAGES DURING ANALYSIS
                    1041  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                      32  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       1  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =      30.    
       SYSTEM TIME (SEC)    =      8.0    
       TOTAL CPU TIME (SEC) =      38.    
       WALLCLOCK TIME (SEC) =           53
       MEMORY PEAK (GB)     =            0

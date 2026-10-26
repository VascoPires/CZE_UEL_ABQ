      subroutine k_cohesive_law(T,T_d,GIc,GSH,tau3c,tau_sh,
     & eta,K3,KSH,delu,mcrd,nrhs,DMG_OLD,DMGv_OLD,
     & DMG,DMGv,delta,delta0,deltaF,B_VAR,DTIME,rho)
      INCLUDE 'ABA_PARAM.INC'
      dimension T(mcrd,nrhs),T_d(mcrd,mcrd),delu(mcrd)
      dimension pert(mcrd),Tp(mcrd,nrhs)
      DOUBLE PRECISION GIc,GSH,tau3c,tau_sh,eta,K3,KSH
      DOUBLE PRECISION DMG_OLD,DMGv_OLD,DMG,DMGv,delta,delta0,deltaF
      DOUBLE PRECISION B_VAR,DTIME,rho,dp,dvp,lp,l0p,lfp,bp,zeta
      PARAMETER (zeta=1.0d-6)

! Turon 2018 Eqs. 33-35. All perturbations use the same committed history.
! zeta has displacement units; the distributed examples use mm.
      call k_cohesive_response(T,GIc,GSH,tau3c,tau_sh,eta,K3,KSH,
     & delu,mcrd,nrhs,DMG_OLD,DMGv_OLD,DMG,DMGv,delta,delta0,
     & deltaF,B_VAR,DTIME,rho)
      do j = 1, mcrd
         do i = 1, mcrd
            pert(i) = delu(i)
         end do
         pert(j) = pert(j) + zeta
         call k_cohesive_response(Tp,GIc,GSH,tau3c,tau_sh,eta,K3,
     & KSH,pert,mcrd,nrhs,DMG_OLD,DMGv_OLD,dp,dvp,lp,l0p,lfp,bp,
     & DTIME,rho)
         do i = 1, mcrd
            T_d(i,j) = (Tp(i,1)-T(i,1))/zeta
         end do
      end do
      return
      end
      subroutine k_cohesive_response(T,GIc,GSH,tau3c,tau_sh,
     & eta,K3,KSH,delu,mcrd,nrhs,DMG_OLD,DMGv_OLD,
     & DMG,DMGv,delta,delta0,deltaF,B_VAR,DTIME,rho)
      INCLUDE 'ABA_PARAM.INC'

!----------------------------------------------------------------------
! UEL overview:
! - Builds local basis and separation at each integration point.
! - Evaluates Turon cohesive law (traction + tangent).
! - Assembles global stiffness and residual for an 8-node interface.
!----------------------------------------------------------------------    
      dimension T(mcrd,nrhs),delu(mcrd)
      DOUBLE PRECISION GIc, GSH, tau3c, tau_sh, eta, K3, KSH
      DOUBLE PRECISION DMG_OLD, DMGv_OLD, DMG, DMGv, delta, delta0,
     & deltaF
      DOUBLE PRECISION delta1, delta2, delta3, delta3_pos, delta_shear
      DOUBLE PRECISION delta0_3, delta0_shear, deltaC_3, deltaC_shear
      DOUBLE PRECISION KB, B_VAR, AUX1, AUX2, AUX3, AUX4
      DOUBLE PRECISION DTIME, rho, s
      DOUBLE PRECISION RT_OLD, RT

!----------------------------------------------------------------------
! Turon mixed-mode cohesive law (static, no fatigue)
! Inputs:
!   delu(1..3)  : local separation [shear1, shear2, normal]
!   GIc,GSH     : mode-I and effective shear fracture energies
!   tau3c,tau_sh : strengths (normal, shear)
!   eta         : BK exponent
!   K3,KSH      : penalty stiffness (normal, shear)
!   DMG_OLD     : history damage variable
! Outputs:
!   T(1..3)     : local traction vector
!   DMG         : updated damage
!   delta       : equivalent displacement
!   delta0      : onset displacement
!   deltaF      : final displacement
!   B_VAR       : mixed-mode ratio (shear contribution)
!----------------------------------------------------------------------

      call k_matrix_zero(T,mcrd,nrhs)

! 1) Split local separation into shear and normal components.
      delta1 = delu(1)
      delta2 = delu(2)
      delta3 = delu(3)
      delta3_pos = MAX(delta3, 0.d0)
      delta_shear = sqrt(delta1**2.0d0 + delta2**2.0d0)

! 2) Equivalent displacement (delta) and mixed-mode ratio (B_VAR).
      AUX1 = KSH*delta_shear**2 + K3*delta3_pos**2
      AUX2 = KSH**2*delta_shear**2 + K3**2*delta3_pos**2

      if (AUX1 .LE. 0.d0 .OR. AUX2 .LE. 0.d0) then
         B_VAR = 0.d0
         delta = 0.d0
      else
         B_VAR = (KSH*delta_shear**2)/AUX1
         delta = AUX1/SQRT(AUX2)
      end if

      KB = K3*(1.d0 - B_VAR) + KSH*B_VAR

! 3) Onset and final displacements (mixed-mode).
      if (KSH .GT. 0.d0 .AND. K3 .GT. 0.d0) then
         delta0_shear = tau_sh/KSH
         delta0_3 = tau3c/K3
      else
         delta0_shear = 0.d0
         delta0_3 = 0.d0
      end if

! Mixed mode onset displacement jump
      AUX3 = (KSH*delta0_shear**2 - K3*delta0_3**2) * (B_VAR**eta)
      if (KB .GT. 0.d0) then
         delta0 = SQRT((K3*delta0_3**2 + AUX3)/KB)
      else
         delta0 = 0.d0
      end if

      if (tau_sh .GT. 0.d0 .AND. tau3c .GT. 0.d0) then
         deltaC_shear = 2.d0*GSH/tau_sh
         deltaC_3 = 2.d0*GIc/tau3c
      else
         deltaC_shear = 0.d0
         deltaC_3 = 0.d0
      end if

      AUX4 = (KSH*delta0_shear*deltaC_shear - K3*delta0_3*deltaC_3)
      if (KB .GT. 0.d0 .AND. delta0 .GT. 0.d0) then
         deltaF = (K3*delta0_3*deltaC_3 + AUX4*(B_VAR**eta))
     &            /(KB*delta0)
      else
         deltaF = 0.d0
      end if

! 4) Damage update (history variable).
      if (deltaF .GT. delta0 .AND. delta0 .GT. 0.d0) then
         RT_OLD = (delta0*deltaF)/
     &            (deltaF - DMG_OLD*(deltaF - delta0))
      else
         RT_OLD = delta0
      end if

      RT = MAX(RT_OLD, delta)

      if (delta .GT. RT_OLD) then
         if (RT .GT. 0.d0 .AND. deltaF .GT. delta0) then
            DMG = (deltaF*(RT - delta0))/
     &            (RT*(deltaF - delta0))
            if (DMG .GT. 1.d0) DMG = 1.d0
            if (DMG .LT. 0.d0) DMG = 0.d0
         else
            DMG = DMG_OLD
         end if
      else
         DMG = DMG_OLD
      end if

! 5) Viscous regularization (Duvaut-Lions / Davila 2007).
!    Regularize damage with time scale rho (units of time).
      if (rho .GT. 0.d0 .AND. DTIME .GT. 0.d0) then
         s = DTIME/(rho + DTIME)
      else
         s = 1.d0
      end if
      DMGv = s*DMG + (1.d0 - s)*DMGv_OLD
      if (DMGv .GT. 1.d0) DMGv = 1.d0
      if (DMGv .LT. 0.d0) DMGv = 0.d0

! 6) Tractions (with compression stiffness in the normal direction).
      T(1,1) = KSH*(1.d0 - DMGv)*delta1
      T(2,1) = KSH*(1.d0 - DMGv)*delta2
      if (delta3 .GE. 0.d0) then
         T(3,1) = K3*(1.d0 - DMGv)*delta3
      else
! Compression case
         T(3,1) = K3*delta3
      end if

      return
      end
      subroutine k_cohesive_energy(GIc,GSH,eta,K3,KSH,delu,
     & Dold,Dnew,Dvold,Dvnew,delta0,deltaF,Bmix,rho,Yold,psi,ddiss,Y)
      INCLUDE 'ABA_PARAM.INC'
      DOUBLE PRECISION GIc,GSH,eta,K3,KSH,delu(3),Dold,Dnew
      DOUBLE PRECISION Dvold,Dvnew,delta0,deltaF,Bmix,rho,psi,ddiss,Y
      DOUBLE PRECISION r_old,r_new,ratio,Gc,closed,Yold

      Y = 0.5d0*(KSH*(delu(1)**2+delu(2)**2)
     &           + K3*MAX(delu(3),0.d0)**2)
      closed = 0.5d0*K3*MIN(delu(3),0.d0)**2
      psi = (1.d0-Dvnew)*Y + closed
      ddiss = 0.d0
      if (rho .LE. 0.d0) then
         if (deltaF .GT. delta0 .AND. delta0 .GT. 0.d0) then
            ratio = deltaF/delta0
            r_old = Dold/(ratio*(1.d0-Dold)+Dold)
            r_new = Dnew/(ratio*(1.d0-Dnew)+Dnew)
            Gc = GIc+(GSH-GIc)*Bmix**eta
! Exact for proportional loading. Changing mode uses endpoint Gc;
! its accumulated value must be checked by increment refinement.
            ddiss = Gc*MAX(r_new-r_old,0.d0)
         end if
      else
! Integrate the damage force against the actual delayed damage change.
! Trapezoidal force quadrature improves the energy balance on refinement.
! No independent dashpot exists, so this stays in damage dissipation.
         ddiss = 0.5d0*(Yold+Y)*MAX(Dvnew-Dvold,0.d0)
      end if
      return
      end
      subroutine k_matrix_zero(A,n,m)
      INCLUDE 'ABA_PARAM.INC'
      dimension A(n,m)

      do i = 1, n
         do j = 1, m
            A(i,j)=0.d0
         end do
      end do

      return
      end

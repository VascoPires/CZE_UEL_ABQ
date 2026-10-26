!*************************************************************************
! TECHNICAL UNIVERSITY OF LEOBEN
! Chair of Designing Plastics and Composite Materials
!*************************************************************************
! 3D CZM UEL
! VERSION: 1.3
!-------------------------------------------------------------------------
! AUTHOR: Vasco Pires
!
! DATE: 16.10.2026
!
!=========================================================================
! HEADER OF THE SUBROUTINE
! +-----------+-------------------------------------------------------------------+
! UEL for square 3D cohesive zone elements, Fixed-form F77 code.
! F77 IMPLICIT NAME CONVENTION, SINGLE/DOUBLE CONTROLLED BY ABAQUS
! +-----------+-------------------------------------------------------------------+
      SUBROUTINE UEL (RHS, AMATRX, SVARS, ENERGY, NDOFEL, NRHS, NSVARS,
     1 PROPS, NPROPS, COORDS, MCRD, NNODE, U, DU, V, A, JTYPE, TIME,
     2 DTIME, KSTEP, KINC, JELEM, PARAMS, NDLOAD, JDLTYP, ADLMAG,
     3 PREDEF, NPREDF, LFLAGS, MLVARX, DDLMAG, MDLOAD, PNEWDT, JPROPS,
     4 NJPROP, PERIOD)

      INCLUDE 'ABA_PARAM.INC'
      INCLUDE 'SMAAspUserSubroutines.hdr'
    
      DIMENSION RHS(MLVARX,*),AMATRX(NDOFEL,NDOFEL),PROPS(*),
     1 SVARS(*),ENERGY(8),COORDS(MCRD,NNODE),U(NDOFEL),
     2 DU(MLVARX,*),V(NDOFEL),A(NDOFEL),TIME(2),PARAMS(*),
     3 JDLTYP(MDLOAD,*),ADLMAG(MDLOAD,*),DDLMAG(MDLOAD,*),
     4 PREDEF(2,NPREDF,NNODE),LFLAGS(*),JPROPS(*)

      DIMENSION ds1(4),ds2(4),dn(4),Trac(MCRD,NRHS),
     1 Trac_Jacob(MCRD,MCRD),R(MCRD,MCRD),coord_l(MCRD,NNODE),
     2 GP_coord(2),sf(4),B(MCRD,NDOFEL),co_de_m(3,4),
     3 B_t(NDOFEL,MCRD), Transformation_M(NDOFEL,NDOFEL),
     4 Transformation_M_T(NDOFEL,NDOFEL),temp1(MCRD,NDOFEL)

      DIMENSION stiff_l(NDOFEL,NDOFEL),temp2(NDOFEL,NDOFEL),
     1 stiff_g(NDOFEL,NDOFEL),residual_l(NDOFEL,NRHS),
     2 residual_g(NDOFEL,NRHS),aJacob_M(2,3),delu_loc_gp(mcrd),
     3 co_de(mcrd,nnode)

      CHARACTER*200 filename1, filename3, file1, file3, nostr, dirname
      CHARACTER*200 file_ip, file_un, file_ipidx, file_unidx
      CHARACTER*3 rankstr
      LOGICAL exist, opened28, opened30
      INTEGER unit_ipvals, unit_unvals, unit_ipidx, unit_unidx
      INTEGER mutex_id, rank, io_initialized
      INTEGER echo, ip_nvals, kinc_out, last_kinc
      INTEGER*8 pos_ip, pos_un
      COMMON /UELIO/ unit_ipvals, unit_unvals, unit_ipidx, unit_unidx,
     &               mutex_id, rank, io_initialized, last_kinc
      SAVE /UELIO/


      DOUBLE PRECISION GIc, GIIc, GSH, tau3c, tau_sh,
     1 eta, K3, KSH, KB, delta3, delta_shear, delta1, delta2, delta,
     2 delta0_3, delta0_shear, delta0, deltaC_3, deltaC_shear, deltaF,
     3 DMG_OLD, DMG, DMGv_OLD, DMGv, RT_OLD, RT, B_VAR, rho,
     4 x_gp, y_gp, z_gp, tr_tan, tr_norm, u1, u2, u3,
     5 xi_gp, eta_gp, wgt, jac_w
     
      DOUBLE PRECISION psi_ip, diss_inc, damage_force

!                          UEL Properties Table
! +------------+-------------------------------------------------+-----------+
! | Parameter  | Description                                     | PROPS(i)  |
! +------------+-------------------------------------------------+-----------+
! | GIc        | Critical energy release rate (mode I)         | 1         |
! | GIIc       | Shear fracture energy (mode II=III)           | 2         |
! | tau3c      | Normal strength (local direction 3)           | 3         |
! | tau_sh     | Shear strength (local shear magnitude)        | 4         |
! | eta        | BK exponent                                   | 5         |
! | K3         | Normal penalty stiffness                      | 6         |
! | rho        | Viscous regularization (time scale)           | 7         |
! +------------+-------------------------------------------------+-----------+

! +++++++++ Notation +++++++++
! ============================
! This UEL follows Turon's notation with the simplification:
! - GIIc is used for both shear modes (GIIc = GIIIc).
! - Local directions 1/2 are shear, local direction 3 is normal opening.
! - K3 is the normal penalty stiffness, KSH is the shear penalty stiffness.
!
! The local basis is built per integration point from the mid-surface.
! The separation vector (delu_loc_gp) is expressed in that local basis.
!
! SDV layout (per Integration point, base = 20*(i-1)):
!   SDV1..3  : Integration point coordinates (x,y,z)
!   SDV4     : delta_shear (sqrt(delta1^2+delta2^2))
!   SDV5     : delta3 (normal opening)
!   SDV6     : |T_shear|
!   SDV7     : T_normal
!   SDV8     : damage DMG (rate-independent)
!   SDV9     : equivalent displacement delta
!   SDV10    : mixed-mode ratio B
!   SDV11    : viscously regularized damage DMGv
!   SDV12    : recoverable energy per unit area
!   SDV13    : accumulated damage dissipation per unit area
!   SDV14    : accumulated integration-point dissipation (area weighted)
!   SDV15    : thermodynamic damage force Y
!   SDV16..20: reserved
! Fortran ENERGY(2): recoverable; ENERGY(4): damage dissipation.



! Output is relative to the solver working directory.
! Run each job in its own folder (see scripts/run_case.py).
      dirname = ''

      unit_ipvals = 28
      unit_unvals = 30

      GIc=PROPS(1)
      GIIc=PROPS(2)
      tau3c=PROPS(3)
      tau_sh=PROPS(4)
      eta=PROPS(5)
      K3=PROPS(6)
      GSH=GIIc
      rho=0.d0
      if (NPROPS .GE. 7) rho=PROPS(7)

      GP_n=4d0
      INT_SCHEME=1
      if (NJPROP .GE. 1) INT_SCHEME=JPROPS(1)

      echo=0
      ip_nvals=20
      unit_ipvals = 28
      unit_unvals = 30
      if (echo .EQ. 1) then
         kinc_out = kinc
         if (io_initialized .EQ. 0) then
            call MutexLock(mutex_id)
            if (io_initialized .EQ. 0) then
               call GETRANK(rank)
               write(rankstr,'(I3.3)') rank
               file_ip = trim(dirname)//'IPVALS_CZM_rank'//rankstr//'.txt'
               file_un = trim(dirname)//'UNVALS_CZM_rank'//rankstr//'.txt'
               file_ipidx = trim(dirname)//'IPVALS_CZM_rank'//rankstr//'.idx'
               file_unidx = trim(dirname)//'UNVALS_CZM_rank'//rankstr//'.idx'

               inquire(file=file_ip, exist=exist)
               if (exist) then
                  open(unit_ipvals, file=file_ip, status='old',
     &                 form='formatted', access='stream')
                  close(unit_ipvals, status='delete')
               end if
               open(unit_ipvals, file=file_ip, status='new',
     &              form='formatted', access='stream', action='write')

               inquire(file=file_un, exist=exist)
               if (exist) then
                  open(unit_unvals, file=file_un, status='old',
     &                 form='formatted', access='stream')
                  close(unit_unvals, status='delete')
               end if
               open(unit_unvals, file=file_un, status='new',
     &              form='formatted', access='stream', action='write')

               inquire(file=file_ipidx, exist=exist)
               if (exist) then
                  open(unit_ipidx, file=file_ipidx, status='old',
     &                 form='formatted', access='stream')
                  close(unit_ipidx, status='delete')
               end if
               open(unit_ipidx, file=file_ipidx, status='new',
     &              form='formatted', access='stream', action='write')

               inquire(file=file_unidx, exist=exist)
               if (exist) then
                  open(unit_unidx, file=file_unidx, status='old',
     &                 form='formatted', access='stream')
                  close(unit_unidx, status='delete')
               end if
               open(unit_unidx, file=file_unidx, status='new',
     &              form='formatted', access='stream', action='write')

               last_kinc = -9999
               io_initialized = 1
            end if
            call MutexUnlock(mutex_id)
         end if
      end if


! Initialize Matrices and Vectors======================================
 
      call k_vector_zero(ds1,4)
      call k_vector_zero(ds2,4)
      call k_vector_zero(dn,4)
      call k_matrix_zero(Trac,mcrd,nrhs)
      call k_matrix_zero(Trac_Jacob,mcrd,mcrd)
      call k_matrix_zero(R,mcrd,mcrd)
      call k_matrix_zero(coord_l,mcrd,nnode)
      call k_vector_zero(GP_coord,2)
      call k_vector_zero(sf,4)
      call k_matrix_zero(Transformation_M,ndofel,ndofel)
      call k_matrix_zero(Transformation_M_T,ndofel,ndofel)
      call k_matrix_zero(B,mcrd,ndofel)
      call k_matrix_zero(B_t,ndofel,mcrd)
      call k_matrix_zero(temp1,mcrd,ndofel)
      call k_matrix_zero(stiff_l,ndofel,ndofel)
      call k_matrix_zero(temp2,ndofel,ndofel)
      call k_matrix_zero(stiff_g,ndofel,ndofel)
      call k_matrix_zero(residual_l,ndofel,nrhs)
      call k_matrix_zero(residual_g,ndofel,nrhs)
      call k_matrix_zero(aJacob_M,2,3)
      call k_matrix_zero(rhs,ndofel,nrhs)
      call k_matrix_zero(amatrx,ndofel,ndofel)
      call k_matrix_zero(co_de,mcrd,nnode)
      a_Jacob=0.d0

! This interface carries no mass, creep or separate viscous dashpot.
! Damage dissipation (including delayed damage when rho>0) uses ENERGY(4).
      do i = 1, 8
         ENERGY(i) = 0.d0
      end do

! Constant parameters for each integration point:
! - Effective shear fracture energy uses GIIc (assumed = GIIIc).
! - KSH follows Turon's mixed-mode penalty stiffness definition.
      if (GSH .GT. 0.d0 .AND. tau3c .GT. 0.d0) then
         KSH = (GIc/GSH)*(tau_sh/tau3c)**2 * K3
      else
         KSH = K3
      end if
      SVARS(108)=KSH

! Do local computations================================================
 
!     Build deformed nodal coordinates (global): coords + displacement.
      do i = 1, mcrd
        do j = 1, nnode
           co_de(i,j)=coords(i,j)+U(3.0*(j-1.0)+i)
        end do
      end do

! Output is written after Gauss-point loop to keep blocks contiguous.

! Do Calculations at Gauss Points======================================

      do i = 1, GP_n

            call k_matrix_zero(aJacob_M,2,3)

            gpt = i

! Determine the values of the shape function at each integration point
            call k_int_point(i,INT_SCHEME,xi_gp,eta_gp,wgt)
            call k_shape_fun(xi_gp,eta_gp,sf)

! Compute local basis R, local coordinates, and transformation matrices.
            call k_local_coordinates(xi_gp,eta_gp,co_de,R,coord_l,
     & Transformation_M,
     & Transformation_M_T,a_Jacob,aJacob_M,coords,u,ndofel,nnode,
     & mcrd, SVARS)

! Compute shear and normal local opening displacments==================
 
            do j = 1, 4
                  ds1(j)=coord_l(1,j+4)-coord_l(1,j)
                  ds2(j)=coord_l(2,j+4)-coord_l(2,j)
                  dn(j) =coord_l(3,j+4)-coord_l(3,j)
            end do

! Deformed global integration point coordinates (mid-surface)

            base = 20*(i-1)
            do k1 = 16, 20
                  SVARS(base+k1) = 0.d0
            end do
            do j = 1, 4
                  co_de_m(1,j) = 0.5d0*(co_de(1,j) + co_de(1,j+4))
                  co_de_m(2,j) = 0.5d0*(co_de(2,j) + co_de(2,j+4))
                  co_de_m(3,j) = 0.5d0*(co_de(3,j) + co_de(3,j+4))
            end do
            x_gp = 0.d0
            y_gp = 0.d0
            z_gp = 0.d0
            do j = 1, 4
                  x_gp = x_gp + sf(j)*co_de_m(1,j)
                  y_gp = y_gp + sf(j)*co_de_m(2,j)
                  z_gp = z_gp + sf(j)*co_de_m(3,j)
            end do
            Svars(base+1) = x_gp
            Svars(base+2) = y_gp
            Svars(base+3) = z_gp

            call k_vector_zero(delu_loc_gp,mcrd)
!
! Determine shear and normal opening displamenets at integration points
! Loops over the 4 Int Points for the three DOF of each point.
! Here sf are the shape-function weight
!
            do j = 1, 4
                  delu_loc_gp(1)=delu_loc_gp(1)+ds1(j)*sf(j)
                  delu_loc_gp(2)=delu_loc_gp(2)+ds2(j)*sf(j)
                  delu_loc_gp(3)=delu_loc_gp(3)+ dn(j)*sf(j)
            end do

! These values correspond to the local displacements at each Integration Point
! delta3 = normal opening displacement -> DOF3 (local)
! delta_shear = tangential/shear opening displacement -> DOF1 and DOF2 (local)
            delta1 = delu_loc_gp(1)
            delta2 = delu_loc_gp(2)
            delta3 = delu_loc_gp(3)
            delta_shear = sqrt(delta1**2.0d0+delta2**2.0d0)

! Store instantaneous openings for post-processing (per integration point)
            Svars(base+4)=delta_shear
            Svars(base+5)=delta3

!
! Determine Traction vector and tangent modulus matrix

! Traction ordering (local):
!   T(1)=shear1, T(2)=shear2, T(3)=normal.
! UMAT STRESS(1..3) ↔ UEL T(3,1), same local ordering as above.
! UMAT DDSDDE(1..3,1..3) ↔ UEL T_d(1..3,1..3) (local tangents).

! The outputs of the cohesive law are the tractions and the tangent stiffness matrix.
! DMG is stored at SVARS base+8, DMGv at SVARS base+11.

            DMG_OLD = Svars(base+8)
            DMGv_OLD = Svars(base+11)
            call k_cohesive_law(Trac,Trac_Jacob,GIc,GSH,tau3c,tau_sh,
     & eta,K3,KSH,delu_loc_gp,mcrd,nrhs,DMG_OLD,DMGv_OLD,
     & DMG,DMGv,delta,delta0,deltaF,B_VAR,DTIME,rho)
            Svars(base+8)=DMG
            Svars(base+9)=delta
            Svars(base+10)=B_VAR
            Svars(base+11)=DMGv

!
! Current tractions (global) at the Integration point
!
            tr_tan = sqrt(Trac(1,1)**2.0 + Trac(2,1)**2.0)
            tr_norm = Trac(3,1)
            Svars(base+6) = tr_tan
            Svars(base+7) = tr_norm
!
! Determine B matrix and its transpose
! 
            call k_Bmatrix(sf,B,mcrd,ndofel)

            call k_matrix_transpose(B,B_t,mcrd,ndofel)
!
! Compute the stiffness matrix
! Local Stiffness = B_t * Trac_Jacob * B
!
            call k_matrix_multiply(Trac_Jacob,B,temp1,mcrd,mcrd,
     & ndofel)
            call k_matrix_multiply(B_t,temp1,stiff_l,ndofel,
     & mcrd,ndofel)
!
! Compute Global stiffness matrix 
! Global_K = T' * K * T
!
            call k_matrix_multiply(Transformation_M_T,stiff_l,
     & temp2,ndofel,ndofel,ndofel)
            call k_matrix_multiply(temp2,Transformation_M,stiff_g,
     & ndofel,ndofel,ndofel)

! Multiply Jacobian with the Global stiffness and add contribution
! from each Gauss Point
!
            jac_w = a_Jacob*wgt

! ENERGY is recomputed from the start-of-increment SVARS on every call.
! Accumulate dissipated energy with the area at which it was produced;
! do not rescale past dissipation when the current surface area changes.
            call k_cohesive_energy(GIc,GSH,eta,K3,KSH,delu_loc_gp,
     &           DMG_OLD,DMG,DMGv_OLD,DMGv,delta0,deltaF,B_VAR,rho,
     &           Svars(base+15),psi_ip,diss_inc,damage_force)
            Svars(base+12) = psi_ip
            Svars(base+13) = Svars(base+13) + diss_inc
            Svars(base+14) = Svars(base+14) + jac_w*diss_inc
            Svars(base+15) = damage_force
            ENERGY(2) = ENERGY(2) + jac_w*psi_ip
            ENERGY(4) = ENERGY(4) + Svars(base+14)
            call k_matrix_plus_scalar(amatrx,stiff_g,jac_w,
     & ndofel,ndofel)
!
! Compute the global residual vector
! Local_residual = B_t * Trac
! Global_residual = T' * Local_residual
!
            call k_matrix_multiply(B_t,Trac,residual_l,ndofel,
     & mcrd,nrhs)
            call k_matrix_multiply(Transformation_M_T,residual_l,
     & residual_g,ndofel,ndofel,nrhs)
!
! Multiply the Global residual by the Jacobian and add the 
! contribution from each point
!   
            call k_matrix_plus_scalar(rhs,residual_g,jac_w,
     & ndofel,nrhs)
!
! Integration-point values are written after the Gauss loop.
      end do
!
! Write output (thread-safe, per-rank single files).
!
      if (echo .EQ. 1 .AND. io_initialized .EQ. 1) then
         call MutexLock(mutex_id)
         if (kinc_out .NE. last_kinc) then
            inquire(unit=unit_ipvals, pos=pos_ip)
            inquire(unit=unit_unvals, pos=pos_un)
            write(unit_ipidx,'(I10,1X,I10,1X,1PE20.12,1X,1PE20.12,
     &           1X,I20)') kinc_out, kstep, time(1), time(2), pos_ip
            write(unit_unidx,'(I10,1X,I10,1X,1PE20.12,1X,1PE20.12,
     &           1X,I20)') kinc_out, kstep, time(1), time(2), pos_un
            write(unit_ipvals,'(A,I8)') 'KINC=', kinc_out
            write(unit_unvals,'(A,I8)') 'KINC=', kinc_out
            last_kinc = kinc_out
         end if

         write(unit_unvals,*) jelem, 8, 3
         do j = 1, 8
            u1 = U(3*(j-1)+1)
            u2 = U(3*(j-1)+2)
            u3 = U(3*(j-1)+3)
            write(unit_unvals,'(12f20.10)') u1, u2, u3,
     &           0.d0, 0.d0, 0.d0, 0.d0, 0.d0, 0.d0,
     &           0.d0, 0.d0, 0.d0
         end do

         write(unit_ipvals,*) jelem, 4, ip_nvals, INT_SCHEME
         do i = 1, GP_n
            base = 20*(i-1)
            write(unit_ipvals,'(20f20.10)') (SVARS(base+k1), k1=1,20)
         end do

         call MutexUnlock(mutex_id)
      end if
!
      return
      end

c======================================================================
c External database subroutine for thread-safe I/O initialization
c======================================================================
      SUBROUTINE UEXTERNALDB(LOP,LRESTART,TIME,DTIME,KSTEP,KINC)
      INCLUDE 'ABA_PARAM.INC'
      INCLUDE 'SMAAspUserSubroutines.hdr'

      DOUBLE PRECISION TIME(2), DTIME
      INTEGER LOP, LRESTART, KSTEP, KINC
      INTEGER unit_ipvals, unit_unvals, unit_ipidx, unit_unidx
      INTEGER mutex_id, rank, io_initialized, last_kinc
      CHARACTER*200 file_ip, file_un, file_ipidx, file_unidx
      CHARACTER*3 rankstr
      LOGICAL exist, opened

      COMMON /UELIO/ unit_ipvals, unit_unvals, unit_ipidx, unit_unidx,
     &               mutex_id, rank, io_initialized, last_kinc
      SAVE /UELIO/

      if (LOP .EQ. 0) then
         unit_ipvals = 28
         unit_unvals = 30
         unit_ipidx = 31
         unit_unidx = 32
         mutex_id = 1
         io_initialized = 0
         last_kinc = -9999

         call MutexInit(mutex_id)
         call GETRANK(rank)
      end if

      if (LOP .EQ. 3) then
         inquire(unit=unit_ipvals, opened=opened)
         if (opened) close(unit_ipvals)
         inquire(unit=unit_unvals, opened=opened)
         if (opened) close(unit_unvals)
         inquire(unit=unit_ipidx, opened=opened)
         if (opened) close(unit_ipidx)
         inquire(unit=unit_unidx, opened=opened)
         if (opened) close(unit_unidx)
         io_initialized = 0
      end if

      return
      end
!======================================================================
!=============================SUBROUTINES==============================
!======================================================================
!
! Determine the strain-displacement (B) matrix
! - Maps nodal separations to local displacement jump at Gauss points.
! - DOF ordering: [u1,u2,u3] for each of 8 nodes.
!
      subroutine k_Bmatrix(sf,B,mcrd,ndofel)
      INCLUDE 'ABA_PARAM.INC'
      dimension sf(4),B(mcrd,ndofel)
      B(1,1) =  sf(1)
      B(1,4) =  sf(2)
      B(1,7) =  sf(3)
      B(1,10)=  sf(4)
      B(1,13)= -sf(1)
      B(1,16)= -sf(2)
      B(1,19)= -sf(3)
      B(1,22)= -sf(4)
      B(2,2) =  sf(1)
      B(2,5) =  sf(2)
      B(2,8) =  sf(3)
      B(2,11)=  sf(4)
      B(2,14)= -sf(1)
      B(2,17)= -sf(2)
      B(2,20)= -sf(3)
      B(2,23)= -sf(4)
      B(3,3) =  sf(1)
      B(3,6) =  sf(2)
      B(3,9) =  sf(3)
      B(3,12)=  sf(4)
      B(3,15)= -sf(1)
      B(3,18)= -sf(2)
      B(3,21)= -sf(3)
      B(3,24)= -sf(4)
!
      return
      end
!======================================================================
! Turon cohesive law:
! - Mixed-mode (mode-dependent penalty stiffness)
! - Linear elastic to damage, with traction softening
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
!======================================================================
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
!======================================================================
! Recoverable and irreversible cohesive energies per unit interface area.
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
!======================================================================
! Build local basis and transformations:
! - Uses mid-surface to define local axes.
! - Returns R, Transformation_M, coord_l, and Jacobian.
      subroutine k_local_coordinates(xi,eta,co_de,R,coord_l,
     & Transformation_M,Transformation_M_T,a_Jacob,aJacob_M,coords,u,
     & ndofel,nnode,mcrd, SVARS)
      INCLUDE 'ABA_PARAM.INC'
      dimension R(mcrd,mcrd),coord_l(mcrd,nnode),aJacob_M(2,3),
     & Transformation_M(ndofel,ndofel),coords(mcrd,nnode),
     & Transformation_M_T(ndofel,ndofel),u(ndofel),
     & co_de(mcrd,nnode), co_de_m(3,4),SFD(2,4),SVARS(*)
       DOUBLE PRECISION x1, x2, x3, x4, y1, y2, y3, y4, y5, y6, z1, z2,
     & z3, z4, xi, eta

      call k_matrix_zero(co_de_m,3,4)

      do i = 1, 3
         co_de_m(i,1)=(co_de(i,1)+co_de(i,5))*0.5
         co_de_m(i,2)=(co_de(i,2)+co_de(i,6))*0.5
         co_de_m(i,3)=(co_de(i,3)+co_de(i,7))*0.5
         co_de_m(i,4)=(co_de(i,4)+co_de(i,8))*0.5
      end do

      x1=co_de_m(1,1)
      x2=co_de_m(1,2)
      x3=co_de_m(1,3)
      x4=co_de_m(1,4)

      y1=co_de_m(2,1)
      y2=co_de_m(2,2)
      y3=co_de_m(2,3)
      y4=co_de_m(2,4)

      z1=co_de_m(3,1)
      z2=co_de_m(3,2)
      z3=co_de_m(3,3)
      z4=co_de_m(3,4)

      c_r=xi
      c_s=eta

      SFD(1,1) =-0.25*(1-c_s)
      SFD(1,2) = 0.25*(1-c_s)
      SFD(1,3) = 0.25*(1+c_s)
      SFD(1,4) =-0.25*(1+c_s)
      SFD(2,1) =-0.25*(1-c_r)
      SFD(2,2) =-0.25*(1+c_r)
      SFD(2,3) = 0.25*(1+c_r)
      SFD(2,4) = 0.25*(1-c_r)

      do i = 1,2
         do j = 1,3
            do k =1, 4
               aJacob_M(i,j) = aJacob_M(i,j) + SFD(i,k)*co_de_m(j,k)
            end do
         end do
      end do

      dum1 = aJacob_M(1,2)*aJacob_M(2,3) - aJacob_M(1,3)*aJacob_M(2,2)
      dum2 = aJacob_M(1,3)*aJacob_M(2,1) - aJacob_M(1,1)*aJacob_M(2,3)
      dum3 = aJacob_M(1,1)*aJacob_M(2,2) - aJacob_M(1,2)*aJacob_M(2,1)

      a_Jacob = sqrt(dum1**2 + dum2**2 + dum3**2)

      R(3,1) = dum1/a_Jacob
      R(3,2) = dum2/a_Jacob
      R(3,3) = dum3/a_Jacob

      aLen=sqrt(aJacob_M(1,1)**2.0d00+aJacob_M(1,2)**2.0d00+aJacob_M(1,3)**2.0d00)
      R(1,1)=aJacob_M(1,1)/aLen
      R(1,2)=aJacob_M(1,2)/aLen
      R(1,3)=aJacob_M(1,3)/aLen

      aLen2a=R(3,2)*R(1,3)-R(3,3)*R(1,2)
      aLen2b=R(3,3)*R(1,1)-R(3,1)*R(1,3)
      aLen2c=R(3,1)*R(1,2)-R(3,2)*R(1,1)

      aLen2 = sqrt(aLen2a**2.0d00 + aLen2b**2.0d00 + aLen2c**2.0d00)

      R(2,1) = aLen2a/aLen2
      R(2,2) = aLen2b/aLen2
      R(2,3) = aLen2c/aLen2

      a_J11 = (-0.25*(1-c_s))*x1 + (0.25*(1-c_s))*x2 + (0.25*(1+c_s))*x3 +
     & (-0.25*(1+c_s))*x4
      a_J12 = (-0.25*(1-c_s))*z1 + (0.25*(1-c_s))*z2 + (0.25*(1+c_s))*z3 +
     & (-0.25*(1+c_s))*z4
      a_J21 = (-0.25*(1-c_r))*x1 + (-0.25*(1+c_r))*x2 + (0.25*(1+c_r))*x3 +
     & (0.25*(1-c_r))*x4
      a_J22 = (-0.25*(1-c_r))*z1 + (-0.25*(1+c_r))*z2 + (0.25*(1+c_r))*z3 +
     & (0.25*(1-c_r))*z4

      b_J11 = (-0.25*(1-c_s))*x1 + (0.25*(1-c_s))*x2 + (0.25*(1+c_s))*x3 +
     & (-0.25*(1+c_s))*x4
      b_J12 = (-0.25*(1-c_s))*y1 + (0.25*(1-c_s))*y2 + (0.25*(1+c_s))*y3 +
     & (-0.25*(1+c_s))*y4
      b_J21 = (-0.25*(1-c_r))*x1 + (-0.25*(1+c_r))*x2 + (0.25*(1+c_r))*x3 +
     & (0.25*(1-c_r))*x4
      b_J22 = (-0.25*(1-c_r))*y1 + (-0.25*(1+c_r))*y2 + (0.25*(1+c_r))*y3 +
     & (0.25*(1-c_r))*y4

      c_J11 = (-0.25*(1-c_s))*y1 + (0.25*(1-c_s))*y2 + (0.25*(1+c_s))*y3 +
     & (-0.25*(1+c_s))*y4
      c_J12 = (-0.25*(1-c_s))*z1 + (0.25*(1-c_s))*z2 + (0.25*(1+c_s))*z3 +
     & (-0.25*(1+c_s))*z4
      c_J21 = (-0.25*(1-c_r))*y1 + (-0.25*(1+c_r))*y2 + (0.25*(1+c_r))*y3 +
     & (0.25*(1-c_r))*y4
      c_J22 = (-0.25*(1-c_r))*z1 + (-0.25*(1+c_r))*z2 + (0.25*(1+c_r))*z3 +
     & (0.25*(1-c_r))*z4


      a_Jacob1 = (a_J11*a_J22 - a_J12*a_J21)
      a_Jacob2 = (b_J11*b_J22 - b_J12*b_J21)
      a_Jacob3 = (c_J11*c_J22 - c_J12*c_J21)

      a_Jacob = sqrt(a_Jacob1**2.0d00 + a_Jacob2**2.0d00 + a_Jacob3**2.0d00)

!======================================================================
      num=nnode

      do i = 1, num
         dum=3.0*(i-1.0)
         Transformation_M(dum+1,dum+1)=R(1,1)
         Transformation_M(dum+1,dum+2)=R(1,2) 
         Transformation_M(dum+1,dum+3)=R(1,3)
         Transformation_M(dum+2,dum+1)=R(2,1)
         Transformation_M(dum+2,dum+2)=R(2,2)
         Transformation_M(dum+2,dum+3)=R(2,3)
         Transformation_M(dum+3,dum+1)=R(3,1)
         Transformation_M(dum+3,dum+2)=R(3,2)
         Transformation_M(dum+3,dum+3)=R(3,3)
      end do

      call k_matrix_transpose(Transformation_M,Transformation_M_T,
     $ ndofel,ndofel)


      do i = 1, nnode
         coord_l(1,i)=(R(1,1)*co_de(1,i)+R(1,2)*co_de(2,i)
     & +R(1,3)*co_de(3,i))
         coord_l(2,i)=(R(2,1)*co_de(1,i)+R(2,2)*co_de(2,i)
     & +R(2,3)*co_de(3,i))
         coord_l(3,i)=(R(3,1)*co_de(1,i)+R(3,2)*co_de(2,i)
     & +R(3,3)*co_de(3,i))
      end do
   
      return
      end
!======================================================================
! 2x2 integration shape functions for 4-node surface.
      subroutine k_int_point(i,scheme,xi,eta,wgt)
      INCLUDE 'ABA_PARAM.INC'
      DOUBLE PRECISION xi, eta, wgt, a
      INTEGER i, scheme

      a = sqrt(1.0d0/3.0d0)
      if (scheme .eq. 2) a = 1.0d0

      if (i .eq. 1) then
         xi  = -a
         eta = -a
      elseif (i .eq. 2) then
         xi  =  a
         eta = -a
      elseif (i .eq. 3) then
         xi  =  a
         eta =  a
      elseif (i .eq. 4) then
         xi  = -a
         eta =  a
      end if
      wgt = 1.0d0
      return
      end
!======================================================================
      subroutine k_shape_fun(xi,eta,sf)
      INCLUDE 'ABA_PARAM.INC'
      DOUBLE PRECISION xi, eta
      dimension sf(4)

      sf(1)=(1-xi)*(1-eta)*0.25
      sf(2)=(1+xi)*(1-eta)*0.25
      sf(3)=(1+xi)*(1+eta)*0.25
      sf(4)=(1-xi)*(1+eta)*0.25

      return
      end
!======================================================================
! Dense matrix multiplication: C = A * B.
      subroutine k_matrix_multiply(A,B,C,l,n,m)
      INCLUDE 'ABA_PARAM.INC'
      dimension A(l,n),B(n,m),C(l,m)

      call k_matrix_zero(C,l,m)

      do i = 1, l
         do j = 1, m
            do k = 1, n
               C(i,j)=C(i,j)+A(i,k)*B(k,j)
            end do
         end do
      end do

      return
      end
!=====================================================================
! Accumulate with scalar: A = A + c*B.
      subroutine k_matrix_plus_scalar(A,B,c,n,m)
      INCLUDE 'ABA_PARAM.INC'
      dimension A(n,m),B(n,m)

      do i = 1, n
         do j = 1, m
            A(i,j)=A(i,j)+c*B(i,j)
         end do
      end do
!
      return
      end
!======================================================================
! Matrix transpose: B = A^T.
      subroutine k_matrix_transpose(A,B,n,m)
      INCLUDE 'ABA_PARAM.INC'
      dimension A(n,m),B(m,n)

      do i = 1, n
         do j = 1, m
            B(j,i)=A(i,j)
         end do
      end do

      return
      end
!======================================================================
! Zero a matrix.
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
!======================================================================
! Zero a vector.
      subroutine k_vector_zero(A,n)
      INCLUDE 'ABA_PARAM.INC'
      dimension A(n)

      do i = 1, n
         A(i)=0.d0
      end do

      return
      end
!======================================================================
! Macaulay bracket: pM = max(a-b, 0).
      subroutine k_Mac(pM,a,b)
      INCLUDE 'ABA_PARAM.INC'

      if ((a-b) .GE. 0.0) then
         pM=a-b
      elseif ((a-b) .LT. 0.0) then
         pM=0.d0
      end if

      return
      end
      
c======================================================================
c=================================END==================================
c======================================================================









      

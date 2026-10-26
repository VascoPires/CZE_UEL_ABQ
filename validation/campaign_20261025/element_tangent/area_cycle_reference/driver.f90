program element_check
implicit none
integer it,j,k,n,scheme,ip,base,nrhs,ndof,ns,mcrd,nn,np,nj,zero,one
integer flags(10),jprops(1),jdl(1,1)
real*8 rhs(24,1),Kmat(24,24),sv(200),energy(8),props(7),coords(3,8),u(24),du(24,1),vel(24),acc(24)
real*8 time(2),params(10),adl(1,1),ddl(1,1),predef(2,1,8),dtime,pnew,period
real*8 rp(24,1),rm(24,1),kp(24,24),svp(200),ep(8),ref(24,24),ubase(24)
real*8 x(3),xx(3),rot(3,3),angle,jump(3),h,err,forceerr,raw(24)
open(11,file='area_cycle.csv',status='new')
write(11,'(A)') 'index,path_time,area_ratio,normal_jump_mm,external_work_Nmm,recoverable_Nmm,dissipation_Nmm,inplane_force_norm_N'
ndof=24;nrhs=1;ns=200;mcrd=3;nn=8;np=7;nj=1;zero=0;one=1
props=(/.187d0,.781d0,38d0,52d0,1.22d0,1d5,0d0/)
flags=0;flags(1)=1;flags(3)=1;jprops(1)=2
time=0d0;params=0d0;jdl=0;adl=0d0;ddl=0d0;predef=0d0
du=0d0;vel=0d0;acc=0d0;dtime=.001d0;period=1d0;h=1d-8
coords(:,1)=(/0d0,0d0,0d0/);coords(:,2)=(/0d0,0d0,1d0/)
coords(:,3)=(/1d0,0d0,1d0/);coords(:,4)=(/1d0,0d0,0d0/)
coords(:,5:8)=coords(:,1:4)
err=0d0;raw=0d0;ubase=0d0;sv=0d0
do it=0,100
 angle=dble(it)/100d0
 if(angle<=.25d0) then
  jump=(/1d0,.0004d0*angle,0d0/)
 else if(angle<=.5d0) then
  jump=(/1d0+.8d0*(angle-.25d0),.0001d0,0d0/)
 else if(angle<=.75d0) then
  jump=(/1.2d0,.0001d0*(3d0-4d0*angle),0d0/)
 else
  jump=(/1.2d0-.8d0*(angle-.75d0),0d0,0d0/)
 endif
 do n=1,8
  xx=coords(:,n);xx(1)=xx(1)*jump(1)
  if(n>4) xx(2)=xx(2)+jump(2)
  u(3*n-2:3*n)=xx-coords(:,n)
 enddo
 energy=0d0;pnew=1d0
 call UEL(rhs,Kmat,sv,energy,ndof,nrhs,ns,props,np,coords,mcrd,nn,u,du,vel,acc, &
   one,time,dtime,one,one,one,params,zero,jdl,adl,predef,one,flags,ndof,ddl,one,pnew,jprops,nj,period)
 err=err-.5d0*sum((rhs(:,1)+raw)*(u-ubase))
 forceerr=sqrt(sum(rhs(1:24:3,1)**2))
 write(11,'(I0,7(A,ES24.16))') it,',',angle,',',jump(1),',',jump(2),',',err,',',energy(2),',',energy(4),',',forceerr
 raw=rhs(:,1);ubase=u
enddo
close(11)
end program
subroutine MutexLock(i)
integer i
end subroutine
subroutine MutexUnlock(i)
integer i
end subroutine
subroutine MutexInit(i)
integer i
end subroutine
subroutine GETRANK(i)
integer i
i=0
end subroutine

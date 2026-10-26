program element_check
implicit none
integer it,j,k,n,scheme,ip,base,nrhs,ndof,ns,mcrd,nn,np,nj,zero,one
integer flags(10),jprops(1),jdl(1,1)
real*8 rhs(24,1),Kmat(24,24),sv(200),energy(8),props(8),coords(3,8),u(24),du(24,1),vel(24),acc(24)
real*8 time(2),params(10),adl(1,1),ddl(1,1),predef(2,1,8),dtime,pnew,period
real*8 rp(24,1),rm(24,1),kp(24,24),svp(200),ep(8),ref(24,24),ubase(24)
real*8 x(3),xx(3),rot(3,3),angle,jump(3),h,err,forceerr,raw(24)
open(12,file='state.csv',status='new')
write(12,'(A)') 'case_id,field,index,value'
open(11,file='element_tangent.csv',status='new')
write(11,'(A)') 'case_id,angle_deg,normal_jump_mm,shear_jump_mm,tangent_relative_error,resultant_force_norm'
ndof=24;nrhs=1;ns=200;mcrd=3;nn=8;np=8;nj=1;zero=0;one=1
props=(/.187d0,.781d0,38d0,52d0,1.22d0,1d5,0d0,1.000000000000d-08/)
flags=0;flags(1)=1;flags(3)=1;jprops(1)=2
time=0d0;params=0d0;jdl=0;adl=0d0;ddl=0d0;predef=0d0
du=0d0;vel=0d0;acc=0d0;dtime=.001d0;period=1d0;h=1d-8
coords(:,1)=(/0d0,0d0,0d0/);coords(:,2)=(/0d0,0d0,1d0/)
coords(:,3)=(/1d0,0d0,1d0/);coords(:,4)=(/1d0,0d0,0d0/)
coords(:,5:8)=coords(:,1:4)
do it=1,9
 angle=dble(mod(it-1,3))*15d0
 rot=0d0;rot(1,1)=cos(angle*acos(-1d0)/180d0);rot(2,2)=rot(1,1)
 rot(1,2)=-sin(angle*acos(-1d0)/180d0);rot(2,1)=-rot(1,2);rot(3,3)=1d0
 if(it<=3) then
  jump=(/.00005d0,.0001d0,0d0/)
 else if(it<=6) then
  jump=(/.002d0,.003d0,0d0/)
 else
  jump=(/.01d0,.015d0,0d0/)
 endif
 do n=1,8
  xx=matmul(rot,coords(:,n))
  if(n>4) xx=xx+matmul(rot,jump)
  u(3*n-2:3*n)=xx-coords(:,n)
 enddo
 ubase=u;sv=0d0;energy=0d0;pnew=1d0
 call UEL(rhs,Kmat,sv,energy,ndof,nrhs,ns,props,np,coords,mcrd,nn,u,du,vel,acc, &
   one,time,dtime,one,one,one,params,zero,jdl,adl,predef,one,flags,ndof,ddl,one,pnew,jprops,nj,period)
 raw=rhs(:,1)
 do j=1,24
  write(12,'(I0,A,I0,A,I0,A,ES24.16)') it,',',1,',',j,',',rhs(j,1)
  do k=1,24
   write(12,'(I0,A,I0,A,I0,A,ES24.16)') it,',',2,',',(j-1)*24+k,',',Kmat(j,k)
  enddo
 enddo
 do j=1,200
  write(12,'(I0,A,I0,A,I0,A,ES24.16)') it,',',3,',',j,',',sv(j)
 enddo
 do j=1,8
  write(12,'(I0,A,I0,A,I0,A,ES24.16)') it,',',4,',',j,',',energy(j)
 enddo
 do j=1,24
  u=ubase;u(j)=u(j)+h;svp=0d0;ep=0d0
  call UEL(rp,kp,svp,ep,ndof,nrhs,ns,props,np,coords,mcrd,nn,u,du,vel,acc, &
   one,time,dtime,one,one,one,params,zero,jdl,adl,predef,one,flags,ndof,ddl,one,pnew,jprops,nj,period)
  u=ubase;u(j)=u(j)-h;svp=0d0;ep=0d0
  call UEL(rm,kp,svp,ep,ndof,nrhs,ns,props,np,coords,mcrd,nn,u,du,vel,acc, &
   one,time,dtime,one,one,one,params,zero,jdl,adl,predef,one,flags,ndof,ddl,one,pnew,jprops,nj,period)
  ref(:,j)=-(rp(:,1)-rm(:,1))/(2d0*h)
 enddo
 err=sqrt(sum((Kmat-ref)**2))/max(1d0,sqrt(sum(ref**2)))
 x=0d0
 do n=1,8
  x=x+raw(3*n-2:3*n)
 enddo
 forceerr=sqrt(sum(x*x))
 write(11,'(I0,5(A,ES24.16))') it,',',angle,',',jump(2),',',jump(1),',',err,',',forceerr
enddo
close(11)
close(12)
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
subroutine XIT
stop 1
end subroutine

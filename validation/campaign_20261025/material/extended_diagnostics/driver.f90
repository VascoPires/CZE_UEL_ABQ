program material_driver
implicit none
integer id,idx,oldid,ios,j,k
real*8 gi,gii,tn,ts,eta,kn,ks,rho,dt,du(3),dold,dvold,d,dv,lam,l0,lf,b
real*8 t(3,1),c(3,3),tp(3,1),tm(3,1),dp,dvp,lp,l0p,lfp,bp,ref(3,3),pert(3)
real*8 yold,psi,ddiss,y,diss,err,hh
open(10,file='input.txt',status='old')
open(11,file='response.csv',status='new')
write(11,'(A)') 'case_id,index,ds1,ds2,dn,ts1,ts2,tn,D,Dv,lambda,lambda0,lambdaf,B,psi,diss,Y,tangent_error'
oldid=-1
do
 read(10,*,iostat=ios) id,idx,gi,gii,tn,ts,eta,kn,rho,dt,du
 if(ios/=0) exit
 if(id/=oldid) then
  dold=0d0;dvold=0d0;yold=0d0;diss=0d0
 endif
 ks=kn*gi/gii*(ts/tn)**2
 call k_cohesive_law(t,c,gi,gii,tn,ts,eta,kn,ks,du,3,1,dold,dvold,d,dv,lam,l0,lf,b,dt,rho)
 call k_cohesive_energy(gi,gii,eta,kn,ks,du,dold,d,dvold,dv,l0,lf,b,rho,yold,psi,ddiss,y)
 diss=diss+ddiss
 hh=1d-9
 do j=1,3
  pert=du;pert(j)=pert(j)+hh
  call k_cohesive_response(tp,gi,gii,tn,ts,eta,kn,ks,pert,3,1,dold,dvold,dp,dvp,lp,l0p,lfp,bp,dt,rho)
  pert=du;pert(j)=pert(j)-hh
  call k_cohesive_response(tm,gi,gii,tn,ts,eta,kn,ks,pert,3,1,dold,dvold,dp,dvp,lp,l0p,lfp,bp,dt,rho)
  ref(:,j)=(tp(:,1)-tm(:,1))/(2d0*hh)
 enddo
 err=sqrt(sum((c-ref)**2))/max(1d0,sqrt(sum(ref**2)))
 write(11,'(I0,A,I0,16(A,ES24.16))') id,',',idx,(',',du(j),j=1,3), &
   (',',t(j,1),j=1,3),',',d,',',dv,',',lam,',',l0,',',lf,',',b,',',psi,',',diss,',',y,',',err
 dold=d;dvold=dv;yold=y;oldid=id
enddo
close(10);close(11)
end program

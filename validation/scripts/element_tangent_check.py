"""Standalone full UEL residual/tangent diagnostics with serial I/O stubs.

Only echo is disabled; the assembled response is otherwise the supplied source.
Numerically differentiate the residual, not a duplicate material formula.
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from campaign import ROOT,CAMPAIGN,win

DRIVER='''program element_check
implicit none
integer it,j,k,n,scheme,ip,base,nrhs,ndof,ns,mcrd,nn,np,nj,zero,one
integer flags(10),jprops(1),jdl(1,1)
real*8 rhs(24,1),Kmat(24,24),sv(200),energy(8),props(7),coords(3,8),u(24),du(24,1),vel(24),acc(24)
real*8 time(2),params(10),adl(1,1),ddl(1,1),predef(2,1,8),dtime,pnew,period
real*8 rp(24,1),rm(24,1),kp(24,24),svp(200),ep(8),ref(24,24),ubase(24)
real*8 x(3),xx(3),rot(3,3),angle,jump(3),h,err,forceerr,raw(24)
open(11,file='element_tangent.csv',status='new')
write(11,'(A)') 'case_id,angle_deg,normal_jump_mm,shear_jump_mm,tangent_relative_error,resultant_force_norm'
ndof=24;nrhs=1;ns=200;mcrd=3;nn=8;np=7;nj=1;zero=0;one=1
props=(/.187d0,.781d0,38d0,52d0,1.22d0,1d5,0d0/)
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
'''

AREA_LOOP='''err=0d0;raw=0d0;ubase=0d0;sv=0d0
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
'''
AREA_DRIVER=DRIVER[:DRIVER.index('do it=1,9')]+AREA_LOOP+DRIVER[DRIVER.index('close(11)'):]
AREA_DRIVER=AREA_DRIVER.replace("file='element_tangent.csv'","file='area_cycle.csv'")
AREA_DRIVER=AREA_DRIVER.replace("'case_id,angle_deg,normal_jump_mm,shear_jump_mm,tangent_relative_error,resultant_force_norm'",
                                "'index,path_time,area_ratio,normal_jump_mm,external_work_Nmm,recoverable_Nmm,dissipation_Nmm,inplane_force_norm_N'")
BASIS_DRIVER=AREA_DRIVER.replace("file='area_cycle.csv'","file='basis_cycle.csv'")
BASIS_DRIVER=BASIS_DRIVER.replace('real*8 x(3),xx(3),rot(3,3),angle,jump(3),h,err,forceerr,raw(24)',
                                'real*8 x(3),xx(3),rot(3,3),angle,jump(3),h,err,forceerr,raw(24),momentnorm')
BASIS_DRIVER=BASIS_DRIVER.replace('jump=(/1d0,.0004d0*angle,0d0/)', 'jump=(/0d0,.0004d0*angle,0d0/)')
BASIS_DRIVER=BASIS_DRIVER.replace('jump=(/1d0+.8d0*(angle-.25d0),.0001d0,0d0/)', 'jump=(/acos(-1d0)*(angle-.25d0),.0001d0,0d0/)')
BASIS_DRIVER=BASIS_DRIVER.replace('jump=(/1.2d0,.0001d0*(3d0-4d0*angle),0d0/)', 'jump=(/acos(-1d0)/4d0,.0001d0*(3d0-4d0*angle),0d0/)')
BASIS_DRIVER=BASIS_DRIVER.replace('jump=(/1.2d0-.8d0*(angle-.75d0),0d0,0d0/)', 'jump=(/acos(-1d0)*(1d0-angle),0d0,0d0/)')
BASIS_DRIVER=BASIS_DRIVER.replace('xx=coords(:,n);xx(1)=xx(1)*jump(1)',
    'xx=coords(:,n)\n  xx(1)=cos(jump(1))*coords(1,n)-sin(jump(1))*coords(2,n)\n  xx(2)=sin(jump(1))*coords(1,n)+cos(jump(1))*coords(2,n)')
BASIS_DRIVER=BASIS_DRIVER.replace(' forceerr=sqrt(sum(rhs(1:24:3,1)**2))',
''' forceerr=sqrt(sum(rhs(1:24:3,1)**2))
 x=0d0
 do n=1,8
  xx=coords(:,n)+u(3*n-2:3*n)
  x(1)=x(1)+xx(2)*rhs(3*n,1)-xx(3)*rhs(3*n-1,1)
  x(2)=x(2)+xx(3)*rhs(3*n-2,1)-xx(1)*rhs(3*n,1)
  x(3)=x(3)+xx(1)*rhs(3*n-1,1)-xx(2)*rhs(3*n-2,1)
 enddo
 momentnorm=sqrt(sum(x*x))''')
BASIS_DRIVER=BASIS_DRIVER.replace('area_ratio','basis_angle_rad').replace('normal_jump_mm','global_y_jump_mm')
BASIS_DRIVER=BASIS_DRIVER.replace('inplane_force_norm_N','global_x_force_norm_N,moment_norm_Nmm')
BASIS_DRIVER=BASIS_DRIVER.replace("'(I0,7(A,ES24.16))'","'(I0,8(A,ES24.16))'")
BASIS_DRIVER=BASIS_DRIVER.replace("',',energy(4),',',forceerr","',',energy(4),',',forceerr,',',momentnorm")
# Split the longer CSV header for free-form Fortran's132-column limit.
BASIS_DRIVER=BASIS_DRIVER.replace("write(11,'(A)') 'index,path_time,basis_angle_rad,global_y_jump_mm,external_work_Nmm,recoverable_Nmm,dissipation_Nmm,global_x_force_norm_N,moment_norm_Nmm'",
 "write(11,'(A)') 'index,path_time,basis_angle_rad,global_y_jump_mm,external_work_Nmm,' // &\n 'recoverable_Nmm,dissipation_Nmm,global_x_force_norm_N,moment_norm_Nmm'")
BASIS_DRIVER=BASIS_DRIVER.replace("',',err,',',energy(2)","',',err, &\n  ',',energy(2)")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',required=True);parser.add_argument('--source',type=Path,default=ROOT/'src/3D_CZM.f');parser.add_argument('--area-cycle',action='store_true');parser.add_argument('--basis-cycle',action='store_true');parser.add_argument('--area-intervals',type=int,default=100);parser.add_argument('--tangent-perturbation',type=float);parser.add_argument('--out',type=Path);parser.add_argument('--export-state',action='store_true');args=parser.parse_args()
    if args.tangent_perturbation is not None and (not math.isfinite(args.tangent_perturbation) or args.tangent_perturbation<=0):parser.error('Tangent perturbation must be finite and positive')
    if args.area_cycle and args.basis_cycle:parser.error('Select only one closed-cycle test')
    if args.export_state and (args.area_cycle or args.basis_cycle):parser.error('--export-state applies to the nine assembled tangent states')
    folder=args.out or CAMPAIGN/'element_tangent'/args.name;folder.mkdir(parents=True,exist_ok=False)
    source=args.source.read_text();test=source.replace('      echo=1','      echo=0')
    assert test!=source or '      echo=0' in source
    driver=BASIS_DRIVER if args.basis_cycle else AREA_DRIVER if args.area_cycle else DRIVER
    if args.tangent_perturbation is not None:
        driver=driver.replace('props(7)','props(8)').replace('np=7;','np=8;')
        driver=driver.replace('1d5,0d0/)','1d5,0d0,'+('%.12e'%args.tangent_perturbation).replace('e','d')+'/)')
    if args.export_state:
        driver=driver.replace("open(11,file=", "open(12,file='state.csv',status='new')\nwrite(12,'(A)') 'case_id,field,index,value'\nopen(11,file=")
        driver=driver.replace(' raw=rhs(:,1)', ''' raw=rhs(:,1)
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
 enddo''')
        driver=driver.replace('close(11)','close(11)\nclose(12)')
    if args.area_cycle or args.basis_cycle:driver=driver.replace('do it=0,100\n angle=dble(it)/100d0','do it=0,%d\n angle=dble(it)/%dd0'%(args.area_intervals,args.area_intervals))
    (folder/'test_source.f').write_text(test);(folder/'driver.f90').write_text(driver)
    (folder/'ABA_PARAM.INC').write_text('      implicit real*8 (a-h,o-z)\n')
    (folder/'SMAAspUserSubroutines.hdr').write_text('')
    batch='@echo off\r\ncd /d "'+win(folder)+'"\r\ncall "C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\2023.0.0\\env\\vars.bat" intel64 vs2022\r\n'
    batch+='ifort /nologo /O2 /extend-source:132 driver.f90 test_source.f /exe:element_check.exe > compiler.txt 2>&1\r\nif errorlevel 1 exit /b 1\r\nelement_check.exe > execution.txt 2>&1\r\nexit /b %errorlevel%\r\n'
    (folder/'execute.cmd').write_text(batch);result=subprocess.run(['cmd.exe','/c',win(folder/'execute.cmd')],cwd=folder)
    manifest=dict(source=str(args.source),sha256=hashlib.sha256(source.encode()).hexdigest(),returncode=result.returncode,tangent_perturbation=args.tangent_perturbation,test='elastic basis cycle' if args.basis_cycle else 'elastic area cycle' if args.area_cycle else 'assembled tangent',area_intervals=args.area_intervals if args.area_cycle or args.basis_cycle else None,
                  limitations='Standalone serial full-assembly check; echo disabled; Mutex/GETRANK stubs; no Abaqus solver interface verification')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()

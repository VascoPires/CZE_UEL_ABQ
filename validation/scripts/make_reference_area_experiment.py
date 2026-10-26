"""Create an isolated nominal/reference-area hypothesis; never edit canonical UEL.

The traction/energy measure changes, so this is not an approved general fix.
It tests the identified elastic area-cycle defect separately from basis terms.
"""
import difflib
import hashlib
import json
from pathlib import Path

from campaign import ROOT,CAMPAIGN

HELPER='''
! Experimental reference midsurface integration measure.
      subroutine k_reference_area(xi,eta,coords,mcrd,nnode,area)
      INCLUDE 'ABA_PARAM.INC'
      INTEGER mcrd,nnode,i,j,k
      DOUBLE PRECISION xi,eta,coords(mcrd,nnode),area
      DOUBLE PRECISION dN(2,4),mid(3,4),tangent(2,3),cross(3)
      dN(1,1)=-0.25d0*(1.d0-eta)
      dN(1,2)= 0.25d0*(1.d0-eta)
      dN(1,3)= 0.25d0*(1.d0+eta)
      dN(1,4)=-0.25d0*(1.d0+eta)
      dN(2,1)=-0.25d0*(1.d0-xi)
      dN(2,2)=-0.25d0*(1.d0+xi)
      dN(2,3)= 0.25d0*(1.d0+xi)
      dN(2,4)= 0.25d0*(1.d0-xi)
      do j=1,3
         do k=1,4
            mid(j,k)=0.5d0*(coords(j,k)+coords(j,k+4))
         end do
      end do
      tangent=0.d0
      do i=1,2
         do j=1,3
            do k=1,4
               tangent(i,j)=tangent(i,j)+dN(i,k)*mid(j,k)
            end do
         end do
      end do
      cross(1)=tangent(1,2)*tangent(2,3)
     &        -tangent(1,3)*tangent(2,2)
      cross(2)=tangent(1,3)*tangent(2,1)
     &        -tangent(1,1)*tangent(2,3)
      cross(3)=tangent(1,1)*tangent(2,2)
     &        -tangent(1,2)*tangent(2,1)
      area=SQRT(SUM(cross**2))
      return
      end
'''


def main():
    original=(ROOT/'src/3D_CZM.f').read_text()
    changed=original.replace('      DOUBLE PRECISION psi_ip, diss_inc, damage_force',
                             '      DOUBLE PRECISION psi_ip, diss_inc, damage_force, ref_area')
    needle='            jac_w = a_Jacob*wgt'
    assert changed.count(needle)==1
    changed=changed.replace(needle,'! EXPERIMENT: nominal traction and energy per reference area.\n'
        '            call k_reference_area(xi_gp,eta_gp,COORDS,\n'
        '     &           MCRD,NNODE,ref_area)\n            jac_w = ref_area*wgt')
    changed=changed.replace('      echo=1','      echo=0')+HELPER
    folder=CAMPAIGN/'sources';path=folder/'3D_CZM_reference_area_experiment.f'
    with path.open('x') as handle:handle.write(changed)
    (folder/'reference_area_experiment.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile='canonical/src/3D_CZM.f',tofile=path.name)))
    (folder/'reference_area_experiment.json').write_text(json.dumps(dict(canonical_sha256=hashlib.sha256(original.encode()).hexdigest(),
        experimental_sha256=hashlib.sha256(changed.encode()).hexdigest(),changes=['Use undeformed midsurface area for force, tangent and energy integration','Disable visualization I/O only'],
        hypothesis='Interpret cohesive tractions and energies as nominal quantities per reference area',
        approval='NOT promoted to canonical. Changing the intended traction/energy measure requires researcher judgment.',
        limitations='Does not supply missing local-basis geometric terms or establish arbitrary finite-deformation energy consistency. Restricted area cycle and constant-area unit suite are separate tests.'),indent=2)+'\n')
    print(path)


if __name__=='__main__':main()

"""Check input preparation only; does not launch Abaqus or change old decks."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from convert_input import convert_to_uel
from run_case import validate_deck


def main():
    checked=[]
    for deck in sorted((ROOT/'examples').rglob('*.inp')):
        validate_deck(deck.read_text());checked.append(str(deck.relative_to(ROOT)))
    native='*ELEMENT,TYPE=COH3D8,ELSET=I\n1,1,2,3,4,5,6,7,8\n*COHESIVE SECTION,ELSET=I,MATERIAL=C\n,\n'
    cfg=dict(evolution=dict(table=[.187,.781],power=1.22),initiation=[38.,52.],elastic=1e5)
    conversions=[]
    for scheme in [1,2]:
        for zeta in [None,1e-6,1e-8]:
            settings=dict(cfg,integration_scheme=scheme)
            if zeta is not None:settings['tangent_perturbation']=zeta
            text=convert_to_uel(native,settings);assert validate_deck(text)
            expected='PROPERTIES=%d,'%(7 if zeta is None else 8)
            assert expected in text
            if zeta is not None:assert '\n%d\n'%scheme in text
            conversions.append(dict(scheme=scheme,zeta=zeta))
    rejected=[]
    for value in [0.,-1.,float('nan'),float('inf')]:
        try:convert_to_uel(native,dict(cfg,tangent_perturbation=value))
        except ValueError:rejected.append(str(value))
        else:raise AssertionError('Accepted invalid zeta '+str(value))
    good=convert_to_uel(native,dict(cfg,tangent_perturbation=1e-8))
    for name,bad in [
        ('zero',good.replace('1e-08','0')),
        ('negative',good.replace('1e-08','-1e-08')),
        ('nan',good.replace('1e-08','NaN')),
        ('infinity',good.replace('1e-08','Inf')),
        ('crowded_line',good.replace('1e-08\n1','1e-08, 1')),
        ('missing_integer',good.replace('\n1\n','\n')),
        ('declaration_mismatch',good.replace('PROPERTIES=8,','PROPERTIES=7,')),
        ('bad_scheme',good.replace('\n1\n','\n3\n'))]:
        try:validate_deck(bad)
        except ValueError:rejected.append(name)
        else:raise AssertionError('Accepted malformed input '+name)
    # The separate structured research builder uses the same property order.
    sys.path.insert(0,str(ROOT/'validation/scripts'))
    import campaign
    import structured_specimen
    with tempfile.TemporaryDirectory() as temporary:
        campaign.CAMPAIGN=Path(temporary)
        for zeta in [None,1e-8]:
            folder=structured_specimen.build(name='plain' if zeta is None else 'explicit',material='turon2010',
                dx=3,ny=1,tangent_perturbation=zeta)
            assert validate_deck((folder/(folder.name+'.inp')).read_text())
        for kind in ['native','bonded']:
            folder=structured_specimen.build(name=kind,material='turon2010',dx=3,ny=1,**{kind:True})
            assert not validate_deck((folder/(folder.name+'.inp')).read_text())
    result=dict(scope='Input preparation only; no Abaqus jobs',example_decks=checked,
        converted_layouts=conversions,rejected_invalid_cases=rejected,structured_builder='PASS')
    Path(__file__).with_name('workflow_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS:',len(checked),'example decks,',len(conversions),'conversion layouts,',len(rejected),'invalid cases, structured UEL/native/tied builders')


if __name__=='__main__':main()

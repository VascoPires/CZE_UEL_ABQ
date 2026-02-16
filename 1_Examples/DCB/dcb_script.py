
# -*- coding: utf-8 -*-
# DCB with seam crack for VCCT evaluation scripted with functions
# important to have square mesh!


from abaqus import *
from abaqusConstants import *
from caeModules import *
import mesh
import regionToolset
import json
import os.path
import os
import shutil
import sys
# Python 2.7

_LOG_FILE = None


try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_PARENT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

try:
    from prepare_run_from_inp import _patch_inp_for_uel as _prep_patch_inp_for_uel
    from prepare_run_from_inp import _insert_uel_metadata_fixed as _prep_insert_uel_metadata_fixed
except Exception:
    _prep_patch_inp_for_uel = None
    _prep_insert_uel_metadata_fixed = None

session.journalOptions.setValues(replayGeometry=COORDINATE,
                                 recoverGeometry=COORDINATE)


def load_config(config_path):
    """
    Load JSON configuration for materials/specimens.
    """
    if not os.path.isfile(config_path):
        raise IOError('Config file not found: {0}'.format(config_path))
    with open(config_path, 'r') as cfg_file:
        return json.load(cfg_file)

class DCB:
    """
    Class for modeling and analyzing a Double Cantilever Beam (DCB) with a seam crack.
    """
    def __init__(self, TOL, DIRO, l, t, b, a, l2, l1, mesh_size, mesh_size_fine,
                 def_u2=0.0, engineering_const=None, key_load='Displacement',
                 sample_number=1, mat_name='Generic', cohesive_cfg=None, mode='VCCT',
                 number_inc=100000, num_attempts=20, elements_across_thickness=3,
                 pseudo_2d=False, symmetry=False, dynamic_implicit=False, density=1.76e-9,
                 cohesive_tie=False):
        """
        Initializes the DCB model parameters.

        Parameters:
        - TOL (float): Tolerance for node selection
        - DIRO (str): Directory for saving files
        - l (float): Length of the specimen (mm)
        - t (float): Thickness of the specimen (mm)
        - b (float): Width of the specimen (mm)
        - a (float): Crack length (mm)
        - l2 (float): Length parameter for crack propagation
        - l1 (float): Additional length parameter
        - mesh_size (float): Mesh element size (mm)
        - def_u2 (float): Displacement applied for deformation
        - force (float): Applied force (N)
        - engineering_const (tuple): Material engineering constants (E, ν, G values)
        - key_load (str): Type of loading ('Displacement' or 'Force')
        - sample_number (int): Sample identification number
        - mat_name (str): Material name (e.g., 'CFRP', 'GFRP')
        - number_inc (int): Target number of increments (used for load step sizing)
        - num_attempts (int): Max solution attempts in step control settings
        - elements_across_thickness (int): Elements seeded through the thickness for bulk mesh
        - pseudo_2d (bool): Fix U3 on z-faces to mimic a 2D response
        - symmetry (bool): Halve width and fix U3 on mid-plane (z=0) for symmetry modeling
        - dynamic_implicit (bool): Use an implicit dynamic (quasi-static) step instead of static general
        - density (float): Mass density for the bulk material (tonne/mm^3)
        """

        self.TOL = TOL
        self.DIR0 = DIRO

        # parameters for geometry
        self.l = l     # length of specimen in mm
        self.t = t         # thickness of specimen in mm
        self.b = b        # width of specimen in mm
        self.a = a
        self.l2 = l2
        self.l1 = l1
        self.mesh_size = mesh_size # mesh size in mm
        self.def_u2 = def_u2
        self.engineering_const = engineering_const
        self.model = None
        self.p = None
        self.whole = None
        self.inst = None
        self.key_load = key_load
        self.sample_number = sample_number
        self.mat_name = mat_name
        self.mesh_size_fine = mesh_size_fine
        self.cohesive_layer_cells = None
        self.cohesive_cfg = cohesive_cfg or {}
        self.force = self.cohesive_cfg.get('force', 0.0)
        self.elements_across_thickness = int(elements_across_thickness)
        # Thickness of the CZM band; default to fine mesh size unless provided.
        self.cohesive_layer_thickness = self.cohesive_cfg.get('layer_thickness', self.mesh_size_fine)
        self.half_band = 0.5 * float(self.cohesive_layer_thickness)
        self.mode = mode.upper() if isinstance(mode, basestring) else 'VCCT'
        self.number_inc = number_inc
        self.num_attempts = num_attempts
        self.step_name = 'Load Step'
        self.pseudo_2d = bool(pseudo_2d)
        self.symmetry = bool(symmetry)
        self.dynamic_implicit = bool(dynamic_implicit)
        self.density = float(density)
        self.cohesive_tie = bool(cohesive_tie)
        # Force a slim width for pseudo-2D runs regardless of input width.
        if self.pseudo_2d:
            self.b = 2.0
        # Parts/instances for the cohesive-tie workflow
        self.tie_part = None
        self.tie_part_bottom = None
        self.tie_part_top = None
        self.inst_top = None
        self.inst_bottom = None


    def make_model(self):

        if self.model is None:
            Mdb()
            mdb.Model(name='test')
            self.model = mdb.models['test']
            del mdb.models['Model-1']

        return

    def seed_width_single_element(self):
        """
        For pseudo-2D models: enforce one element across width by seeding z-direction edges at key (x, y).
        """
        if not self.pseudo_2d:
            return
        tol = max(self.TOL, 1e-3)
        x_positions = [0.0, self.a, self.a + self.l2, self.a + 0.5 * self.a, self.l]
        y_positions = [0.0, self.t, self.t / 2.0 - self.half_band, self.t / 2.0 + self.half_band]
        edges_to_seed = []
        for x_val in x_positions:
            for y_val in y_positions:
                edges_to_seed += list(self.p.edges.getByBoundingBox(
                    xMin=x_val - tol, xMax=x_val + tol,
                    yMin=y_val - tol, yMax=y_val + tol,
                    zMin=-tol, zMax=self.b + tol
                ))
        if edges_to_seed:
            self.p.seedEdgeByNumber(edges=tuple(edges_to_seed), number=1, constraint=FINER)
        else:
            print("Warning: pseudo-2D width seeding found no edges.")
        return

    def make_geometry(self):
        """
        Creates the geometry of the DCB specimen.
        """
        # create a sketch
        model = self.model
        s = model.ConstrainedSketch(name='my_sketch', sheetSize=200.0)
        # create a rectangle with explicit lines
        s.Line(point1=(0.0, 0.0), point2=(self.l, 0.0))
        s.Line(point1=(self.l, 0.0), point2=(self.l, self.t))
        s.Line(point1=(self.l, self.t), point2=(0.0, self.t))
        

        if self.mode == 'CZM':
            s.Line(point1 = (0.0, self.t), point2=(0.0, self.t/2.0 + self.cohesive_layer_thickness/2.0))
            s.Line(point1 = (0.0, self.t/2.0 + self.cohesive_layer_thickness/2.0), 
                   point2 = (self.a + self.l2, self.t/2.0 + self.cohesive_layer_thickness/2.0))

            s.Line(point1 = (self.a + self.l2, self.t/2.0 + self.cohesive_layer_thickness/2.0), 
                   point2 = (self.a + self.l2, self.t/2.0 - self.cohesive_layer_thickness/2.0))

            s.Line(point1 = (self.a + self.l2, self.t/2.0 - self.cohesive_layer_thickness/2.0), 
                   point2 = (0.0, self.t/2.0 - self.cohesive_layer_thickness/2.0))
            
            s.Line(point1=(0.0, self.t/2.0 - self.cohesive_layer_thickness/2.0), point2=(0.0, 0.0))
            

        elif self.mode == 'VCCT':
            s.Line(point1=(0.0, self.t), point2=(0.0, 0.0))

        # create part
        self.p = model.Part(name='DCB', dimensionality=THREE_D, type=DEFORMABLE_BODY)
        self.p.BaseSolidExtrude(sketch=s, depth=self.b)

        c = self.p.cells
        self.whole = c.getByBoundingBox(zMin=-1000000)


        return

    def slicing_vcct(self):
        """
        Partitions the DCB specimen for VCCT-based crack propagation analysis.
        """
        # datum plane for seam
        c = self.p.cells
        datum_plane1 = self.p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=self.t / 2)
        # partition of crack in over whole length
        whole_model_cell = c.getByBoundingBox(zMin=-1000000)
        self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane1.id], cells=whole_model_cell)


        # datum plane for length of crack
        datum_plane2 = self.p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=self.a + self.l2)
        # partition
        point_tuple = (self.l, self.t-0.1, self.b)
        sequence = (point_tuple,)
        pickedCells = c.findAt(sequence)
        self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane2.id], cells=pickedCells)

        # partition
        point_tuple = (self.l, 0.1, self.b)
        sequence = (point_tuple,)
        pickedCells = c.findAt(sequence)
        self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane2.id], cells=pickedCells)

        datum_plane3 = self.p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=2*self.l2)
        point_tuple = (0.1, self.t, self.b)
        sequence = (point_tuple,)
        pickedCells2 = c.findAt(sequence)
        self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane3.id], cells=pickedCells2)

        point_tuple = (0.1, 0, self.b)
        sequence = (point_tuple,)
        pickedCells2 = c.findAt(sequence)
        self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane3.id], cells=pickedCells2)

        datum_plane4 = self.p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=self.a + self.l2+self.t/2)
        datum_plane5 = self.p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=self.a + self.l2-self.t/2)

        # pickedCells = c.findAt(((self.a, self.t/2+0.1, self.b/2),), ((self.a, self.t/2-0.1, self.b/2),))
        # self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane4.id], cells=pickedCells)
        # c = self.p.cells
        # pickedCells = c.findAt(((self.l, self.t/2-0.1, self.b/2),), ((self.l, self.t/2+0.1, self.b/2),))
        # self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane5.id], cells=pickedCells)


        #recreate whole model after partitioning
        self.whole = c.getByBoundingBox(zMin=-1000000)

        return


    def slicing_czm(self):
        """
        Partitions the DCB specimen for CZM analysis with a finite-thickness interface.
        """

        if self.half_band <= 0.0:
            raise ValueError("cohesive_layer_thickness must be positive; got {0}".format(self.cohesive_layer_thickness))
        
        # Create cut for sketch -> pre-crack

        # s = self.model.ConstrainedSketch(name='cut')
        # s.rectangle(point1=(0, self.t / 2 + half_band), point2=(self.a + self.l2, self.t / 2 - half_band))

        # f = self.p.faces
        # e = self.p.edges
    
        # self.p.CutExtrude(sketchPlane=f.findAt(coordinates=(self.l2/8, self.t/8, self.b)), 
        #     sketchUpEdge=e.findAt(coordinates=(0.0, self.t/8, self.b)), 
        #     sketchPlaneSide=SIDE1, sketch=s)

        # datum planes sandwich the cohesive band to give it finite thickness
        c = self.p.cells
        

        datum_plane_low = self.p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=self.t / 2 - self.half_band)
        datum_plane_up = self.p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=self.t / 2 + self.half_band)
        whole_model_cell = c.getByBoundingBox(zMin=-1000000)
        self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane_low.id], cells=whole_model_cell)

        whole_model_cell = c.getByBoundingBox(zMin=-1000000)
        self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane_up.id], cells=whole_model_cell)

        # datum plane for length of crack
        datum_plane2 = self.p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=self.a + self.l2)
        # partition)
        whole_model_cell = c.getByBoundingBox(zMin=-1000000)
        self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane2.id], cells=whole_model_cell)

        # datum_plane3 = self.p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=2*self.l2)
        # whole_model_cell = c.getByBoundingBox(zMin=-1000000)
        # self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane3.id], cells=whole_model_cell)


        datum_plane4 = self.p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=self.a + self.a/2.0)
        whole_model_cell = c.getByBoundingBox(zMin=-1000000)
        self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane4.id], cells=whole_model_cell)


        #del cells_to_delete

        # pickedCells = c.findAt(((self.a, self.t/2+0.1, self.b/2),), ((self.a, self.t/2-0.1, self.b/2),))
        # self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane4.id], cells=pickedCells)
        # c = self.p.cells
        # pickedCells = c.findAt(((self.l, self.t/2-0.1, self.b/2),), ((self.l, self.t/2+0.1, self.b/2),))
        # self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane5.id], cells=pickedCells)


        #recreate whole model after partitioning
        self.whole = c.getByBoundingBox(zMin=-1000000)
        y_min = self.t / 2 - self.half_band - self.TOL
        y_max = self.t / 2 + self.half_band + self.TOL
        cohesive_cells = c.getByBoundingBox(yMin=y_min, yMax=y_max, zMin=-1000000)
        if cohesive_cells:
            # Store a set so cohesive sections/elements can be assigned later.
            self.p.Set(cells=cohesive_cells, name='Cohesive_Layer')
        self.cohesive_layer_cells = cohesive_cells

        return

    def gen_material(self):
        """
        Assigns bulk material to the solid cells. If a cohesive layer exists, it is excluded.
        """
        if self.engineering_const is None:
            raise ValueError("engineering_const must be provided to generate material properties.")
        my_model = self.model
        material_name = 'Material-1'
        # extract engineering constants
        e1, e2, e3, nu12, nu13, nu23, g12, g13, g23 = self.engineering_const
        # create material
        if material_name not in my_model.materials.keys():
            my_model.Material(name=material_name)
            my_model.materials[material_name].Elastic(table=((e1, e2, e3, nu12, nu13, nu23, g12, g13, g23),),
                                                      type=ENGINEERING_CONSTANTS)
            my_model.materials[material_name].Density(table=((self.density,),))
        else:
            # ensure density exists when reusing an existing material definition
            if not hasattr(my_model.materials[material_name], 'density'):
                my_model.materials[material_name].Density(table=((self.density,),))
        # create section if needed
        section_name = 'all'
        if section_name not in my_model.sections.keys():
            my_model.HomogeneousSolidSection(name=section_name, material=material_name, thickness=None)

        # assign section to bulk (exclude cohesive layer cells via boolean set when present)
        if not self.whole:
            raise ValueError("No cells found for solid material assignment.")
        self.p.Set(cells=self.whole, name='all')
        if self.cohesive_layer_cells and 'Cohesive_Layer' in self.p.sets:
            self.p.SetByBoolean(name='Bulk_material', operation=DIFFERENCE,
                                sets=(self.p.sets['all'], self.p.sets['Cohesive_Layer'],))
            bulk_set = self.p.sets['Bulk_material']
        else:
            bulk_set = self.p.sets['all']

        self.p.SectionAssignment(
            region=bulk_set, sectionName=section_name, offset=0.0, offsetType=MIDDLE_SURFACE,
            offsetField='', thicknessAssignment=FROM_SECTION)
        # orient material
        my_model.parts['DCB'].MaterialOrientation(region=bulk_set,
                                                  orientationType=GLOBAL, axis=AXIS_1, additionalRotationType=ROTATION_NONE,
                                                  localCsys=None, fieldName='', stackDirection=STACK_3)

        return


    def gen_cohesive_material(self):
        """
        Defines cohesive traction-separation material and section.
        """
        if not self.cohesive_cfg:
            raise ValueError("cohesive_cfg must be provided to generate cohesive material.")

        cfg = self.cohesive_cfg
        # Abaqus/CAE (Python 2) expects plain str, not unicode from JSON.
        mat_name = str(cfg.get('name', 'Cohesive-1'))
        elastic = tuple(cfg.get('elastic', (1e6, 1e6, 1e6)))
        initiation = tuple(cfg.get('initiation', (36.0, 36.0, 36.0)))
        evol_cfg = cfg.get('evolution', {})
        evol_power = evol_cfg.get('power', 1.0)
        evol_table = tuple(evol_cfg.get('table', (0.2, 0.8, 0.8)))
        section_name = str(cfg.get('section_name', 'Cohesive-Section'))
        stab_coeff = cfg.get('damage_stabilization_coeff', None)
        coh_density = float(cfg.get('density', self.density))

        my_model = self.model

        my_model.Material(name=mat_name)
        my_model.materials[mat_name].Density(table=((coh_density,),))
        my_model.materials[mat_name].Elastic(type=TRACTION, table=(elastic,))
        my_model.materials[mat_name].QuadsDamageInitiation(table=(initiation,))
        my_model.materials[mat_name].quadsDamageInitiation.DamageEvolution(
            type=ENERGY, mixedModeBehavior=POWER_LAW, power=evol_power, table=(evol_table,))
        if stab_coeff is not None:
            my_model.materials[mat_name].quadsDamageInitiation.DamageStabilizationCohesive(
                cohesiveCoeff=stab_coeff)

        my_model.CohesiveSection(name=section_name, material=mat_name, response=TRACTION_SEPARATION,
                                    outOfPlaneThickness=None, initialThicknessType=SPECIFY, 
                                    initialThickness=1.0)

        return

    def assign_cohesive_section(self):
        """
        Assign cohesive section/element type to the cohesive layer cells.
        """
        if not self.cohesive_layer_cells:
            return

        section_name = str(self.cohesive_cfg.get('section_name', 'Cohesive-Section'))
        if section_name not in self.model.sections.keys():
            raise ValueError("Cohesive section '{0}' not created.".format(section_name))

        region = self.p.Set(cells=self.cohesive_layer_cells, name='Cohesive_Layer')
        self.p.SectionAssignment(
            region=region, sectionName=section_name, offset=0.0, offsetType=MIDDLE_SURFACE,
            offsetField='', thicknessAssignment=FROM_SECTION)

        elem_deletion = bool(self.cohesive_cfg.get('elem_deletion', True))
        viscosity = float(self.cohesive_cfg.get('viscosity', 0.0))
        max_deg = float(self.cohesive_cfg.get('max_degradation', 0.99))

        coh_hex = mesh.ElemType(elemCode=COH3D8, elemLibrary=STANDARD,
                                elemDeletion=elem_deletion, viscosity=viscosity,
                                maxDegradation=max_deg)
        coh_wedge = mesh.ElemType(elemCode=COH3D6, elemLibrary=STANDARD,
                                  elemDeletion=elem_deletion, viscosity=viscosity,
                                  maxDegradation=max_deg)
        coh_tet =  mesh.ElemType(elemCode=UNKNOWN_TET, elemLibrary=STANDARD)
        self.p.setElementType(regions=(region.cells,), elemTypes=(coh_hex, coh_wedge, coh_tet))

        e = self.p.edges
        c = self.p.cells
        pickedEdges = e.getByBoundingBox(xMin = self.a + self.l2 - self.TOL, 
                                         xMax = self.a + self.l2 + self.TOL,
                                         yMin = self.t/2.0 - self.half_band + self.TOL,
                                         yMax = self.t/2.0 + self.half_band  + self.TOL)
        

        self.p.seedEdgeByNumber(edges=pickedEdges, number=1, constraint=FINER)

        # Apply sweep controls to all cohesive cells
        coh_cells = c.getByBoundingBox(xMin=self.a - self.TOL, xMax=self.l + self.TOL,
                                       yMin=self.t/2.0 - self.half_band - self.TOL,
                                       yMax=self.t/2.0 + self.half_band + self.TOL,
                                       zMin=-self.TOL, zMax=self.b + self.TOL)
        if coh_cells:
            self.cohesive_layer_cells = coh_cells
            self.p.setMeshControls(regions=tuple(coh_cells), technique=SWEEP, algorithm=ADVANCING_FRONT)

            
            sweep_edge = e.findAt(coordinates=(self.a + self.a/2.0, self.t / 2.0, self.b))

            if sweep_edge:
                for cell in coh_cells:
                    self.p.setSweepPath(region=cell, edge=sweep_edge, sense=FORWARD)
            else:
                print("Warning: sweep path edge for cohesive cells not found.")
        else:
            print("Warning: cohesive cells not found for sweep control.")

        return

    def assign_stack_directions_bulk(self):
        """
        Assign stack direction for bulk cells to align layers through thickness.
        """
        if self.p is None:
            return
        f = self.p.faces
        try:
            cells_bulk = self.p.cells.getByBoundingBox(zMin=-1e6)
            ref_face = f.findAt(coordinates=(0.2444444 * self.l, self.t, self.b / 3.0))
            self.p.assignStackDirection(referenceRegion=ref_face, cells=cells_bulk)
        except Exception:
            print("Warning: bulk stack direction assignment failed.")
        return

    def assign_stack_directions_cohesive(self):
        """
        Assign stack direction for cohesive layer cells.
        """
        if self.p is None:
            return
        c = self.p.cells
        f = self.p.faces
        tol = max(self.TOL, 1e-3)
        try:
            # cells_coh = c.findAt(
            #     ((0.4888889 * self.l, 0.496 * self.t, 0.0),),
            #     ((0.7 * self.l, 0.504 * self.t, 0.0),)
            # )
            cells_coh = self.p.sets['Cohesive_Layer'].cells
            y_ref = self.t / 2.0 + self.half_band
            ref_face = f.findAt(coordinates=(self.a + 0.6 * self.a, y_ref, self.b / 2.0))
            self.p.assignStackDirection(referenceRegion=ref_face, cells=cells_coh)
        except Exception:
            print("Warning: cohesive stack direction assignment failed.")
        return



    def make_assembly(self):
        """
        Assembles the DCB model by defining constraints and reference points.
        """
        my_model = self.model
        self.assembly = my_model.rootAssembly
        self.inst = self.assembly.Instance(name='DCB-1', part=self.p, dependent=ON)
        f1 = self.inst.faces
        faces2 = f1.findAt(((self.l, self.t * 0.3, self.b / 2),), ((self.l, self.t * 0.8, self.b / 2),))
        self.assembly.Set(faces=faces2, name='fixed_end')

        return

    def assign_seam(self):

        a = self.model.rootAssembly
        a.makeIndependent(instances=(self.inst,))
        f1 = self.inst.faces
        faces1 = f1.findAt(((self.l2 + self.a - 0.1, self.t / 2, self.b / 2),),
                           ((self.l2 / 2, self.t / 2, self.b / 2),), (((self.l2 + self.a)/2, self.t / 2, self.b / 2),))
        crack_surf_set = a.Set(faces=faces1, name='crack_surf')
        a.engineeringFeatures.assignSeam(regions=crack_surf_set)
        return

    def seed_edges_vcct(self):
        """
        Fine seeding near the crack tip for VCCT (uses specific coordinates).
        """
        e = self.p.edges
        pickedEdges = e.findAt(((self.l2 + self.a + 0.1, self.t / 2, self.b),),
                               ((self.l2 + self.a, self.t / 4, self.b),),
                               ((self.l2 + self.a + 0.1, 0.0, self.b),),
                               ((self.l2 + self.a + 0.1, self.t, 0.0),),
                               ((self.l2 + self.a, self.t, self.b / 2),),
                               ((self.l2 + self.a + 0.1, self.t, self.b),),
                               ((self.l2 + self.a + 0.1, self.t / 2, 0.0),),
                               ((self.l2 + self.a, self.t/2+0.1, 0.0),),
                               ((self.l2 + self.a, self.t / 2, self.b / 2),),
                               ((self.l2 + self.a - 0.1, self.t / 2, self.b),),
                               ((self.l2 + self.a, self.t/2+0.1, self.b),),
                               ((self.l2 + self.a - 0.1, self.t, self.b),),
                               ((self.l2 + self.a - 0.1, 0.0, 0.0),),
                               ((self.l2 + self.a, 0.0, self.b / 2),),
                               ((self.l2 + self.a - 0.1, 0.0, self.b),),
                               ((self.l2 + self.a - 0.1, self.t / 2, 0.0),),
                               ((self.l2 + self.a, self.t / 2-0.1, 0.0),),
                               ((self.l2 + self.a - 0.1, self.t, 0.0),),
                               ((self.l2 + self.a + 0.1, 0.0, 0.0),))
        if pickedEdges:
            self.p.seedEdgeBySize(edges=pickedEdges, size=self.mesh_size_fine, deviationFactor=0.1,
                                  minSizeFactor=0.1, constraint=FINER)
        else:
            print("Warning: VCCT fine edge selection failed; no edges seeded.")

    def seed_edges_czm(self):
        """
        Fine seeding for CZM using bounding boxes around the cohesive band.
        """
        e = self.p.edges
        half_band = 0.5 * float(self.cohesive_layer_thickness)
        tol = max(self.mesh_size_fine, self.TOL * 10.0)
        edges_band = e.getByBoundingBox(
            xMin=self.a - tol, xMax=self.a + self.a/2.0 + tol,
            yMin=self.t / 2 - half_band - tol, yMax=self.t / 2 + half_band + tol)
        if edges_band:
            self.p.seedEdgeBySize(edges=edges_band, size=self.mesh_size_fine, deviationFactor=0.1,
                                  minSizeFactor=0.1, constraint=FINER)
        else:
            print("Warning: CZM fine seeding: no edges found near cohesive layer.")
        # Additional fine seeds to cover vertical/horizontal edges per CAE picks
        pickedEdges = e.findAt(
            ((0.5041667 * self.l, self.t, 0.0),),
            ((0.4125 * self.l, 0.0, 0.0),),
            ((0.5041667 * self.l, 0.0, self.b / 5.0),),
            ((0.4125 * self.l, self.t, self.b / 5.0),)
        )
        if pickedEdges:
            self.p.seedEdgeBySize(edges=pickedEdges, size=self.mesh_size_fine, deviationFactor=0.1,
                                  minSizeFactor=0.1, constraint=FINER)

    def seed_vertical_edges_czm(self, n_elements=3):
        """
        Seed vertical edges at key x-positions across thickness/width for CZM.
        """
        tol = max(self.TOL, 1e-3)
        x_positions = [0.0, 2.0 * self.l2, self.a + self.l2, self.l]
        z_positions = [0.0, self.b]
        half_band = 0.5 * float(self.cohesive_layer_thickness)

        edges_to_seed = []
        for x in x_positions:
            for z in z_positions:
                edges_to_seed += [
                    edge for edge in self.p.edges.getByBoundingBox(
                        xMin=x - tol, xMax=x + tol,
                        zMin=z - tol, zMax=z + tol,
                        yMin=-tol, yMax=self.t + tol
                    )
                    if not (self.t/2 - half_band - tol <= edge.pointOn[0][1] <= self.t/2 + half_band + tol)
                ]

        if edges_to_seed:
            self.p.seedEdgeByNumber(edges=tuple(edges_to_seed),
                                    number=n_elements,
                                    constraint=FINER)
        else:
            print("Warning: CZM vertical edge seeding: no edges found.")

    def gen_mesh(self):
        """
        Generates the finite element mesh for the DCB specimen.
        """

        a = self.model.rootAssembly
        if not self.inst.dependent:
            a.makeDependent(instances=(self.inst,))
        self.p.seedPart(size=self.mesh_size, deviationFactor=0.1, minSizeFactor=0.1)

        elemType1 = mesh.ElemType(
            elemCode=C3D8R, elemLibrary=STANDARD, kinematicSplit=AVERAGE_STRAIN,
            secondOrderAccuracy=OFF, hourglassControl=DEFAULT, distortionControl=DEFAULT)
        elem_types = (elemType1,)

        c1 = self.inst.cells
        cells1 = c1.getByBoundingBox(xMin=-self.b, yMin=-self.b, zMin=-self.b)

        if not cells1:
            raise ValueError("No cells found in the bounding box.")

        pickedRegions = (cells1,)
        a.setElementType(regions=pickedRegions, elemTypes=elem_types)

        if self.cohesive_layer_cells:
            # Assign cohesive section/element type then fine seed around cohesive band.
            self.assign_cohesive_section()
            self.seed_edges_czm()
            self.seed_vertical_edges_czm(n_elements=self.elements_across_thickness)
        else:
            # VCCT default edge seeding
            self.seed_edges_vcct()

        # For pseudo-2D runs, collapse width to one element via targeted seeding
        self.seed_width_single_element()

        # Assign stack directions (bulk and cohesive)
        self.assign_stack_directions_bulk()
        if self.cohesive_layer_cells:
            self.assign_stack_directions_cohesive()

        self.p.generateMesh()

        return

    def gen_node_set(self):

        e1 = self.inst.edges
        a = self.model.rootAssembly
        e_crack = e1.findAt(((self.a + self.l2, self.t / 2, self.b / 2),))
        crack_edge_set = a.Set(edges=e_crack, name='crack_edge')
        crack_node = crack_edge_set.nodes
        c_node_set = a.Set(nodes=crack_node, name='crack_nodes')
        c_node_labels = [node.label for node in c_node_set.nodes]

        return


    def gen_node_set_b4_crack(self):

        a = self.model.rootAssembly
        n = self.inst.nodes
        nodes_infront_crack = n.getByBoundingBox(xMin=(self.a + self.l2 - self.mesh_size_fine * 1.5), xMax=(self.a + self.l2 - self.mesh_size_fine * 0.5),
                                                 yMin=(self.t/2-self.TOL), yMax=(self.t/2+self.TOL),
                                                 zMin=(-0.1 * self.b), zMax=(self.b + 0.1 * self.b))
        node_set_infront_crack = a.Set(nodes=nodes_infront_crack, name='node_set_before_crack')
        print(len(nodes_infront_crack))
        node_labels = [node.label for node in nodes_infront_crack]
        # print(node_labels)
        labels_coords_b4 = []

        for i in range(len(node_labels)):
            c1 = node_set_infront_crack.nodes[i].coordinates[0]
            c2 = node_set_infront_crack.nodes[i].coordinates[1]
            c3 = node_set_infront_crack.nodes[i].coordinates[2]
            label = node_set_infront_crack.nodes[i].label
            labels_coords_b4.append([label, c1, c2, c3])

        l_c_b4 = sorted(labels_coords_b4, key=lambda x: x[3])

        return


    def gen_node_set_czm(self):
        """
        Create node sets just below/above the cohesive layer for CZM.
        """

        #n = self.p.nodes
        n = self.inst.nodes
        tol = self.TOL

        self.assembly.regenerate()

        bottom_nodes = n.getByBoundingBox(
            yMin=self.t / 2 - self.half_band - tol,
            yMax=self.t / 2 - self.half_band + tol)
        if bottom_nodes:
            self.assembly.Set(nodes=bottom_nodes, name='CZM_BOTTOM')

        else:
            print("Warning: CZM_BOTTOM node set is empty.")

        upper_nodes = n.getByBoundingBox(
            yMin=self.t / 2 + self.half_band - tol,
            yMax=self.t / 2 + self.half_band + tol)
        if upper_nodes:
            self.assembly.Set(nodes=upper_nodes, name='CZM_TOP')
        else:
            print("Warning: CZM_TOP node set is empty.")

        self.assembly.editNode(nodes=bottom_nodes + upper_nodes, coordinate2=self.t/2.0)


        # n1 = self.inst.nodes
        # nodes1 = n1.getByBoundingBox(yMin=self.t / 2 + self.half_band - tol,
        #                              yMax=self.t / 2 + self.half_band + tol)

        # nodes2 = n1.getByBoundingBox(yMin=self.t / 2 - self.half_band - tol,
        #                              yMax=self.t / 2 - self.half_band + tol)

        # self.assembly.editNode(nodes=nodes1, coordinate2=self.t/2.0)
        # self.assembly.editNode(nodes=nodes2, coordinate2=self.t/2.0)

        # self.assembly.regenerate()

        return


    def apply_pseudo_2d_constraints(self, instances=None):
        """
        Optionally fix U3 on the front/back faces to emulate a 2D response.
        """
        if not self.pseudo_2d or self.symmetry:
            return

        tol = max(self.TOL, 1e-6)
        a = self.model.rootAssembly
        if instances is None:
            instances = ()
            if self.inst is not None:
                instances = (self.inst,)

        for inst in instances:
            faces = inst.faces
            tag = inst.name.replace('-', '_')
            z0_faces = faces.getByBoundingBox(zMin=-tol, zMax=tol)
            zb_faces = faces.getByBoundingBox(zMin=self.b - tol, zMax=self.b + tol)

            if z0_faces:
                region_z0 = a.Set(faces=z0_faces, name='Pseudo2D_Z0_{0}'.format(tag))
                self.model.DisplacementBC(
                    name='Pseudo2D_Z0_U3_{0}'.format(tag), createStepName='Initial', region=region_z0,
                    u1=UNSET, u2=UNSET, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET,
                    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='',
                    localCsys=None)
            else:
                print("Warning: Pseudo-2D: no faces found at z=0 for U3 fix on {0}.".format(inst.name))

            if zb_faces:
                region_zb = a.Set(faces=zb_faces, name='Pseudo2D_ZB_{0}'.format(tag))
                self.model.DisplacementBC(
                    name='Pseudo2D_ZB_U3_{0}'.format(tag), createStepName='Initial', region=region_zb,
                    u1=UNSET, u2=UNSET, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET,
                    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='',
                    localCsys=None)
            else:
                print("Warning: Pseudo-2D: no faces found at z=b for U3 fix on {0}.".format(inst.name))

        return


    def apply_symmetry_constraints(self, instances=None):
        """
        Apply symmetry along z=0 by fixing U3 there (width is already halved).
        """
        tol = max(self.TOL, 1e-6)
        a = self.model.rootAssembly
        if instances is None:
            instances = ()
            if self.inst is not None:
                instances = (self.inst,)

        for inst in instances:
            faces = inst.faces
            tag = inst.name.replace('-', '_')
            z0_faces = faces.getByBoundingBox(zMin=-tol, zMax=tol)
            if z0_faces:
                region_z0 = a.Set(faces=z0_faces, name='Symmetry_Z0_{0}'.format(tag))
                self.model.DisplacementBC(
                    name='Symmetry_Z0_U3_{0}'.format(tag), createStepName='Initial', region=region_z0,
                    u1=UNSET, u2=UNSET, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET,
                    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='',
                    localCsys=None)
            else:
                print("Warning: Symmetry flag set but no faces found at z=0 for U3 fix on {0}.".format(inst.name))

        return


    def create_load_step(self):
        """
        Create the load step with consistent incrementation for all workflows.
        """
        my_model = self.model
        step_name = self.step_name
        if self.number_inc <= 0:
            raise ValueError("number_inc must be positive; got {0}".format(self.number_inc))
        max_num_inc = int(self.number_inc)
        kwargs_default = {
            'nlgeom': ON,
            'initialInc': 0.1,
            'minInc': 1E-20,
            'maxInc': 0.1,
            'maxNumInc': max_num_inc
        }
        if self.dynamic_implicit:
            load_step = my_model.ImplicitDynamicsStep(
                name=step_name,
                previous=my_model.steps.keys()[-1],
                description='deformation of DCB (quasi-static implicit dynamic)',
                application=QUASI_STATIC,
                nohaf=OFF,
                amplitude=RAMP,
                alpha=DEFAULT,
                initialConditions=OFF,
                **kwargs_default
            )
            load_step.control.setValues(
                allowPropagation=OFF,
                resetDefaultValues=OFF,
                timeIncrementation=(4.0, 8.0, 9.0, 16.0, 10.0, 4.0, 12.0, self.num_attempts, 6.0, 3.0, 50.0))
        else:
            load_step = my_model.StaticStep(
                name=step_name, previous=my_model.steps.keys()[-1], description='deformation of DCB', **kwargs_default)
            load_step.control.setValues(
                allowPropagation=OFF,
                resetDefaultValues=OFF,
                timeIncrementation=(4.0, 8.0, 9.0, 16.0, 10.0, 4.0, 12.0, self.num_attempts, 6.0, 3.0, 50.0))

        if 'F-Output-1' in my_model.fieldOutputRequests.keys():
            my_model.fieldOutputRequests['F-Output-1'].setValues(
                variables=('CDISP', 'CF', 'COORD', 'CSTRESS', 'LE', 'NFORC', 'RF', 'S', 'U', 'SDEG'))
        return load_step


    def gen_BC(self):
        my_model = self.model
        d_u2 = self.def_u2
        step_name = self.step_name
        self.create_load_step()

        a = my_model.rootAssembly
        if self.symmetry:
            self.apply_symmetry_constraints()
        else:
            self.apply_pseudo_2d_constraints()

        r1 = a.referencePoints
        # create Reference points
        # refP_up = a.ReferencePoint([self.l2, self.l1 + self.t, self.b / 2])
        # refP_low = a.ReferencePoint([self.l2, -self.l1, self.b / 2])

        refP_up = a.ReferencePoint([0.0, self.t, self.b / 2])
        refP_low = a.ReferencePoint([0.0, 0.0, self.b / 2])

        refPoints1 = (r1[refP_up.id],)
        refPoints2 = (r1[refP_low.id],)

        a.Set(referencePoints=refPoints1, name='rp_up')
        regionDef = self.model.rootAssembly.sets['rp_up']
        self.model.HistoryOutputRequest(name='rf_up', createStepName=step_name,
                                                variables=('U2', 'RF2', 'CF2'), region=regionDef, sectionPoints=DEFAULT,
                                                rebar=EXCLUDE)

        a.Set(referencePoints=refPoints2, name='rp_down')
        regionDef = self.model.rootAssembly.sets['rp_down']
        self.model.HistoryOutputRequest(name='rf_down', createStepName=step_name,
                                        variables=('U2', 'RF2', 'CF2'), region=regionDef, sectionPoints=DEFAULT,
                                        rebar=EXCLUDE)

        del self.model.historyOutputRequests['H-Output-1']

        # f1 = self.inst.faces
        # faces_up = f1.findAt(((self.l2 / 2, self.t, self.b / 2),))
        #     #f1.getByBoundingBox(xMin=0.0, xMax=self.b0*2, yMin=self.t, yMax=self.t, zMin=0.0, zMax=self.w)
        # region_surface_up = a.Surface(side1Faces=faces_up, name='coupling_surf_up')

        # refPoint_up = (r1[refP_up.id],)
        # region_refP_up = regionToolset.Region(referencePoints=refPoint_up)

        # # create lower coupling
        # faces_low = f1.findAt(((self.l2 / 2, 0, self.b / 2),))
        # region_surface_low = a.Surface(side1Faces=faces_low, name='coupling_surf_low')

        
        
        f1 = self.inst.edges
        faces_up = f1.findAt(((0.0, self.t, self.b / 2),))
        #f1.getByBoundingBox(xMin=0.0, xMax=self.b0*2, yMin=self.t, yMax=self.t, zMin=0.0, zMax=self.w)

        region_surface_up = a.Set(edges=faces_up, name='coupling_surf_up')

        refPoint_up = (r1[refP_up.id],)
        region_refP_up = regionToolset.Region(referencePoints=refPoint_up)

        # create lower coupling
        faces_low = f1.findAt(((0.0, 0, self.b / 2),))
        region_surface_low = a.Set(edges=faces_low, name='coupling_surf_low')


        # region_surface_low = a.sets['coupling_surf_low']
        refPoint_low = (r1[refP_low.id],)
        region_refP_low = regionToolset.Region(referencePoints=refPoint_low)

        

        if self.key_load == 'Displacement':

            my_model.Coupling(
                name='ref_upper', controlPoint=region_refP_up, surface=region_surface_up, influenceRadius=WHOLE_SURFACE,
                couplingType=KINEMATIC, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON,
                ur3=ON)

            my_model.Coupling(
                    name='ref_lower', controlPoint=region_refP_low, surface=region_surface_low, influenceRadius=WHOLE_SURFACE,
                    couplingType=KINEMATIC, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON,
                    ur3=ON)
            # create deformation of upper couple
            my_model.DisplacementBC(
                  name='upper_move', createStepName=step_name, region=region_refP_up, u1=0, u2=d_u2/2.0, u3=0,
                  ur1=0, ur2=0, ur3=0, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='',
                  localCsys=None)

            # create deformation of lower couple
            my_model.DisplacementBC(
                  name='lower_move', createStepName=step_name, region=region_refP_low, u1=0, u2=-d_u2/2.0, u3=0,
                  ur1=0, ur2=0, ur3=0, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM,
                  fieldName='', localCsys=None)

        if self.key_load == 'Force':

            my_model.Coupling(
                name='ref_upper', controlPoint=region_refP_up, surface=region_surface_up, influenceRadius=WHOLE_SURFACE,
                couplingType=STRUCTURAL, weightingMethod=UNIFORM, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=OFF, ur3=OFF)

            my_model.Coupling(
                name='ref_lower', controlPoint=region_refP_low, surface=region_surface_low, influenceRadius=WHOLE_SURFACE,
                couplingType=STRUCTURAL, weightingMethod=UNIFORM, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=OFF,
                ur3=OFF)

            my_model.ConcentratedForce(name='lower_move', createStepName=step_name,
                                       region=region_refP_low, cf2=-self.force, distributionType=UNIFORM, field='',
                                       localCsys=None)
            my_model.ConcentratedForce(name='upper_move', createStepName=step_name,
                                       region=region_refP_up, cf2=self.force, distributionType=UNIFORM, field='',
                                       localCsys=None)

        # fixed end boundary condition at the far edge
        # if 'fixed_end' in a.sets:
        #     my_model.EncastreBC(name='fix', createStepName='Initial', region=a.sets['fixed_end'], localCsys=None)
        # else:
        #     print("Warning: 'fixed_end' set not found; encastre BC not applied.")


        return


    def make_half_part_for_tie(self):
        """
        Build two half-thickness solids for the cohesive-tie workflow (one per instance).
        """
        model = self.model
        half_t = 0.5 * self.t
        sketch_size = max(self.l, self.t) * 2.0
        parts = []
        for part_name, sketch_name in (('DCB_HALF_BOTTOM', 'half_sketch_bottom'),
                                       ('DCB_HALF_TOP', 'half_sketch_top')):
            if part_name in model.parts.keys():
                del model.parts[part_name]
            sketch = model.ConstrainedSketch(name=sketch_name, sheetSize=sketch_size)
            sketch.rectangle(point1=(0.0, 0.0), point2=(self.l, half_t))
            part = model.Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
            part.BaseSolidExtrude(sketch=sketch, depth=self.b)
            parts.append(part)

        self.tie_part_bottom, self.tie_part_top = parts
        # Keep legacy attributes pointing somewhere sensible for downstream helpers.
        self.tie_part = self.tie_part_bottom
        self.p = self.tie_part_bottom
        return tuple(parts)


    def partition_half_for_tie(self, part):
        """
        Partition the half block along x to split cracked/intact regions for the interface surfaces.
        """
        c = part.cells
        x_cuts_raw = [self.a, self.a + self.l2, 2.0 * self.l2, self.a + 0.5 * self.a]
        x_cuts = []
        for x_val in x_cuts_raw:
            if x_val is None:
                continue
            if x_val <= self.TOL or x_val >= self.l - self.TOL:
                continue
            if all(abs(x_val - existing) > self.TOL for existing in x_cuts):
                x_cuts.append(x_val)

        for x_cut in x_cuts:
            datum = part.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=x_cut)
            whole_cell = c.getByBoundingBox(zMin=-1000000)
            part.PartitionCellByDatumPlane(datumPlane=part.datums[datum.id], cells=whole_cell)

        return


    def assign_material_to_parts(self, parts):
        """
        Assign bulk material/section to multiple parts (used by the cohesive-tie workflow).
        """
        if self.engineering_const is None:
            raise ValueError("engineering_const must be provided to generate material properties.")

        my_model = self.model
        material_name = 'Material-1'
        e1, e2, e3, nu12, nu13, nu23, g12, g13, g23 = self.engineering_const
        if material_name not in my_model.materials.keys():
            my_model.Material(name=material_name)
            my_model.materials[material_name].Elastic(table=((e1, e2, e3, nu12, nu13, nu23, g12, g13, g23),),
                                                      type=ENGINEERING_CONSTANTS)
            my_model.materials[material_name].Density(table=((self.density,),))
        else:
            if not hasattr(my_model.materials[material_name], 'density'):
                my_model.materials[material_name].Density(table=((self.density,),))
        section_name = 'all'
        if section_name not in my_model.sections.keys():
            my_model.HomogeneousSolidSection(name=section_name, material=material_name, thickness=None)

        for part in parts:
            cells = part.cells
            if not cells:
                raise ValueError("No cells found for material assignment on part {0}".format(part.name))
            set_name = 'all_{0}'.format(part.name)
            part.Set(cells=cells, name=set_name)
            part.SectionAssignment(
                region=part.sets[set_name], sectionName=section_name, offset=0.0, offsetType=MIDDLE_SURFACE,
                offsetField='', thicknessAssignment=FROM_SECTION)
            part.MaterialOrientation(region=part.sets[set_name],
                                     orientationType=GLOBAL, axis=AXIS_1, additionalRotationType=ROTATION_NONE,
                                     localCsys=None, fieldName='', stackDirection=STACK_3)

        return


    def seed_vertical_edges_cohesive_tie(self, part, n_elements=None):
        """
        Seed vertical edges across thickness for matched meshes between the two halves.
        """
        if n_elements is None:
            n_elements = self.elements_across_thickness
        tol = max(self.TOL, 1e-3)
        x_candidates = [0.0, self.a, self.a + self.l2, self.l]
        x_positions = []
        for val in x_candidates:
            if val is None:
                continue
            if val < -tol or val > self.l + tol:
                continue
            if all(abs(val - existing) > tol for existing in x_positions):
                x_positions.append(val)
        z_positions = [0.0, self.b]
        edges_to_seed = []
        for x in x_positions:
            for z in z_positions:
                edges_to_seed += list(part.edges.getByBoundingBox(
                    xMin=x - tol, xMax=x + tol,
                    zMin=z - tol, zMax=z + tol,
                    yMin=-tol, yMax=self.t + tol
                ))
        if edges_to_seed:
            part.seedEdgeByNumber(edges=tuple(edges_to_seed),
                                  number=int(n_elements),
                                  constraint=FINER)
        else:
            print("Warning: Cohesive-tie vertical edge seeding found no edges.")

        return


    def seed_part_for_cohesive_tie(self, part):
        """
        Apply coarse/fine seeding and element types for the cohesive-tie half block.
        """
        part.seedPart(size=self.mesh_size, deviationFactor=0.1, minSizeFactor=0.1)

        elemType1 = mesh.ElemType(
            elemCode=C3D8R, elemLibrary=STANDARD, kinematicSplit=AVERAGE_STRAIN,
            secondOrderAccuracy=OFF, hourglassControl=ENHANCED, distortionControl=DEFAULT)
        elem_types = (elemType1,)
        part.setElementType(regions=(part.cells,), elemTypes=elem_types)

        tol = max(self.mesh_size_fine, self.TOL * 10.0)
        edges_front = part.edges.getByBoundingBox(
            xMin=self.a - tol, xMax=self.a + self.a/2.0 + tol,
            yMin=-tol, yMax=self.t + tol, zMin=-tol, zMax=self.b + tol)
        if edges_front:
            part.seedEdgeBySize(edges=edges_front, size=self.mesh_size_fine, deviationFactor=0.1,
                                minSizeFactor=0.1, constraint=FINER)
        else:
            print("Warning: Cohesive-tie fine edge selection failed; no edges seeded near the crack tip.")

        self.seed_vertical_edges_cohesive_tie(part, n_elements=self.elements_across_thickness)
        part.generateMesh()
        return


    def make_assembly_cohesive_tie(self):
        """
        Instantiate two halves and position them with a coincident mid-plane.
        """
        if self.tie_part_bottom is None or self.tie_part_top is None:
            raise ValueError("Cohesive-tie parts are not created; call make_half_part_for_tie first.")
        a = self.model.rootAssembly
        self.assembly = a
        self.inst_bottom = a.Instance(name='DCB-BOTTOM', part=self.tie_part_bottom, dependent=ON)
        self.inst_top = a.Instance(name='DCB-TOP', part=self.tie_part_top, dependent=ON)
        a.translate(instanceList=('DCB-TOP',), vector=(0.0, self.t / 2.0, 0.0))
        # Keep self.inst pointing to a valid instance for shared utilities.
        self.inst = self.inst_bottom
        return


    def create_interface_surfaces_for_tie(self):
        """
        Create interface surfaces (cracked/intact) and tie the intact segment.
        """
        if self.inst_top is None or self.inst_bottom is None:
            raise ValueError("Cohesive-tie instances are missing; call make_assembly_cohesive_tie first.")
        a = self.model.rootAssembly
        tol = max(self.TOL, 1e-3)
        interface_y = self.t / 2.0

        top_all = self.inst_top.faces.getByBoundingBox(
            xMin=-tol, xMax=self.l + tol, yMin=interface_y - tol, yMax=interface_y + tol)
        bot_all = self.inst_bottom.faces.getByBoundingBox(
            xMin=-tol, xMax=self.l + tol, yMin=interface_y - tol, yMax=interface_y + tol)

        top_intact = self.inst_top.faces.getByBoundingBox(
            xMin=self.a - tol, xMax=self.l + tol, yMin=interface_y - tol, yMax=interface_y + tol)
        bot_intact = self.inst_bottom.faces.getByBoundingBox(
            xMin=self.a - tol, xMax=self.l + tol, yMin=interface_y - tol, yMax=interface_y + tol)

        top_crack = self.inst_top.faces.getByBoundingBox(
            xMin=-tol, xMax=self.a + tol, yMin=interface_y - tol, yMax=interface_y + tol)
        bot_crack = self.inst_bottom.faces.getByBoundingBox(
            xMin=-tol, xMax=self.a + tol, yMin=interface_y - tol, yMax=interface_y + tol)

        if top_all:
            a.Surface(side1Faces=top_all, name='Interface_Tie_Top_All')
        else:
            print("Warning: no top interface faces found for cohesive-tie workflow.")
        if bot_all:
            a.Surface(side1Faces=bot_all, name='Interface_Tie_Bot_All')
        else:
            print("Warning: no bottom interface faces found for cohesive-tie workflow.")

        surf_top_intact = None
        surf_bot_intact = None

        
        if top_intact:
            surf_top_intact = a.Surface(side1Faces=top_intact, name='Interface_Tie_Top_Intact')
        else:
            print("Warning: no intact top faces found for cohesive-tie workflow.")
        if bot_intact:
            surf_bot_intact = a.Surface(side1Faces=bot_intact, name='Interface_Tie_Bot_Intact')
        else:
            print("Warning: no intact bottom faces found for cohesive-tie workflow.")

        if top_crack:
            a.Surface(side1Faces=top_crack, name='Interface_Tie_Top_Crack')
        if bot_crack:
            a.Surface(side1Faces=bot_crack, name='Interface_Tie_Bot_Crack')

        if surf_top_intact and surf_bot_intact:
            self.model.Tie(
                name='Interface_Tie_Intact', master=surf_top_intact, slave=surf_bot_intact,
                positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON)
        else:
            print("Warning: intact interface tie not created (missing surfaces).")

        return


    def gen_BC_cohesive_tie(self):
        """
        Apply BCs/couplings for the cohesive-tie workflow (two arms, same loading as baseline).
        """
        my_model = self.model
        d_u2 = self.def_u2
        step_name = self.step_name
        self.create_load_step()

        a = my_model.rootAssembly
        insts = tuple(inst for inst in (self.inst_bottom, self.inst_top) if inst is not None)
        if self.symmetry:
            self.apply_symmetry_constraints(instances=insts)
        else:
            self.apply_pseudo_2d_constraints(instances=insts)

        r1 = a.referencePoints
        refP_up = a.ReferencePoint([0.0, self.t, self.b / 2])
        refP_low = a.ReferencePoint([0.0, 0.0, self.b / 2])

        refPoints1 = (r1[refP_up.id],)
        refPoints2 = (r1[refP_low.id],)

        a.Set(referencePoints=refPoints1, name='rp_up_tie')
        regionDef = self.model.rootAssembly.sets['rp_up_tie']
        self.model.HistoryOutputRequest(name='rf_up_tie', createStepName=step_name,
                                        variables=('U2', 'RF2', 'CF2'), region=regionDef, sectionPoints=DEFAULT,
                                        rebar=EXCLUDE)

        a.Set(referencePoints=refPoints2, name='rp_down_tie')
        regionDef = self.model.rootAssembly.sets['rp_down_tie']
        self.model.HistoryOutputRequest(name='rf_down_tie', createStepName=step_name,
                                        variables=('U2', 'RF2', 'CF2'), region=regionDef, sectionPoints=DEFAULT,
                                        rebar=EXCLUDE)

        if 'H-Output-1' in self.model.historyOutputRequests.keys():
            del self.model.historyOutputRequests['H-Output-1']

        tol = max(self.TOL, 1e-3)
        top_edges = self.inst_top.edges.getByBoundingBox(
            xMin=-tol, xMax=tol, yMin=self.t - tol, yMax=self.t + tol, zMin=self.b / 2 - tol, zMax=self.b / 2 + tol)
        bot_edges = self.inst_bottom.edges.getByBoundingBox(
            xMin=-tol, xMax=tol, yMin=-tol, yMax=tol, zMin=self.b / 2 - tol, zMax=self.b / 2 + tol)

        region_surface_up = None
        region_surface_low = None
        if top_edges:
            region_surface_up = a.Set(edges=top_edges, name='coupling_surf_up_tie')
        else:
            print("Warning: top coupling edge set is empty for cohesive-tie workflow.")
        if bot_edges:
            region_surface_low = a.Set(edges=bot_edges, name='coupling_surf_low_tie')
        else:
            print("Warning: bottom coupling edge set is empty for cohesive-tie workflow.")

        region_refP_up = regionToolset.Region(referencePoints=refPoints1)
        region_refP_low = regionToolset.Region(referencePoints=refPoints2)

        if self.key_load == 'Displacement' and region_surface_up and region_surface_low:

            my_model.Coupling(
                name='ref_upper_tie', controlPoint=region_refP_up, surface=region_surface_up, influenceRadius=WHOLE_SURFACE,
                couplingType=KINEMATIC, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON,
                ur3=ON)

            my_model.Coupling(
                name='ref_lower_tie', controlPoint=region_refP_low, surface=region_surface_low, influenceRadius=WHOLE_SURFACE,
                couplingType=KINEMATIC, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON,
                ur3=ON)
            my_model.DisplacementBC(
                name='upper_move_tie', createStepName=step_name, region=region_refP_up, u1=0, u2=d_u2/2.0, u3=0,
                ur1=0, ur2=0, ur3=0, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='',
                localCsys=None)

            my_model.DisplacementBC(
                name='lower_move_tie', createStepName=step_name, region=region_refP_low, u1=0, u2=-d_u2/2.0, u3=0,
                ur1=0, ur2=0, ur3=0, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM,
                fieldName='', localCsys=None)

        if self.key_load == 'Force' and region_surface_up and region_surface_low:

            my_model.Coupling(
                name='ref_upper_tie', controlPoint=region_refP_up, surface=region_surface_up, influenceRadius=WHOLE_SURFACE,
                couplingType=STRUCTURAL, weightingMethod=UNIFORM, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=OFF, ur3=OFF)

            my_model.Coupling(
                name='ref_lower_tie', controlPoint=region_refP_low, surface=region_surface_low, influenceRadius=WHOLE_SURFACE,
                couplingType=STRUCTURAL, weightingMethod=UNIFORM, localCsys=None, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=OFF,
                ur3=OFF)

            my_model.ConcentratedForce(name='lower_move_tie', createStepName=step_name,
                                       region=region_refP_low, cf2=-self.force, distributionType=UNIFORM, field='',
                                       localCsys=None)
            my_model.ConcentratedForce(name='upper_move_tie', createStepName=step_name,
                                       region=region_refP_up, cf2=self.force, distributionType=UNIFORM, field='',
                                       localCsys=None)

        return


    def run_model_cohesive_tie(self, submit_job=False):
        """
        Cohesive-tie workflow: two half-thickness instances aligned for zero-thickness cohesive insertion.
        """
        self.make_model()
        bottom_part, top_part = self.make_half_part_for_tie()
        half_parts = (bottom_part, top_part)
        for part in half_parts:
            self.partition_half_for_tie(part)
        self.assign_material_to_parts(list(half_parts))
        for part in half_parts:
            self.seed_part_for_cohesive_tie(part)
        self.make_assembly_cohesive_tie()
        self.assembly.regenerate()
        self.create_interface_surfaces_for_tie()
        self.gen_BC_cohesive_tie()
        run_name_cfg = getattr(self, 'run_name', None)
        job_name, job = self.create_job(suffix='_COHTIE', run_name=run_name_cfg)
        if submit_job:
            job.submit(consistencyChecking=OFF)
            job.waitForCompletion()
        return job_name



    def create_job(self, suffix='', run_name=None):
        """
        Creates an Abaqus job object (not submitted unless requested).
        """
        base = run_name if run_name else 'DCB_{0}_sample{1}'.format(self.mat_name, self.sample_number)
        job_name = '{0}{1}'.format(base, suffix)
        job_name = str(job_name)
        mdb.saveAs(str(job_name))
        job = mdb.Job(
            name=job_name, model='test', description=job_name, type=ANALYSIS,
            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=60,
            memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
            modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
            scratch='', resultsFormat=ODB,
            multiprocessingMode=DEFAULT, numCpus=4, numDomains=4, numGPUs=0)
        return job_name, job


    def run_model_vcct(self, submit_job=False):
        """
        VCCT workflow: geometry, partitions, material, seam, mesh, and BCs.
        """
        self.make_model()
        self.make_geometry()
        self.slicing_vcct()
        self.gen_material()
        self.make_assembly()
        self.assign_seam()
        self.gen_mesh()
        self.gen_node_set()
        self.gen_node_set_b4_crack()
        self.gen_BC()
        run_name_cfg = getattr(self, 'run_name', None)
        job_name, job = self.create_job(run_name=run_name_cfg)
        if submit_job:
            job.submit(consistencyChecking=OFF)
            job.waitForCompletion()
        return job_name


    def run_model_czm(self, submit_job=False, return_job=False):
        """
        CZM workflow: geometry, partitions (finite cohesive band), materials, cohesive section, mesh, BCs.
        """
        self.make_model()
        self.make_geometry()
        self.slicing_czm()
        self.gen_material()
        self.gen_cohesive_material()
        self.make_assembly()
        self.gen_mesh()
        self.gen_BC()
        self.gen_node_set_czm()
        run_name_cfg = getattr(self, 'run_name', None)
        job_name, job = self.create_job(suffix='_CZM', run_name=run_name_cfg)
        if submit_job:
            job.submit(consistencyChecking=OFF)
            job.waitForCompletion()
        if return_job:
            return job_name, job
        return job_name



TOL = 1e-3
DIR0 = os.path.abspath('')
try:
    _BASE_DIR = os.path.dirname(__file__)
except NameError:
    _BASE_DIR = os.getcwd()
CONFIG_PATH = os.path.join(_BASE_DIR, 'dcb_config.json')
config = load_config(CONFIG_PATH)


def build_dcb_from_config(config_path=CONFIG_PATH):
    """
    Build a DCB object from JSON config with minimal user touchpoints.
    """
    config = load_config(config_path)

    material_key = config.get('material_key', 'GFRP')
    specimen_index = config.get('specimen_index', 0)
    key_load = config.get('key_load', 'Displacement')
    mode = config.get('mode', 'VCCT').upper()
    pseudo_2d = bool(config.get('Pseudo_2D', False))
    symmetry = bool(config.get('Symmetry', False))
    dynamic_implicit = bool(config.get('dynamic_implicit', False))
    cohesive_tie = bool(config.get('cohesive_tie', False))
    density = float(config.get('density', 1.76e-9))
    run_name = config.get('run_name', None)

    materials_cfg = config.get('materials', {})
    if material_key not in materials_cfg:
        raise ValueError('Material key "{0}" not found in config.'.format(material_key))

    mat_cfg = materials_cfg[material_key]
    specimens = mat_cfg.get('specimens', [])
    if specimen_index < 0 or specimen_index >= len(specimens):
        raise IndexError('Specimen index {0} out of range for material "{1}".'.format(specimen_index, material_key))

    specimen = specimens[specimen_index]
    geometry_cfg = config.get('geometry', {})
    mesh_cfg = config.get('mesh', {})
    cohesive_cfg = config.get('cohesive', None)
    if not cohesive_cfg:
        cohesive_cfg = config.get('cohesive_standard', {})
    config_def_u2 = config.get('def_u2', None)
    number_inc = config.get('number_inc', 100000)
    num_attempts = config.get('num_attempts', 20)

    l = geometry_cfg.get('l', 200.0)
    t = specimen['t']
    b = specimen['b']
    a = specimen['a']
    l1 = specimen['l1']
    l2 = specimen['l2']
    def_u2 = specimen.get('def_u2', 0.0)
    if config_def_u2 is not None:
        def_u2 = config_def_u2
    sample_number = specimen.get('sample_number', specimen_index + 1)

    eng = mat_cfg['engineering_const']
    engineering_const = (eng['e1'], eng['e2'], eng['e3'], eng['nu_12'], eng['nu_13'], eng['nu_23'], eng['g12'], eng['g13'], eng['g23'])

    # Symmetry halves width; mesh sizes are derived from thickness so they stay untouched.
    if symmetry:
        b = b / 2.0

    mesh_size = specimen.get('mesh_size', t / float(mesh_cfg.get('global_divisor', 4.0)))
    mesh_size_fine = specimen.get('mesh_size_fine', t / float(mesh_cfg.get('fine_divisor', 16.0)))
    elements_across_thickness = specimen.get('elements_across_thickness', mesh_cfg.get('elements_across_thickness', 3))

    dcb = DCB(TOL, DIR0, l, t, b, a, l2, l1, mesh_size, mesh_size_fine, def_u2, engineering_const, key_load, sample_number, material_key, cohesive_cfg, mode, number_inc=number_inc, num_attempts=num_attempts, elements_across_thickness=elements_across_thickness, pseudo_2d=pseudo_2d, symmetry=symmetry, dynamic_implicit=dynamic_implicit, density=density, cohesive_tie=cohesive_tie)
    dcb.run_name = run_name
    return dcb, config, material_key, specimen_index


def summarize_choice(material_key, specimen_index, config):
    materials_cfg = config.get('materials', {})
    mat_cfg = materials_cfg[material_key]
    specimen = mat_cfg['specimens'][specimen_index]
    summary = [
        'Material: {0}'.format(material_key),
        'Specimen index: {0} (sample {1})'.format(specimen_index, specimen.get('sample_number', specimen_index + 1)),
        'Geometry: t={0}, b={1}, a={2}, l1={3}, l2={4}'.format(specimen['t'], specimen['b'], specimen['a'], specimen['l1'], specimen['l2']),
        'Load key: {0}, def_u2={1}'.format(config.get('key_load', 'Displacement'), specimen.get('def_u2', 0.0)),
        'Mode: {0}'.format(config.get('mode', 'VCCT')),
        'Cohesive tie workflow: {0}'.format(config.get('cohesive_tie', False))
    ]
    for line in summary:
        print(line)


def _log(message):
    print('[DCB] {0}'.format(message))
    try:
        sys.stdout.flush()
    except Exception:
        pass
    if _LOG_FILE:
        try:
            with open(_LOG_FILE, 'a') as log_handle:
                log_handle.write('[DCB] {0}\n'.format(message))
        except Exception:
            pass


def _set_log_file(path):
    global _LOG_FILE
    _LOG_FILE = path




def _move_job_outputs(job_name, source_dir, target_dir):
    if not os.path.isdir(source_dir):
        return
    if not os.path.isdir(target_dir):
        return
    for fname in os.listdir(source_dir):
        if not fname.startswith(job_name):
            continue
        src = os.path.join(source_dir, fname)
        dst = os.path.join(target_dir, fname)
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if not os.path.isfile(src):
            continue
        try:
            shutil.move(src, dst)
        except Exception as exc:
            _log('Move failed for {0}: {1}'.format(src, exc))




def _patch_inp_for_uel(inp_path, cohesive_uel_cfg, run_folder, uel_subroutine_path):
    if _prep_patch_inp_for_uel is None:
        raise ImportError('prepare_run_from_inp._patch_inp_for_uel could not be imported.')
    if _prep_insert_uel_metadata_fixed is None:
        raise ImportError('prepare_run_from_inp._insert_uel_metadata_fixed could not be imported.')

    manual_cfg = dict(cohesive_uel_cfg)
    manual_cfg['use_manual'] = True
    first_label = _prep_patch_inp_for_uel(inp_path, manual_cfg)
    _prep_insert_uel_metadata_fixed(uel_subroutine_path, run_folder, first_label)
    _log('UEL metadata written (first_el_label={0}).'.format(first_label))
    return first_label


def run_from_config(config_path=CONFIG_PATH):
    dcb, config, material_key, specimen_index = build_dcb_from_config(config_path)
    summarize_choice(material_key, specimen_index, config)
    mode = config.get('mode', 'VCCT').upper()
    cohesive_tie = bool(config.get('cohesive_tie', False))
    run_subroutine = bool(config.get('Run Subroutine', False))
    cohesive_uel_cfg = config.get('cohesive_UEL', {})
    uel_subroutine_path = config.get('uel_subroutine_path', os.path.join(_BASE_DIR, '3D_CZM.f'))

    # Make a run folder, copy the config for traceability
    job_tag = 'COHTIE' if cohesive_tie else ('VCCT' if mode == 'VCCT' else 'CZM')
    run_name_cfg = config.get('run_name', None)
    folder_name = run_name_cfg if run_name_cfg else 'run_{0}_{1}_{2}'.format(material_key, specimen_index, job_tag)
    if run_name_cfg and os.path.basename(DIR0) == run_name_cfg:
        run_folder = DIR0
    else:
        run_folder = os.path.join(DIR0, 'runs', folder_name)
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
    cfg_copy_path = os.path.join(run_folder, 'dcb_config_copy.json')
    shutil.copy(config_path, cfg_copy_path)
    _set_log_file(os.path.join(run_folder, 'dcb_run.log'))
    _log('Run folder: {0}'.format(run_folder))
    _log('Run subroutine: {0}, mode: {1}'.format(run_subroutine, mode))
    _log('Working directory: {0}'.format(os.getcwd()))
    if run_subroutine:
        if cohesive_tie:
            raise ValueError('Run Subroutine is not supported for cohesive_tie workflow.')
        if mode != 'CZM':
            raise ValueError('Run Subroutine requires mode "CZM".')
        if not cohesive_uel_cfg:
            raise ValueError('cohesive_UEL config must be provided when Run Subroutine is enabled.')

        uel_subroutine_path = os.path.abspath(uel_subroutine_path)
        if not os.path.isfile(uel_subroutine_path):
            raise IOError('UEL subroutine not found: {0}'.format(uel_subroutine_path))

        uel_copy_path = os.path.join(run_folder, os.path.basename(uel_subroutine_path))
        if os.path.abspath(uel_subroutine_path) != os.path.abspath(uel_copy_path):
            shutil.copy(uel_subroutine_path, uel_copy_path)
        uel_subroutine_path = uel_copy_path

        job_name, job = dcb.run_model_czm(submit_job=False, return_job=True)
        inp_path = os.path.join(run_folder, job_name + '.inp')
        job.writeInput(consistencyChecking=OFF)
        _log('Patching input deck: {0}'.format(inp_path))
        _patch_inp_for_uel(inp_path, cohesive_uel_cfg, run_folder, uel_subroutine_path)
        _log('Input deck patched for UEL.')

        uel_job_name = str(job_name + '_UEL')
        inp_path = str(inp_path)
        uel_subroutine_path = str(uel_subroutine_path)
        uel_job = mdb.JobFromInputFile(
            name=uel_job_name, inputFileName=inp_path, userSubroutine=uel_subroutine_path,
            multiprocessingMode=DEFAULT, numCpus=4, numDomains=4)
        _log('Submitting UEL job: {0}'.format(uel_job_name))
        uel_job.submit(consistencyChecking=OFF)
        uel_job.waitForCompletion()
        _log('UEL job completed: {0}'.format(uel_job_name))
    else:
        if cohesive_tie:
            job_name = dcb.run_model_cohesive_tie()
        elif mode == 'VCCT':
            job_name = dcb.run_model_vcct()
        elif mode == 'CZM':
            job_name = dcb.run_model_czm()
        else:
            raise ValueError('Unsupported mode "{0}". Use "VCCT" or "CZM".'.format(mode))
        _log('Model build complete (job not submitted): {0}'.format(job_name))

    # Save the model in the run folder for convenience
    try:
        shutil.move(job_name + '.cae', os.path.join(run_folder, job_name + '.cae'))
    except:
        pass
    return dcb

if __name__ == '__main__':
    # Minimal entry: adjust dcb_config.json, then run.
    dcb = run_from_config()




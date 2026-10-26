# -*- coding: utf-8 -*-
# MMB with cohesive zone model (CZM) scripted with functions
# important to have square mesh!

from abaqus import *
from abaqusConstants import *
from caeModules import *
import mesh
import regionToolset
import json
import os
import os.path
import sys
# Executed by Abaqus CAE (Python 2.7 or 3).

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


class MMB:
    """
    Class for modeling and analyzing an MMB specimen with CZM.
    """
    def __init__(self, TOL, DIRO, l, t, b, a, l2, l1, L, C, loading_height, mesh_size, mesh_size_fine,
                 def_u2=0.0, engineering_const=None, key_load='Displacement',
                 sample_number=1, mat_name='Generic', cohesive_cfg=None,
                 number_inc=100000, num_attempts=20, elements_across_thickness=3,
                 pseudo_2d=False, symmetry=False, dynamic_implicit=False, density=1.76e-9):
        """
        Initializes the MMB model parameters.

        Parameters:
        - TOL (float): Tolerance for node selection
        - DIRO (str): Directory for saving files
        - l (float): Length of the specimen (mm)
        - t (float): Thickness of the specimen (mm)
        - b (float): Width of the specimen (mm)
        - a (float): Crack length (mm)
        - l2 (float): Length parameter for crack propagation
        - l1 (float): Additional length parameter
        - L (float): Loading arm span (mm)
        - C (float): Loading arm offset (mm)
        - loading_height (float): Height of the loading arm above the specimen (mm)
        - mesh_size (float): Mesh element size (mm)
        - def_u2 (float): Displacement applied for deformation
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
        self.L = L
        self.C = C
        self.loading_height = loading_height
        self.mesh_size = mesh_size # mesh size in mm
        self.def_u2 = def_u2
        self.engineering_const = engineering_const
        self.model = None
        self.p = None
        self.whole = None
        self.inst = None
        self.apparatus_part = None
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
        self.number_inc = number_inc
        self.num_attempts = num_attempts
        self.step_name = 'Load Step'
        self.pseudo_2d = bool(pseudo_2d)
        self.symmetry = bool(symmetry)
        self.dynamic_implicit = bool(dynamic_implicit)
        self.density = float(density)
        # Force a slim width for pseudo-2D runs regardless of input width.
        if self.pseudo_2d:
            self.b = 2.0


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
        x_positions = [0.0, self.a, self.a + self.l2, self.L, 2.0*self.L, self.l]
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

    def seed_apparatus_width_single_element(self):
        """
        Seed the apparatus width with one element by targeting z-direction edges at key (x, y) locations.
        """
        if self.apparatus_part is None:
            return
        tol = max(self.TOL, 1e-3)
        z_pick = self.b / 2.0
        y_top = self.loading_height
        y_bot = self.t
        x0 = 0.0
        xL = self.L
        xLC = self.L + self.C
        e = self.apparatus_part.edges
        try:
            edges_to_seed = e.findAt(
                ((xL, y_top, z_pick),),
                ((xLC, y_top, z_pick),),
                ((x0, y_top, z_pick),),
                ((x0, y_bot, z_pick),),
                ((xL, y_bot, z_pick),)
            )
        except Exception:
            edges_to_seed = ()
        if edges_to_seed:
            self.apparatus_part.seedEdgeByNumber(edges=edges_to_seed, number=1, constraint=FINER)
        else:
            print("Warning: apparatus width seeding found no edges.")
        return

    def mesh_apparatus(self):
        """
        Mesh the rigid apparatus with structured quads and a single element across the width.
        """
        if self.apparatus_part is None:
            return
        self.apparatus_part.seedPart(size=self.mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
        faces = self.apparatus_part.faces
        if faces:
            self.apparatus_part.setMeshControls(regions=faces, elemShape=QUAD, technique=STRUCTURED)
        self.seed_apparatus_width_single_element()
        try:
            self.apparatus_part.generateMesh()
        except Exception:
            print("Warning: apparatus mesh generation failed.")
        return

    def make_geometry(self):
        """
        Creates the geometry of the MMB specimen for CZM.
        """
        # create a sketch
        model = self.model
        s = model.ConstrainedSketch(name='my_sketch', sheetSize=200.0)
        # create a rectangle with explicit lines
        s.Line(point1=(0.0, 0.0), point2=(self.l, 0.0))
        s.Line(point1=(self.l, 0.0), point2=(self.l, self.t))
        s.Line(point1=(self.l, self.t), point2=(0.0, self.t))

        s.Line(point1=(0.0, self.t), point2=(0.0, self.t / 2.0 + self.cohesive_layer_thickness / 2.0))
        s.Line(point1=(0.0, self.t / 2.0 + self.cohesive_layer_thickness / 2.0),
               point2=(self.a + self.l2, self.t / 2.0 + self.cohesive_layer_thickness / 2.0))

        s.Line(point1=(self.a + self.l2, self.t / 2.0 + self.cohesive_layer_thickness / 2.0),
               point2=(self.a + self.l2, self.t / 2.0 - self.cohesive_layer_thickness / 2.0))

        s.Line(point1=(self.a + self.l2, self.t / 2.0 - self.cohesive_layer_thickness / 2.0),
               point2=(0.0, self.t / 2.0 - self.cohesive_layer_thickness / 2.0))

        s.Line(point1=(0.0, self.t / 2.0 - self.cohesive_layer_thickness / 2.0), point2=(0.0, 0.0))

        # create part
        self.p = model.Part(name='MMB', dimensionality=THREE_D, type=DEFORMABLE_BODY)
        self.p.BaseSolidExtrude(sketch=s, depth=self.b)

        c = self.p.cells
        self.whole = c.getByBoundingBox(zMin=-1000000)

        # create apparatus (rigid loading arm)
        s1 = model.ConstrainedSketch(name='apparatus_profile', sheetSize=200.0)
        s1.setPrimaryObject(option=STANDALONE)
        s1.Line(point1=(0.0, self.t), point2=(0.0, self.loading_height))
        s1.Line(point1=(0.0, self.loading_height), point2=(self.L, self.loading_height))
        s1.Line(point1=(self.L, self.loading_height), point2=(self.L, self.t))
        s1.Line(point1=(self.L, self.loading_height), point2=(self.L + self.C, self.loading_height))
        p_app = model.Part(name='Apparatus', dimensionality=THREE_D, type=DISCRETE_RIGID_SURFACE)
        p_app.BaseShellExtrude(sketch=s1, depth=self.b)
        s1.unsetPrimaryObject()
        self.apparatus_part = p_app

        return


    def slicing_czm(self):
        """
        Partitions the MMB specimen for CZM analysis with a finite-thickness interface.
        """

        if self.half_band <= 0.0:
            raise ValueError("cohesive_layer_thickness must be positive; got {0}".format(self.cohesive_layer_thickness))

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
        whole_model_cell = c.getByBoundingBox(zMin=-1000000)
        self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane2.id], cells=whole_model_cell)

        datum_plane4 = self.p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=self.L)
        whole_model_cell = c.getByBoundingBox(zMin=-1000000)
        self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane4.id], cells=whole_model_cell)

        datum_plane4b = self.p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=2.0 * self.L)
        whole_model_cell = c.getByBoundingBox(zMin=-1000000)
        self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane4b.id], cells=whole_model_cell)


        #datum_plane5 = self.p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=self.L+self.C)
        #whole_model_cell = c.getByBoundingBox(zMin=-1000000)
        #self.p.PartitionCellByDatumPlane(datumPlane=self.p.datums[datum_plane5.id], cells=whole_model_cell)


        # recreate whole model after partitioning
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
        my_model.parts['MMB'].MaterialOrientation(region=bulk_set,
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
        evolution_type = str(evol_cfg.get('type', 'B-K')).upper()
        if evolution_type in ('B-K', 'BK'):
            mixed_behavior = BK
        elif evolution_type in ('POWER-LAW', 'POWER_LAW'):
            mixed_behavior = POWER_LAW
        else:
            raise ValueError('Unsupported cohesive evolution type: '+evolution_type)
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
            type=ENERGY, mixedModeBehavior=mixed_behavior, power=evol_power, table=(evol_table,))
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
        coh_tet = mesh.ElemType(elemCode=UNKNOWN_TET, elemLibrary=STANDARD)
        self.p.setElementType(regions=(region.cells,), elemTypes=(coh_hex, coh_wedge, coh_tet))

        e = self.p.edges
        c = self.p.cells
        pickedEdges = e.getByBoundingBox(xMin=self.a + self.l2 - self.TOL,
                                         xMax=self.a + self.l2 + self.TOL,
                                         yMin=self.t / 2.0 - self.half_band + self.TOL,
                                         yMax=self.t / 2.0 + self.half_band + self.TOL)

        self.p.seedEdgeByNumber(edges=pickedEdges, number=1, constraint=FINER)

        # Apply sweep controls to all cohesive cells
        coh_cells = c.getByBoundingBox(xMin=self.a - self.TOL, xMax=self.l + self.TOL,
                                       yMin=self.t / 2.0 - self.half_band - self.TOL,
                                       yMax=self.t / 2.0 + self.half_band + self.TOL,
                                       zMin=-self.TOL, zMax=self.b + self.TOL)
        if coh_cells:
            self.cohesive_layer_cells = coh_cells
            self.p.setMeshControls(regions=tuple(coh_cells), technique=SWEEP, algorithm=ADVANCING_FRONT)

            coh_cell1 = c.getByBoundingBox(xMin=self.a - self.TOL, xMax=self.L + self.TOL,
                                           yMin=self.t / 2.0 - self.half_band - self.TOL,
                                           yMax=self.t / 2.0 + self.half_band + self.TOL,
                                           zMin=-1E1, zMax=1E1)

            coh_cell2 = c.getByBoundingBox(xMin=self.L - self.TOL, xMax=2.0*self.L + self.TOL,
                                yMin=self.t / 2.0 - self.half_band - self.TOL,
                                yMax=self.t / 2.0 + self.half_band + self.TOL,
                                zMin=-1E1, zMax=1E1)

            coh_cell3 = c.getByBoundingBox(xMin=2.0*self.L - self.TOL, xMax=self.l + self.TOL,
                    yMin=self.t / 2.0 - self.half_band - self.TOL,
                    yMax=self.t / 2.0 + self.half_band + self.TOL,
                    zMin=-1E1, zMax=1E1)

            sweep_edge_L = e.findAt(coordinates=(self.L, self.t / 2.0, self.b))
            sweep_edge_l = e.findAt(coordinates=(self.l, self.t / 2.0, self.b))

            def _apply_sweep(cell_seq, edge, label):
                if not cell_seq:
                    print("Warning: {0} is empty; sweep path not set.".format(label))
                    return
                for cell in cell_seq:
                    self.p.setSweepPath(region=cell, edge=edge, sense=FORWARD)

            _apply_sweep(coh_cell1, sweep_edge_L, 'coh_cell1')
            _apply_sweep(coh_cell2, sweep_edge_L, 'coh_cell2')
            _apply_sweep(coh_cell3, sweep_edge_l, 'coh_cell3')


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
        try:
            cells_coh = c.findAt(
                ((0.4888889 * self.l, 0.496 * self.t, 0.0),),
                ((0.7 * self.l, 0.504 * self.t, 0.0),)
            )
            y_ref = self.t / 2.0 + self.half_band
            ref_face = f.findAt(coordinates=(self.a + 0.6 * self.a, y_ref, self.b / 2.0))
            self.p.assignStackDirection(referenceRegion=ref_face, cells=cells_coh)
        except Exception:
            print("Warning: cohesive stack direction assignment failed.")
        return


    def make_assembly(self):
        """
        Assembles the MMB model by defining constraints and reference points.
        """
        my_model = self.model
        self.assembly = my_model.rootAssembly
        self.inst = self.assembly.Instance(name='MMB-1', part=self.p, dependent=ON)
        if self.apparatus_part is not None:
            self.assembly.Instance(name='Apparatus-1', part=self.apparatus_part, dependent=ON)
        f1 = self.inst.faces
        faces2 = f1.findAt(((self.l, self.t * 0.3, self.b / 2),), ((self.l, self.t * 0.8, self.b / 2),))
        self.assembly.Set(faces=faces2, name='fixed_end')

        return

    def seed_edges_czm(self):
        """
        Fine seeding for CZM using bounding boxes around the cohesive band.
        """
        e = self.p.edges
        half_band = 0.5 * float(self.cohesive_layer_thickness)
        tol = max(self.mesh_size_fine, self.TOL * 10.0)
        edges_band = e.getByBoundingBox(
            xMin=self.a - 0.1*self.a + tol, xMax=self.L + self.C + tol)
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
                    if not (self.t / 2 - half_band - tol <= edge.pointOn[0][1] <= self.t / 2 + half_band + tol)
                ]

        if edges_to_seed:
            self.p.seedEdgeByNumber(edges=tuple(edges_to_seed),
                                    number=n_elements,
                                    constraint=FINER)
        else:
            print("Warning: CZM vertical edge seeding: no edges found.")

    def gen_mesh(self):
        """
        Generates the finite element mesh for the MMB specimen.
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

        if not self.cohesive_layer_cells:
            raise ValueError("Cohesive layer cells not found; check slicing_czm.")

        # Assign cohesive section/element type then fine seed around cohesive band.

        self.assign_cohesive_section()
        self.seed_edges_czm()
        self.seed_vertical_edges_czm(n_elements=self.elements_across_thickness)

        # For pseudo-2D runs, collapse width to one element via targeted seeding
        self.seed_width_single_element()

        # Assign stack directions (bulk and cohesive)
        self.assign_stack_directions_bulk()
        self.assign_stack_directions_cohesive()

        self.p.generateMesh()
        self.mesh_apparatus()

        return

    def gen_node_set_czm(self):
        """
        Create node sets just below/above the cohesive layer for CZM.
        """

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

        self.assembly.editNode(nodes=bottom_nodes + upper_nodes, coordinate2=self.t / 2.0)

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
                previous=list(my_model.steps.keys())[-1],
                description='deformation of MMB (quasi-static implicit dynamic)',
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
                name=step_name, previous=list(my_model.steps.keys())[-1], description='deformation of MMB', **kwargs_default)
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

        rp_region = None
        rp_set = None

#       r1 = a.referencePoints
        # create Reference points
 #       refP_up = a.ReferencePoint([0.0, self.t, self.b / 2])


#        refPoints1 = (r1[refP_up.id],)


#        a.Set(referencePoints=refPoints1, name='rp_up')
#        regionDef = self.model.rootAssembly.sets['rp_up']
#        self.model.HistoryOutputRequest(name='rf_up', createStepName=step_name,
#                                        variables=('U2', 'RF2', 'CF2'), region=regionDef, sectionPoints=DEFAULT,
#                                        rebar=EXCLUDE)

        if 'Apparatus-1' in a.instances.keys():
            if 'loading_rp' in a.sets.keys():
                del a.sets['loading_rp']
            if 'rigid_body' in a.sets.keys():
                del a.sets['rigid_body']
            if 'Rigid_body_const' in my_model.constraints.keys():
                del my_model.constraints['Rigid_body_const']
            if 'disp' in my_model.boundaryConditions.keys():
                del my_model.boundaryConditions['disp']

            rp_coords = (self.L + self.C, self.loading_height, self.b / 2.0)
            rp = a.ReferencePoint(point=rp_coords)
            rp_ref = a.referencePoints[rp.id]
            rp_set = a.Set(referencePoints=(rp_ref,), name='loading_rp')
            rp_region = regionToolset.Region(referencePoints=(rp_ref,))

            app_faces = a.instances['Apparatus-1'].faces
            region2 = a.Set(faces=app_faces, name='rigid_body')
            my_model.RigidBody(name='Rigid_body_const', refPointRegion=rp_region, bodyRegion=region2)

            if 'rf_up' in my_model.historyOutputRequests.keys():
                del my_model.historyOutputRequests['rf_up']
            my_model.HistoryOutputRequest(
                name='rf_up', createStepName=step_name,
                variables=('U2', 'RF2', 'CF2'), region=rp_set,
                sectionPoints=DEFAULT, rebar=EXCLUDE)

            if self.key_load == 'Displacement':
                my_model.DisplacementBC(
                    name='disp', createStepName=step_name, region=rp_set,
                    u1=UNSET, u2=-d_u2, u3=0.0, ur1=0.0, ur2=0.0, ur3=UNSET,
                    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='',
                    localCsys=None)
                my_model.boundaryConditions['disp'].setValues(ur1=0.0, ur2=0.0)


        if 'H-Output-1' in self.model.historyOutputRequests.keys():
            del self.model.historyOutputRequests['H-Output-1']

        if 'Tie_edge' in my_model.constraints.keys():
            del my_model.constraints['Tie_edge']
        if 'edge_BC' in my_model.boundaryConditions.keys():
            del my_model.boundaryConditions['edge_BC']
        if 'bc_roller' in my_model.boundaryConditions.keys():
            del my_model.boundaryConditions['bc_roller']
        for set_name in ('tie_edge_Lever', 'Tie_edge_specimen',
                         'edge_BC', 'bc_roller', 'lever_0', 'lever_b',
                         'spec_lever_0', 'spec_lever_b'):
            if set_name in a.sets.keys():
                del a.sets[set_name]

        if 'Apparatus-1' in a.instances.keys() and 'MMB-1' in a.instances.keys():
            e_app = a.instances['Apparatus-1'].edges
            e_mmb = a.instances['MMB-1'].edges
            z_pick = self.b / 2.0

            edges_lever = e_app.findAt(((0.0, self.t, z_pick),))
            region_lever = a.Set(edges=edges_lever, name='tie_edge_Lever')
            edges_spec = e_mmb.findAt(((0.0, self.t, z_pick),))
            region_spec = a.Set(edges=edges_spec, name='Tie_edge_specimen')
            my_model.Tie(
                name='Tie_edge', main=region_lever, secondary=region_spec,
                positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON)

            edges_bc = e_mmb.findAt(((0.0, 0.0, z_pick),))
            region_bc = a.Set(edges=edges_bc, name='edge_BC')
            my_model.DisplacementBC(
                name='edge_BC', createStepName='Initial', region=region_bc,
                u1=SET, u2=SET, u3=SET, ur1=UNSET, ur2=UNSET, ur3=UNSET,
                amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)

            edges_roller_bc = e_mmb.findAt(((2.0 * self.L, 0.0, z_pick),))
            region_roller_bc = a.Set(edges=edges_roller_bc, name='bc_roller')
            my_model.DisplacementBC(
                name='bc_roller', createStepName='Initial', region=region_roller_bc,
                u1=UNSET, u2=SET, u3=SET, ur1=UNSET, ur2=UNSET, ur3=UNSET,
                amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)


        f1 = self.inst.edges
        faces_up = f1.findAt(((0.0, self.t, self.b / 2),))
        region_surface_up = a.Set(edges=faces_up, name='coupling_surf_up')

#        refPoint_up = (r1[refP_up.id],)
#        region_refP_up = regionToolset.Region(referencePoints=refPoint_up)



        return


    def gen_equations(self):

        a = mdb.models['test'].rootAssembly
        if 'MMB-1' in a.instances.keys():
            v_spec = a.instances['MMB-1'].vertices
            verts_b = v_spec.findAt(((self.L, self.t, self.b), ))
            roller_b_specimen_set = a.Set(vertices=verts_b, name='roller_b_specimen_set')
            nodes_b = roller_b_specimen_set.nodes
            a.Set(nodes=nodes_b, name='specimen_b')

            verts_0 = v_spec.findAt(((self.L, self.t, 0.0), ))
            roller_0_specimen_set = a.Set(vertices=verts_0, name='roller_0_specimen_set')
            nodes_0 = roller_0_specimen_set.nodes
            a.Set(nodes=nodes_0, name='specimen_0')

        if 'Apparatus-1' in a.instances.keys():
            v_app = a.instances['Apparatus-1'].vertices
            verts_0_app = v_app.findAt(((self.L, self.t, 0.0), ))
            roller_0_app_set = a.Set(vertices=verts_0_app, name='roller_0_lever_set')
            nodes_0_app = roller_0_app_set.nodes
            a.Set(nodes=nodes_0_app, name='lever_0')
            verts_b_app = v_app.findAt(((self.L, self.t, self.b), ))
            roller_b_app_set = a.Set(vertices=verts_b_app, name='roller_b_lever_set')
            nodes_b_app = roller_b_app_set.nodes
            a.Set(nodes=nodes_b_app, name='lever_b')

        if all(name in a.sets.keys() for name in ('specimen_0', 'lever_0', 'specimen_b', 'lever_b')):
            if 'Eq_Roller_0' in self.model.constraints.keys():
                del self.model.constraints['Eq_Roller_0']
            if 'Eq_Roller_b' in self.model.constraints.keys():
                del self.model.constraints['Eq_Roller_b']
            self.model.Equation(
                name='Eq_Roller_0',
                terms=((1.0, 'specimen_0', 2), (-1.0, 'lever_0', 2)))
            self.model.Equation(
                name='Eq_Roller_b',
                terms=((1.0, 'specimen_b', 2), (-1.0, 'lever_b', 2)))


    def create_job(self, suffix='', run_name=None):
        """
        Creates an Abaqus job object (not submitted unless requested).
        """
        base = run_name if run_name else 'MMB_{0}_sample{1}'.format(self.mat_name, self.sample_number)
        job_name = str('{0}{1}'.format(base, suffix))
        mdb.saveAs(job_name)
        job = mdb.Job(
            name=job_name, model='test', description=job_name, type=ANALYSIS,
            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=60,
            memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
            modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
            scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=4, numDomains=4, numGPUs=0)
        return job_name, job


    def run_model_czm(self, submit_job=False):
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

        self.gen_equations()


        run_name_cfg = getattr(self, 'run_name', None)
        job_name, job = self.create_job(suffix='_CZM', run_name=run_name_cfg)
        if submit_job:
            job.submit(consistencyChecking=OFF)
            job.waitForCompletion()
        return job_name



TOL = 1e-3
DIR0 = os.path.abspath('')
try:
    _BASE_DIR = os.path.dirname(__file__)
except NameError:
    _BASE_DIR = os.getcwd()
CONFIG_PATH = os.path.abspath(sys.argv[sys.argv.index('--')+1]) if '--' in sys.argv else os.path.join(_BASE_DIR, 'mmb_config.json')


def build_mmb_from_config(config_path=CONFIG_PATH):
    """
    Build an MMB object from JSON config with minimal user touchpoints.
    """
    config = load_config(config_path)

    material_key = config.get('material_key', 'GFRP')
    specimen_index = config.get('specimen_index', 0)
    key_load = config.get('key_load', 'Displacement')
    mode = config.get('mode', 'CZM').upper()
    pseudo_2d = bool(config.get('Pseudo_2D', False))
    symmetry = bool(config.get('Symmetry', False))
    dynamic_implicit = bool(config.get('dynamic_implicit', False))
    density = float(config.get('density', 1.76e-9))
    run_name = config.get('run_name', None)

    if mode != 'CZM':
        raise ValueError('Only CZM mode is supported in this baseline; got "{0}".'.format(mode))

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
    cohesive_cfg = config.get('cohesive_standard', {})
    config_def_u2 = config.get('def_u2', None)
    number_inc = config.get('number_inc', 100000)
    num_attempts = config.get('num_attempts', 20)

    if not cohesive_cfg:
        raise ValueError('Cohesive config is required for CZM.')

    l = geometry_cfg.get('l', 200.0)
    L = geometry_cfg.get('L', 50.0)
    C = geometry_cfg.get('C', 46.0)
    loading_height = geometry_cfg.get('loading_height', 30.0)
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

    mmb = MMB(TOL, DIR0, l, t, b, a, l2, l1, L, C, loading_height, mesh_size, mesh_size_fine, def_u2, engineering_const, key_load, sample_number, material_key, cohesive_cfg, number_inc=number_inc, num_attempts=num_attempts, elements_across_thickness=elements_across_thickness, pseudo_2d=pseudo_2d, symmetry=symmetry, dynamic_implicit=dynamic_implicit, density=density)
    mmb.run_name = run_name
    return mmb, config, material_key, specimen_index


def summarize_choice(material_key, specimen_index, config, mmb=None):
    materials_cfg = config.get('materials', {})
    mat_cfg = materials_cfg[material_key]
    specimen = mat_cfg['specimens'][specimen_index]
    if mmb is not None:
        t_val = mmb.t
        b_val = mmb.b
        a_val = mmb.a
        l1_val = mmb.l1
        l2_val = mmb.l2
    else:
        t_val = specimen['t']
        b_val = specimen['b']
        a_val = specimen['a']
        l1_val = specimen['l1']
        l2_val = specimen['l2']
    summary = [
        'Material: {0}'.format(material_key),
        'Specimen index: {0} (sample {1})'.format(specimen_index, specimen.get('sample_number', specimen_index + 1)),
        'Geometry: t={0}, b={1}, a={2}, l1={3}, l2={4}'.format(t_val, b_val, a_val, l1_val, l2_val),
        'Load key: {0}, def_u2={1}'.format(config.get('key_load', 'Displacement'), specimen.get('def_u2', 0.0)),
        'Mode: CZM'
    ]
    for line in summary:
        print(line)


def run_from_config(config_path=CONFIG_PATH):
    mmb, config, material_key, specimen_index = build_mmb_from_config(config_path)
    summarize_choice(material_key, specimen_index, config, mmb=mmb)
    name = mmb.run_model_czm(submit_job=False)
    mdb.jobs[name].writeInput(consistencyChecking=OFF)
    return name


if __name__ == '__main__':
    run_from_config(CONFIG_PATH)

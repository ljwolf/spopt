import numpy as np

import pulp
from geopandas import GeoDataFrame

from spopt.locate.base import (
    BaseOutputMixin,
    CoveragePercentageMixin,
    BackupPercentageMixinMixin,
    LocateSolver,
    FacilityModelBuilder,
)
from scipy.spatial.distance import cdist

import warnings


class LSCP(LocateSolver, BaseOutputMixin):
    r"""
    Implement the Location Set Covering Problem (*LSCP*) optimization model
    and solve it. The *LSCP*, as adapted from :cite:`church_murray_2018`,
    can be formulated as:

    .. math::

       \begin{array}{lllll}
       \displaystyle \textbf{Minimize}      & \displaystyle \sum_{j=1}^{n}{Y_j}         &           & (1)                                                                           \\
       \displaystyle \textbf{Subject To}    & \displaystyle \sum_{j\in N_i}{Y_j} \geq 1 & \forall i & (2)                                                                           \\
                                            & Y_j \in {0,1}                             & \forall j & (3)                                                                           \\
                                            &                                           &           &                                                                               \\
       \displaystyle \textbf{Where}         &  i                                        &  =        & \textrm{index of demand locations}                                            \\
                                            &  j                                        &  =        & \textrm{index of facility sites}                                              \\
                                            &  S                                        &  =        & \textrm{maximum acceptable service distance or time standard}                 \\
                                            &  d_{ij}                                   &  =        & \textrm{shortest distance or travel time between nodes } i \textrm{ and } j   \\
                                            &  N_i                                      &  =        & \{j | d_{ij} < S\}                                                            \\
                                            &  Y_j                                      &  =        & \begin{cases}
                                                                                                        1, \text{if a facility is located at node } j                               \\
                                                                                                        0, \text{otherwise}                                                         \\
                                                                                                      \end{cases}
       \end{array}

    Parameters
    ----------

    name : str
        The problem name.
    problem : pulp.LpProblem
        A ``pulp`` instance of an optimization model that contains
        constraints, variables, and an objective function.

    Attributes
    ----------

    name : str
        The problem name.
    problem : pulp.LpProblem
        A ``pulp`` instance of an optimization model that contains
        constraints, variables, and an objective function.
    fac2cli : numpy.array
        A 2D array storing facility to client relationships where each
        row represents a facility and contains an array of client indices
        with which it is associated. An empty client array indicates
        the facility is associated with no clients.
    cli2fac : numpy.array
        The inverse of ``fac2cli`` where client to facility relationships
        are shown.
    aij : numpy.array
        A cost matrix in the form of a 2D array between origins and destinations.

    """  # noqa

    def __init__(self, name: str, problem: pulp.LpProblem):
        super().__init__(name, problem)

    def __add_obj(self) -> None:
        """
        Add objective function to model:

        Minimize y1 + y2 + ... + yj

        Returns
        -------

        None

        """
        fac_vars = getattr(self, "fac_vars")
        self.problem += pulp.lpSum(fac_vars), "objective function"

    @classmethod
    def from_cost_matrix(
        cls,
        cost_matrix: np.array,
        service_radius: float,
        predefined_facilities_arr: np.array = None,
        name: str = "LSCP",
    ):
        """
        Create an ``LSCP`` object based on a cost matrix.

        Parameters
        ----------

        cost_matrix : numpy.array
            A cost matrix in the form of a 2D array between origins and destinations.
        service_radius : float
            Maximum acceptable service distance.
        predefined_facilities_arr : numpy.array (default None)
            Predefined facilities that must appear in the solution.
        name : str, (default 'LSCP')
            The name of the problem.

        Returns
        -------

        spopt.locate.coverage.LSCP

        Examples
        --------

        >>> from spopt.locate import LSCP
        >>> from spopt.locate.util import simulated_geo_points
        >>> import geopandas
        >>> import numpy
        >>> import pulp
        >>> import spaghetti

        Create a regular lattice.

        >>> lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        >>> ntw = spaghetti.Network(in_data=lattice)
        >>> streets = spaghetti.element_as_gdf(ntw, arcs=True)
        >>> streets_buffered = geopandas.GeoDataFrame(
        ...     geopandas.GeoSeries(streets["geometry"].buffer(0.2).unary_union),
        ...     crs=streets.crs,
        ...     columns=["geometry"]
        ... )

        Simulate points about the lattice.

        >>> demand_points = simulated_geo_points(streets_buffered, needed=100, seed=5)
        >>> facility_points = simulated_geo_points(streets_buffered, needed=5, seed=6)

        Snap the points to the network of lattice edges.

        >>> ntw.snapobservations(demand_points, "clients", attribute=True)
        >>> clients_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="clients", snapped=True
        ... )
        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="facilities", snapped=True
        ... )

        Calculate the cost matrix from origins to destinations.

        >>> cost_matrix = ntw.allneighbordistances(
        ...    sourcepattern=ntw.pointpatterns["clients"],
        ...    destpattern=ntw.pointpatterns["facilities"])

        Create and solve an ``LSCP`` instance from the cost matrix.

        >>> lscp_from_cost_matrix = LSCP.from_cost_matrix(
        ...     cost_matrix, service_radius=8
        ... )
        >>> lscp_from_cost_matrix = lscp_from_cost_matrix.solve(
        ...     pulp.PULP_CBC_CMD(msg=False)
        ... )

        Get the facility lookup demand coverage array.

        >>> for fac, cli in enumerate(lscp_from_cost_matrix.fac2cli):
        ...     print(f"facility {fac} serving {len(cli)} clients")
        facility 0 serving 0 clients
        facility 1 serving 63 clients
        facility 2 serving 85 clients
        facility 3 serving 0 clients
        facility 4 serving 58 clients

        """

        r_fac = range(cost_matrix.shape[1])
        r_cli = range(cost_matrix.shape[0])

        model = pulp.LpProblem(name, pulp.LpMinimize)
        lscp = LSCP(name, model)

        FacilityModelBuilder.add_facility_integer_variable(lscp, r_fac, "y[{i}]")

        lscp.aij = np.zeros(cost_matrix.shape)
        lscp.aij[cost_matrix <= service_radius] = 1

        if predefined_facilities_arr is not None:
            FacilityModelBuilder.add_predefined_facility_constraint(
                lscp, lscp.problem, predefined_facilities_arr
            )

        lscp.__add_obj()
        FacilityModelBuilder.add_set_covering_constraint(
            lscp, lscp.problem, lscp.aij, r_fac, r_cli
        )

        return lscp

    @classmethod
    def from_geodataframe(
        cls,
        gdf_demand: GeoDataFrame,
        gdf_fac: GeoDataFrame,
        demand_col: str,
        facility_col: str,
        service_radius: float,
        predefined_facility_col: str = None,
        distance_metric: str = "euclidean",
        name: str = "LSCP",
    ):
        """
        Create an ``LSCP`` object from ``geopandas.GeoDataFrame`` objects.
        Calculate the cost matrix between demand and facility locations
        before building the problem within the ``from_cost_matrix()`` method.

        Parameters
        ----------

        gdf_demand : geopandas.GeoDataFrame
            Demand locations.
        gdf_fac : geopandas.GeoDataFrame
            Facility locations.
        demand_col : str
            Demand sites geometry column name.
        facility_col : str
            Facility candidate sites geometry column name.
        service_radius : float
             Maximum acceptable service distance.
        predefined_facility_col : str (default None)
            Column name representing facilities are already defined.
        distance_metric : str (default 'euclidean')
            A metric used for the distance calculations supported by
            `scipy.spatial.distance.cdist <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html>`_.
        name : str (default 'LSCP')
            The name of the problem.

        Returns
        -------

        spopt.locate.coverage.LSCP

        Examples
        --------

        >>> from spopt.locate import LSCP
        >>> from spopt.locate.util import simulated_geo_points
        >>> import geopandas
        >>> import numpy
        >>> import pulp
        >>> import spaghetti

        Create a regular lattice.

        >>> lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        >>> ntw = spaghetti.Network(in_data=lattice)
        >>> streets = spaghetti.element_as_gdf(ntw, arcs=True)
        >>> streets_buffered = geopandas.GeoDataFrame(
        ...     geopandas.GeoSeries(streets["geometry"].buffer(0.2).unary_union),
        ...     crs=streets.crs,
        ...     columns=["geometry"]
        ... )

        Simulate points about the lattice.

        >>> demand_points = simulated_geo_points(streets_buffered, needed=100, seed=5)
        >>> facility_points = simulated_geo_points(streets_buffered, needed=5, seed=6)

        Snap the points to the network of lattice edges
        and extract as ``GeoDataFrame`` objects.

        >>> ntw.snapobservations(demand_points, "clients", attribute=True)
        >>> clients_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="clients", snapped=True
        ... )
        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="facilities", snapped=True
        ... )

        Create and solve an ``LSCP`` instance from the ``GeoDataFrame`` objects.

        >>> lscp_from_geodataframe = LSCP.from_geodataframe(
        ...     clients_snapped,
        ...     facilities_snapped,
        ...     "geometry",
        ...     "geometry",
        ...     service_radius=8,
        ...     distance_metric="euclidean"
        ... )
        >>> lscp_from_geodataframe = lscp_from_geodataframe.solve(
        ...     pulp.PULP_CBC_CMD(msg=False)
        ... )

        Get the facility lookup demand coverage array.

        >>> for fac, cli in enumerate(lscp_from_geodataframe.fac2cli):
        ...     print(f"facility {fac} serving {len(cli)} clients")
        facility 0 serving 0 clients
        facility 1 serving 0 clients
        facility 2 serving 100 clients
        facility 3 serving 0 clients
        facility 4 serving 0 clients

        """  # noqa

        predefined_facilities_arr = None
        if predefined_facility_col is not None:
            predefined_facilities_arr = gdf_fac[predefined_facility_col].to_numpy()

        dem = gdf_demand[demand_col]
        fac = gdf_fac[facility_col]

        dem_type_geom = dem.geom_type.unique()
        fac_type_geom = fac.geom_type.unique()

        _msg = (
            " geodataframe contains mixed type geometries or is not a point. Be "
            "sure deriving centroid from geometries doesn't affect the results."
        )
        if len(dem_type_geom) > 1 or not "Point" in dem_type_geom:
            warnings.warn(f"Demand{_msg}", UserWarning)
            dem = dem.centroid

        if len(fac_type_geom) > 1 or not "Point" in fac_type_geom:
            warnings.warn(f"Facility{_msg}", UserWarning)
            fac = fac.centroid

        dem_data = np.array([dem.x.to_numpy(), dem.y.to_numpy()]).T
        fac_data = np.array([fac.x.to_numpy(), fac.y.to_numpy()]).T

        if gdf_demand.crs != gdf_fac.crs:
            raise ValueError(
                "Geodataframes crs are different: "
                f"gdf_demand-{gdf_demand.crs}, gdf_fac-{gdf_fac.crs}"
            )

        distances = cdist(dem_data, fac_data, distance_metric)

        return cls.from_cost_matrix(
            distances, service_radius, predefined_facilities_arr, name
        )

    def facility_client_array(self) -> None:
        """
        Create a 2D array storing **facility to client relationships** where each
        row represents a facility and contains an array of client indices
        with which it is associated. An empty client array indicates
        the facility is associated with no clients.

        Returns
        -------

        None

        """

        fac_vars = getattr(self, "fac_vars")
        len_fac_vars = len(fac_vars)

        self.fac2cli = []

        for j in range(len_fac_vars):
            array_cli = []
            if fac_vars[j].value() > 0:
                for i in range(self.aij.shape[0]):
                    if self.aij[i][j] > 0:
                        array_cli.append(i)

            self.fac2cli.append(array_cli)

    def solve(self, solver: pulp.LpSolver, results: bool = True):
        """
        Solve the ``LSCP`` model.

        Parameters
        ----------

        solver : pulp.apis.LpSolver
            A solver supported by ``pulp``.
        results : bool (default True)
            If ``True`` it will create metainfo (which facilities cover
            which demand) and vice-versa, and the uncovered demand.

        Returns
        -------

        spopt.locate.coverage.LSCP

        """
        self.problem.solve(solver)
        self.check_status()

        if results:
            self.facility_client_array()
            self.client_facility_array()

        return self


class LSCPB(LocateSolver, BaseOutputMixin, BackupPercentageMixinMixin):
    """
    LSCPB class implements Location Set Covering Problem - Backup
    optimization model and solves it.

    Parameters
    ----------

    name : str
        The problem name.
    problem : pulp.LpProblem
        A ``pulp`` instance of an optimization model that contains
        constraints, variables, and an objective function.
    solver : pulp.LpSolver
        A solver supported by ``pulp``.

    Attributes
    ----------

    name : str
        The problem name.
    problem : pulp.LpProblem
        A ``pulp`` instance of an optimization model that contains
        constraints, variables, and an objective function.
    solver : pulp.LpSolver
        A solver supported by ``pulp``.
    lscp_obj_value : float
        Objective value returned from solved LSCP instance.
    fac2cli : np.array
        2-d array MxN, where m is number of facilities and n is number of clients.
        Each row represents a facility and has an array containing clients index
        meaning that the facility-i cover the entire array.
    cli2fac : np.array
        2-d MxN, where m is number of clients and n is number of facilities.
        Each row represent a client and has an array containing facility index
        meaning that the client is covered by the facility ith.
    aij : np.array
        A cost matrix in the form of a 2D array between origins and destinations.

    """

    def __init__(
        self,
        name: str,
        problem: pulp.LpProblem,
        solver: pulp.LpSolver,
    ):
        self.solver = solver
        super().__init__(name, problem)

    def __add_obj(self) -> None:
        """
        Add objective function to model:
        (Coverage Variable)

        Maximize U1 + U2 + U3 + U4 + U5 + ... + Uj

        Returns
        -------

        None

        """
        cov_vars = getattr(self, "cli_vars")
        self.problem += pulp.lpSum(cov_vars), "objective function"

    @classmethod
    def from_cost_matrix(
        cls,
        cost_matrix: np.array,
        service_radius: float,
        solver: pulp.LpSolver,
        predefined_facilities_arr: np.array = None,
        name: str = "LSCP-B",
    ):
        """
        Create an ``LSCPB`` object based on a cost matrix.

        Parameters
        ----------

        cost_matrix : np.array
            A cost matrix in the form of a 2D array between origins and destinations.
        service_radius : float
            maximum acceptable service distance by problem
        solver : pulp.LpSolver
            A solver supported by ``pulp``.
        predefined_facilities_arr : numpy.array
            Predefined facilities that must appear in the solution.
            Default is ``None``.
        name : str, default="LSCP-B"
            name of the problem

        Returns
        -------

        spopt.locate.LSCPB

        Examples
        --------

        >>> from spopt.locate import LSCPB
        >>> from spopt.locate.util import simulated_geo_points
        >>> import geopandas
        >>> import numpy
        >>> import pulp
        >>> import spaghetti

        Create a regular lattice.

        >>> lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        >>> ntw = spaghetti.Network(in_data=lattice)
        >>> streets = spaghetti.element_as_gdf(ntw, arcs=True)
        >>> streets_buffered = geopandas.GeoDataFrame(
        ...     geopandas.GeoSeries(streets["geometry"].buffer(0.2).unary_union),
        ...     crs=streets.crs,
        ...     columns=["geometry"]
        ... )

        Simulate points about the lattice.

        >>> demand_points = simulated_geo_points(streets_buffered, needed=100, seed=5)
        >>> facility_points = simulated_geo_points(streets_buffered, needed=5, seed=6)

        Snap the points to the network of lattice edges.

        >>> ntw.snapobservations(demand_points, "clients", attribute=True)
        >>> clients_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="clients", snapped=True
        ... )
        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="facilities", snapped=True
        ... )

        Calculate the cost matrix from origins to destinations.

        >>> cost_matrix = ntw.allneighbordistances(
        ...     sourcepattern=ntw.pointpatterns["clients"],
        ...     destpattern=ntw.pointpatterns["facilities"]
        ... )

        Create and solve an ``LSCPB`` instance from the cost matrix.

        >>> lscpb_from_cost_matrix = LSCPB.from_cost_matrix(
        ...     cost_matrix, service_radius=8, solver=pulp.PULP_CBC_CMD(msg=False)
        ... )
        >>> lscpb_from_cost_matrix = lscpb_from_cost_matrix.solve()

        Get the facility lookup demand coverage array.

        >>> for fac, cli in enumerate(lscpb_from_cost_matrix.fac2cli):
        ...     print(f"facility {fac} serving {len(cli)} clients")
        facility 0 serving 0 clients
        facility 1 serving 63 clients
        facility 2 serving 85 clients
        facility 3 serving 92 clients
        facility 4 serving 0 clients

        Get the percentage of clients covered by more than one facility.

        >>> round(lscpb_from_cost_matrix.backup_perc, 3)
        88.0

        88% of clients are covered by more than 1 facility

        """
        if predefined_facilities_arr is not None:
            lscp = LSCP.from_cost_matrix(
                cost_matrix, service_radius, predefined_facilities_arr
            )
        else:
            lscp = LSCP.from_cost_matrix(cost_matrix, service_radius)
        lscp.solve(solver)

        r_fac = range(cost_matrix.shape[1])
        r_cli = range(cost_matrix.shape[0])

        model = pulp.LpProblem(name, pulp.LpMaximize)

        lscpb = LSCPB(name, model, solver)
        lscpb.lscp_obj_value = lscp.problem.objective.value()

        FacilityModelBuilder.add_facility_integer_variable(lscpb, r_fac, "y[{i}]")
        FacilityModelBuilder.add_client_integer_variable(lscpb, r_cli, "x[{i}]")

        lscpb.aij = np.zeros(cost_matrix.shape)
        lscpb.aij[cost_matrix <= service_radius] = 1

        if predefined_facilities_arr is not None:
            FacilityModelBuilder.add_predefined_facility_constraint(
                lscpb, lscpb.problem, predefined_facilities_arr
            )

        lscpb.__add_obj()
        FacilityModelBuilder.add_facility_constraint(
            lscpb, lscpb.problem, lscpb.lscp_obj_value
        )
        FacilityModelBuilder.add_backup_covering_constraint(
            lscpb, lscpb.problem, lscpb.aij, r_fac, r_cli
        )

        return lscpb

    @classmethod
    def from_geodataframe(
        cls,
        gdf_demand: GeoDataFrame,
        gdf_fac: GeoDataFrame,
        demand_col: str,
        facility_col: str,
        service_radius: float,
        solver: pulp.LpSolver,
        predefined_facility_col: str = None,
        distance_metric: str = "euclidean",
        name: str = "LSCP-B",
    ):
        """
        Create a LSCPB object based on geodataframes. Calculate the cost matrix
        between demand and facility, and then use from_cost_matrix method.

        Parameters
        ----------

        gdf_demand : geopandas.GeoDataFrame
            demand geodataframe with point geometry
        gdf_fac : geopandas.GeoDataframe
            facility geodataframe with point geometry
        demand_col : str
            demand geometry column name
        facility_col : str
            facility candidate sites geometry column name
        service_radius : float
            maximum acceptable service distance by problem
        solver : pulp.LpSolver
            A solver supported by ``pulp``.
        predefined_facility_col : str
            Column name representing facilities are already defined.
            Default is ``None``.
        distance_metric : str, default="euclidean"
            metrics supported by :method: `scipy.spatial.distance.cdist` used for
            the distance calculations
        name : str, default="LSCP"
            name of the problem

        Returns
        -------

        spopt.locate.LSCPB

        Examples
        --------

        >>> from spopt.locate import LSCPB
        >>> from spopt.locate.util import simulated_geo_points
        >>> import geopandas
        >>> import numpy
        >>> import pulp
        >>> import spaghetti

        Create a regular lattice.

        >>> lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        >>> ntw = spaghetti.Network(in_data=lattice)
        >>> streets = spaghetti.element_as_gdf(ntw, arcs=True)
        >>> streets_buffered = geopandas.GeoDataFrame(
        ...     geopandas.GeoSeries(streets["geometry"].buffer(0.2).unary_union),
        ...     crs=streets.crs,
        ...     columns=["geometry"]
        ... )

        Simulate points about the lattice.

        >>> demand_points = simulated_geo_points(streets_buffered, needed=100, seed=5)
        >>> facility_points = simulated_geo_points(streets_buffered, needed=5, seed=6)

        Snap the points to the network of lattice edges
        and extract as ``GeoDataFrame`` objects.

        >>> ntw.snapobservations(demand_points, "clients", attribute=True)
        >>> clients_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="clients", snapped=True
        ... )
        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="facilities", snapped=True
        ... )

        Create and solve an ``LSCPB`` instance from the ``GeoDataFrame`` objects.

        >>> lscpb_from_geodataframe = LSCPB.from_geodataframe(
        ...     clients_snapped,
        ...     facilities_snapped,
        ...     "geometry",
        ...     "geometry",
        ...     service_radius=8,
        ...     solver=pulp.PULP_CBC_CMD(msg=False),
        ...     distance_metric="euclidean"
        ... )
        >>> lscpb_from_geodataframe = lscpb_from_geodataframe.solve()

        Get the facility lookup demand coverage array.

        >>> for fac, cli in enumerate(lscpb_from_geodataframe.fac2cli):
        ...     print(f"facility {fac} serving {len(cli)} clients")
        facility 0 serving 0 clients
        facility 1 serving 0 clients
        facility 2 serving 100 clients
        facility 3 serving 0 clients
        facility 4 serving 0 clients

        Get the percentage of clients covered by more than one facility.

        >>> round(lscpb_from_geodataframe.backup_perc, 3)
        0.0

        All clients are covered by 1 facility because only one facility
        is needed to solve the LSCP.

        """

        predefined_facilities_arr = None
        if predefined_facility_col is not None:
            predefined_facilities_arr = gdf_fac[predefined_facility_col].to_numpy()

        dem = gdf_demand[demand_col]
        fac = gdf_fac[facility_col]

        dem_type_geom = dem.geom_type.unique()
        fac_type_geom = fac.geom_type.unique()

        _msg = (
            " geodataframe contains mixed type geometries or is not a point. Be "
            "sure deriving centroid from geometries doesn't affect the results."
        )
        if len(dem_type_geom) > 1 or not "Point" in dem_type_geom:
            warnings.warn(f"Demand{_msg}", UserWarning)
            dem = dem.centroid

        if len(fac_type_geom) > 1 or not "Point" in fac_type_geom:
            warnings.warn(f"Facility{_msg}", UserWarning)
            fac = fac.centroid

        dem_data = np.array([dem.x.to_numpy(), dem.y.to_numpy()]).T
        fac_data = np.array([fac.x.to_numpy(), fac.y.to_numpy()]).T

        if gdf_demand.crs != gdf_fac.crs:
            raise ValueError(
                "Geodataframes crs are different: "
                f"gdf_demand-{gdf_demand.crs}, gdf_fac-{gdf_fac.crs}"
            )

        distances = cdist(dem_data, fac_data, distance_metric)

        return cls.from_cost_matrix(
            distances, service_radius, solver, predefined_facilities_arr, name
        )

    def facility_client_array(self) -> None:
        """

        Create a 2D :math:`m \times n` array, where :math:`m` is number of
        facilities and :math:`n` is number of clients. Each row represent a
        facility and has an array containing a clients indices.

        Returns
        -------

        None

        """

        fac_vars = getattr(self, "fac_vars")
        len_fac_vars = len(fac_vars)

        self.fac2cli = []

        for j in range(len_fac_vars):
            array_cli = []
            if fac_vars[j].value() > 0:
                for i in range(self.aij.shape[0]):
                    if self.aij[i][j] > 0:
                        array_cli.append(i)

            self.fac2cli.append(array_cli)

    def solve(self, results: bool = True):
        """
        Solve the ``LSCPB`` model.

        Parameters
        ----------

        results : bool
            If ``True`` it will create metainfo (which facilities cover
            which demand) and vice-versa, and the uncovered demand.

        Returns
        -------

        spopt.locate.LSCPB

        """
        self.problem.solve(self.solver)
        self.check_status()

        if results:
            self.facility_client_array()
            self.client_facility_array()
            self.get_percentage()

        return self


class MCLP(LocateSolver, BaseOutputMixin, CoveragePercentageMixin):
    """
    MCLP class implements Maximal Coverage Location optimization model and solve it.

    Parameters
    ----------

    name : str
        The problem name.
    problem : pulp.LpProblem
        A ``pulp`` instance of an optimization model that contains
        constraints, variables, and an objective function.

    Attributes
    ----------

    name : str
        Problem name
    problem : pulp.LpProblem
        Pulp instance of optimization model that contains constraints,
        variables and objective function.
    fac2cli : np.array
        2-d array MxN, where m is number of facilities and n is number of clients.
        Each row represents a facility and has an array containing clients index
        meaning that the facility-i cover the entire array.
    cli2fac : np.array
        2-d MxN, where m is number of clients and n is number of facilities.
        Each row represent a client and has an array containing facility index
        meaning that the client is covered by the facility ith.
    aij : np.array
        A cost matrix in the form of a 2D array between origins and destinations.
    n_cli_uncov : int
        Specify how many clients points are not covered.

    """

    def __init__(self, name: str, problem: pulp.LpProblem):
        super().__init__(name, problem)

    def __add_obj(self, weights: np.array, range_clients: range) -> None:
        """
        Add objective function to model:

        Maximize w1 * y1 + w2 * y2 +  ... + wi * yi

        Returns
        -------

        None

        """
        dem_vars = getattr(self, "cli_vars")

        self.problem += (
            pulp.lpSum([weights.flatten()[i] * dem_vars[i] for i in range_clients]),
            "objective function",
        )

    @classmethod
    def from_cost_matrix(
        cls,
        cost_matrix: np.array,
        weights: np.array,
        service_radius: float,
        p_facilities: int,
        predefined_facilities_arr: np.array = None,
        name: str = "MCLP",
    ):
        """
        Create a ``MCLP`` object based on cost matrix.

        Parameters
        ----------

        cost_matrix : np.array
            A cost matrix in the form of a 2D array between origins and destinations.
        weights : np.array
            one-dimensional service load or population demand
        service_radius : float
            maximum acceptable service distance by problem
        p_facilities : int
            number of facilities to be located
        predefined_facilities_arr : numpy.array
            Predefined facilities that must appear in the solution.
            Default is ``None``.
        name : str, default="MCLP"
            name of the problem

        Returns
        -------

        spopt.locate.MCLP

        Examples
        --------

        >>> from spopt.locate import MCLP
        >>> from spopt.locate.util import simulated_geo_points
        >>> import geopandas
        >>> import numpy
        >>> import pulp
        >>> import spaghetti

        Create a regular lattice.

        >>> lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        >>> ntw = spaghetti.Network(in_data=lattice)
        >>> streets = spaghetti.element_as_gdf(ntw, arcs=True)
        >>> streets_buffered = geopandas.GeoDataFrame(
        ...     geopandas.GeoSeries(streets["geometry"].buffer(0.2).unary_union),
        ...     crs=streets.crs,
        ...     columns=["geometry"]
        ... )

        Simulate points about the lattice.

        >>> demand_points = simulated_geo_points(streets_buffered, needed=100, seed=5)
        >>> facility_points = simulated_geo_points(streets_buffered, needed=5, seed=6)

        Snap the points to the network of lattice edges.

        >>> ntw.snapobservations(demand_points, "clients", attribute=True)
        >>> clients_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="clients", snapped=True
        ... )
        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="facilities", snapped=True
        ... )

        Calculate the cost matrix from origins to destinations.

        >>> cost_matrix = ntw.allneighbordistances(
        ...    sourcepattern=ntw.pointpatterns["clients"],
        ...    destpattern=ntw.pointpatterns["facilities"]
        ... )

        Simulate demand weights from ``1`` to ``12``.

        >>> ai = numpy.random.randint(1, 12, 100)

        Create and solve an ``MCLP`` instance from the cost matrix.

        >>> mclp_from_cost_matrix = MCLP.from_cost_matrix(
        ...     cost_matrix, ai, service_radius=7, p_facilities=4
        ... )
        >>> mclp_from_cost_matrix = mclp_from_cost_matrix.solve(
        ...     pulp.PULP_CBC_CMD(msg=False)
        ... )

        Get the facility lookup demand coverage array.

        >>> for fac, cli in enumerate(mclp_from_cost_matrix.fac2cli):
        ...     print(f"facility {fac} serving {len(cli)} clients")
        facility 0 serving 44 clients
        facility 1 serving 54 clients
        facility 2 serving 75 clients
        facility 3 serving 77 clients
        facility 4 serving 0 clients

        Get the number of clients uncovered and percentage covered.

        >>> mclp_from_cost_matrix.n_cli_uncov
        1
        >>> mclp_from_cost_matrix.perc_cov
        99.0

        """
        r_fac = range(cost_matrix.shape[1])
        r_cli = range(cost_matrix.shape[0])

        model = pulp.LpProblem(name, pulp.LpMaximize)
        mclp = MCLP(name, model)

        FacilityModelBuilder.add_facility_integer_variable(mclp, r_fac, "x[{i}]")
        FacilityModelBuilder.add_client_integer_variable(mclp, r_cli, "y[{i}]")

        mclp.aij = np.zeros(cost_matrix.shape)
        mclp.aij[cost_matrix <= service_radius] = 1
        weights = np.reshape(weights, (cost_matrix.shape[0], 1))

        mclp.__add_obj(weights, r_cli)

        if predefined_facilities_arr is not None:
            FacilityModelBuilder.add_predefined_facility_constraint(
                mclp, mclp.problem, predefined_facilities_arr
            )

        FacilityModelBuilder.add_maximal_coverage_constraint(
            mclp, mclp.problem, mclp.aij, r_fac, r_cli
        )

        FacilityModelBuilder.add_facility_constraint(mclp, mclp.problem, p_facilities)

        return mclp

    @classmethod
    def from_geodataframe(
        cls,
        gdf_demand: GeoDataFrame,
        gdf_fac: GeoDataFrame,
        demand_col: str,
        facility_col: str,
        weights_cols: str,
        service_radius: float,
        p_facilities: int,
        predefined_facility_col: str = None,
        distance_metric: str = "euclidean",
        name: str = "MCLP",
    ):
        """
        Create a MCLP object based on geodataframes. Calculate the cost matrix
        between demand and facility, and then use from_cost_matrix method.

        Parameters
        ----------

        gdf_demand : geopandas.GeoDataFrame
            demand geodataframe with point geometry
        gdf_fac : geopandas.GeoDataframe
            facility geodataframe with point geometry
        demand_col : str
            demand geometry column name
        facility_col : str
            facility candidate sites geometry column name
        weights_cols : str
            weight column name representing service load or demand
        service_radius : float
            maximum acceptable service distance by problem
        p_facilities : int
            number of facilities to be located
        predefined_facility_col : str
            Column name representing facilities are already defined.
            Default is ``None``.
        distance_metric : str, default="euclidean"
            metrics supported by :method: `scipy.spatial.distance.cdist` used for
            the distance calculations
        name : str, default="MCLP"
            name of the problem

        Returns
        -------

        spopt.locate.MCLP

        Examples
        --------

        >>> from spopt.locate import MCLP
        >>> from spopt.locate.util import simulated_geo_points
        >>> import geopandas
        >>> import pulp
        >>> import numpy
        >>> import spaghetti

        Create a regular lattice.

        >>> lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        >>> ntw = spaghetti.Network(in_data=lattice)
        >>> streets = spaghetti.element_as_gdf(ntw, arcs=True)
        >>> streets_buffered = geopandas.GeoDataFrame(
        ...     geopandas.GeoSeries(streets["geometry"].buffer(0.2).unary_union),
        ...     crs=streets.crs,
        ...     columns=["geometry"]
        ... )

        Simulate points about the lattice.

        >>> demand_points = simulated_geo_points(streets_buffered, needed=100, seed=5)
        >>> facility_points = simulated_geo_points(streets_buffered, needed=5, seed=6)

        Snap the points to the network of lattice edges
        and extract as ``GeoDataFrame`` objects.

        >>> ntw.snapobservations(demand_points, "clients", attribute=True)
        >>> clients_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="clients", snapped=True
        ... )
        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="facilities", snapped=True
        ... )

        Simulate demand weights from ``1`` to ``12``.

        >>> ai = numpy.random.randint(1, 12, 100)
        >>> clients_snapped['weights'] = ai

        Create and solve an ``MCLP`` instance from the ``GeoDataFrame`` objects.

        >>> mclp_from_geodataframe = MCLP.from_geodataframe(
        ...     clients_snapped,
        ...     facilities_snapped,
        ...     "geometry",
        ...     "geometry",
        ...     "weights",
        ...     service_radius=7,
        ...     p_facilities=4,
        ...     distance_metric="euclidean"
        ... )

        >>> mclp_from_geodataframe = mclp_from_geodataframe.solve(
        ...     pulp.PULP_CBC_CMD(msg=False)
        ... )

        Get the facility lookup demand coverage array.

        >>> for fac, cli in enumerate(mclp_from_geodataframe.fac2cli):
        ...     print(f"facility {fac} serving {len(cli)} clients")
        facility 0 serving 63 clients
        facility 1 serving 80 clients
        facility 2 serving 96 clients
        facility 3 serving 0 clients
        facility 4 serving 58 clients

        Get the number of clients uncovered and percentage covered.

        >>> mclp_from_geodataframe.n_cli_uncov
        0
        >>> mclp_from_geodataframe.perc_cov
        100.0

        """

        predefined_facilities_arr = None
        if predefined_facility_col is not None:
            predefined_facilities_arr = gdf_fac[predefined_facility_col].to_numpy()

        service_load = gdf_demand[weights_cols].to_numpy()
        dem = gdf_demand[demand_col]
        fac = gdf_fac[facility_col]

        dem_type_geom = dem.geom_type.unique()
        fac_type_geom = fac.geom_type.unique()

        _msg = (
            " geodataframe contains mixed type geometries or is not a point. Be "
            "sure deriving centroid from geometries doesn't affect the results."
        )
        if len(dem_type_geom) > 1 or not "Point" in dem_type_geom:
            warnings.warn(f"Demand{_msg}", UserWarning)
            dem = dem.centroid

        if len(fac_type_geom) > 1 or not "Point" in fac_type_geom:
            warnings.warn(f"Facility{_msg}", UserWarning)
            fac = fac.centroid

        dem_data = np.array([dem.x.to_numpy(), dem.y.to_numpy()]).T
        fac_data = np.array([fac.x.to_numpy(), fac.y.to_numpy()]).T

        if gdf_demand.crs != gdf_fac.crs:
            raise ValueError(
                "Geodataframes crs are different: "
                f"gdf_demand-{gdf_demand.crs}, gdf_fac-{gdf_fac.crs}"
            )

        distances = cdist(dem_data, fac_data, distance_metric)

        return cls.from_cost_matrix(
            distances,
            service_load,
            service_radius,
            p_facilities,
            predefined_facilities_arr,
            name,
        )

    def facility_client_array(self) -> None:
        """

        Create a 2D :math:`m \times n` array, where :math:`m` is number of
        facilities and :math:`n` is number of clients. Each row represent a
        facility and has an array containing a clients indices.

        Returns
        -------

        None

        """

        fac_vars = getattr(self, "fac_vars")
        cli_vars = getattr(self, "cli_vars")
        len_fac_vars = len(fac_vars)

        self.fac2cli = []

        for j in range(len_fac_vars):
            array_cli = []
            if fac_vars[j].value() > 0:
                for i in range(self.aij.shape[0]):
                    if cli_vars[i].value() > 0:
                        if self.aij[i][j] > 0:
                            array_cli.append(i)

            self.fac2cli.append(array_cli)

    def solve(self, solver: pulp.LpSolver, results: bool = True):
        """
        Solve the ``MCLP`` model

        Parameters
        ----------

        solver : pulp.LpSolver
            A solver supported by ``pulp``.
        results : bool
            If ``True`` it will create metainfo (which facilities cover
            which demand) and vice-versa, and the uncovered demand.

        Returns
        -------

        spopt.locate.MCLP

        """
        self.problem.solve(solver)
        self.check_status()

        if results:
            self.facility_client_array()
            self.client_facility_array()
            self.uncovered_clients()
            self.get_percentage()
        return self

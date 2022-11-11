import numpy as np

import pulp
from geopandas import GeoDataFrame

from spopt.locate.base import LocateSolver, FacilityModelBuilder
from scipy.spatial.distance import cdist

import warnings


class PDispersion(LocateSolver):
    """
    PDispersion class implements the p-dispersion optimization model and solves it.

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

    """

    def __init__(self, name: str, problem: pulp.LpProblem, p_facilities: int):
        self.p_facilities = p_facilities
        super().__init__(name, problem)

    def __add_obj(self) -> None:
        """
        Add objective function to model:

        Maximize D

        Returns
        -------

        None

        """
        disperse = getattr(self, "disperse_var")

        self.problem += disperse, "objective function"

    @classmethod
    def from_cost_matrix(
        cls,
        cost_matrix: np.array,
        p_fac: int,
        predefined_facilities_arr: np.array = None,
        name: str = "P-Dispersion",
    ):
        """
        Create a ``PDispersion`` object based on a cost matrix.

        Parameters
        ----------

        cost_matrix: np.array
            A cost matrix in the form of a 2D array between origins and destinations.
        p_fac: int
            number of facilities to be located
        predefined_facilities_arr : numpy.array
            Predefined facilities that must appear in the solution.
            Default is ``None``.
        name: str, default="P-Dispersion"
            name of the problem

        Returns
        -------

        spopt.locate.PDispersion

        Examples
        --------

        >>> from spopt.locate.p_dispersion import PDispersion
        >>> from spopt.locate.util import simulated_geo_points
        >>> import geopandas
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

        >>> facility_points = simulated_geo_points(streets_buffered, needed=5, seed=6)

        Snap the points to the network of lattice edges.

        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="facilities", snapped=True
        ... )

        Calculate the cost matrix from origins to destinations. Origins
        and destinations are both ``'facilities'`` in this case.

        >>> cost_matrix = ntw.allneighbordistances(
        ...    sourcepattern=ntw.pointpatterns["facilities"],
        ...    destpattern=ntw.pointpatterns["facilities"]
        ... )

        Create and solve a ``PDispersion`` instance from the cost matrix.

        >>> pdispersion_from_cost_matrix = PDispersion.from_cost_matrix(
        ...     cost_matrix, p_fac=2
        ... )
        >>> pdispersion_from_cost_matrix = pdispersion_from_cost_matrix.solve(
        ...     pulp.PULP_CBC_CMD(msg=False)
        ... )

        Examine the solution.

        >>> for dv in pdispersion_from_cost_matrix.fac_vars:
        ...     if dv.varValue:
        ...         print(f"facility {dv.name} is selected")
        facility y_0_ is selected
        facility y_1_ is selected

        """

        r_fac = range(cost_matrix.shape[1])

        model = pulp.LpProblem(name, pulp.LpMaximize)
        p_dispersion = PDispersion(name, model, p_fac)

        FacilityModelBuilder.add_maximized_min_variable(p_dispersion)
        p_dispersion.__add_obj()

        FacilityModelBuilder.add_facility_integer_variable(
            p_dispersion, r_fac, "y[{i}]"
        )

        FacilityModelBuilder.add_facility_constraint(
            p_dispersion, p_dispersion.problem, p_dispersion.p_facilities
        )

        if predefined_facilities_arr is not None:
            FacilityModelBuilder.add_predefined_facility_constraint(
                p_dispersion, p_dispersion.problem, predefined_facilities_arr
            )

        FacilityModelBuilder.add_p_dispersion_interfacility_constraint(
            p_dispersion, p_dispersion.problem, cost_matrix, r_fac
        )

        return p_dispersion

    @classmethod
    def from_geodataframe(
        cls,
        gdf_fac: GeoDataFrame,
        facility_col: str,
        p_fac: int,
        predefined_facility_col: str = None,
        distance_metric: str = "euclidean",
        name: str = "P-Dispersion",
    ):
        """
        Create a ``PDispersion`` object based on a geodataframe. Calculate the
        cost matrix between facilities, and then use the from_cost_matrix method.

        Parameters
        ----------

        gdf_fac: geopandas.GeoDataframe
            facility geodataframe with point geometry
        facility_col: str
            facility candidate sites geometry column name
        p_fac: int
            number of facilities to be located
        predefined_facility_col: str
            Column name representing facilities are already defined.
            Default is ``None``.
        distance_metric: str, default="euclidean"
            metrics supported by :method: `scipy.spatial.distance.cdist`
            used for the distance calculations
        name: str, default="P-Dispersion"
            name of the problem

        Returns
        -------

        spopt.locate.PDispersion

        Examples
        --------

        >>> from spopt.locate.p_dispersion import PDispersion
        >>> from spopt.locate.util import simulated_geo_points
        >>> import geopandas
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

        >>> facility_points = simulated_geo_points(streets_buffered, needed=5, seed=6)

        Snap the points to the network of lattice edges
        and extract as a ``GeoDataFrame`` object.

        >>> ntw.snapobservations(facility_points, "facilities", attribute=True)
        >>> facilities_snapped = spaghetti.element_as_gdf(
        ...     ntw, pp_name="facilities", snapped=True
        ... )

        Create and solve a ``PDispersion`` instance from the ``GeoDataFrame`` object.

        >>> pdispersion_from_geodataframe = PDispersion.from_geodataframe(
        ...     facilities_snapped,
        ...     "geometry",
        ...     p_fac=2,
        ...     distance_metric="euclidean"
        ... )
        >>> pdispersion_from_geodataframe = pdispersion_from_geodataframe.solve(
        ...     pulp.PULP_CBC_CMD(msg=False)
        ... )

        Examine the solution.

        >>> for dv in pdispersion_from_geodataframe.fac_vars:
        ...     if dv.varValue:
        ...         print(f"facility {dv.name} is selected")
        facility y_0_ is selected
        facility y_1_ is selected

        """

        predefined_facilities_arr = None
        if predefined_facility_col is not None:
            predefined_facilities_arr = gdf_fac[predefined_facility_col].to_numpy()

        fac = gdf_fac[facility_col]

        fac_type_geom = fac.geom_type.unique()

        if len(fac_type_geom) > 1 or not "Point" in fac_type_geom:
            warnings.warn(
                (
                    "Facility geodataframe contains mixed type geometries "
                    "or is not a point. Be sure deriving centroid from "
                    "geometries doesn't affect the results."
                ),
                UserWarning,
            )
            fac = fac.centroid

        fac_data = np.array([fac.x.to_numpy(), fac.y.to_numpy()]).T

        distances = cdist(fac_data, fac_data, distance_metric)

        return cls.from_cost_matrix(distances, p_fac, predefined_facilities_arr, name)

    def solve(self, solver: pulp.LpSolver, results: bool = True):
        """
        Solve the ``PDispersion`` model.

        Parameters
        ----------

        solver : pulp.LpSolver
            A solver supported by ``pulp``.
        results : bool
            If ``True`` it will create metainfo (which facilities cover
            which demand) and vice-versa, and the uncovered demand.

        Returns
        -------

        spopt.locate.PDispersion

        """
        self.problem.solve(solver)
        self.check_status()

        return self

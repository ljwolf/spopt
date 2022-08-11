from pyproj import crs
from spopt.locate.base import FacilityModelBuilder, LocateSolver, T_FacModel
import numpy
import geopandas
import pandas
import pulp
import spaghetti
from shapely.geometry import Point, Polygon

from spopt.locate import PDispersion
from spopt.locate.util import simulated_geo_points
import unittest
import os
import pickle
import platform

operating_system = platform.platform()[:7].lower()
if operating_system == "windows":
    WINDOWS = True
else:
    WINDOWS = False


class TestSyntheticLocate(unittest.TestCase):
    def setUp(self) -> None:
        self.dirpath = os.path.join(os.path.dirname(__file__), "./data/")

        lattice = spaghetti.regular_lattice((0, 0, 10, 10), 9, exterior=True)
        ntw = spaghetti.Network(in_data=lattice)
        gdf = spaghetti.element_as_gdf(ntw, arcs=True)
        street = geopandas.GeoDataFrame(
            geopandas.GeoSeries(gdf["geometry"].buffer(0.2).unary_union),
            crs=gdf.crs,
            columns=["geometry"],
        )

        facility_count = 5

        self.facility_points = simulated_geo_points(
            street, needed=facility_count, seed=6
        )

        ntw = spaghetti.Network(in_data=lattice)

        ntw.snapobservations(self.facility_points, "facilities", attribute=True)

        self.facilities_snapped = spaghetti.element_as_gdf(
            ntw, pp_name="facilities", snapped=True
        )

        self.cost_matrix = ntw.allneighbordistances(
            sourcepattern=ntw.pointpatterns["facilities"],
            destpattern=ntw.pointpatterns["facilities"],
        )


    def test_p_dispersion_from_cost_matrix(self):
        pdispersion = PDispersion.from_cost_matrix(self.cost_matrix, p_facilities=2)
        result = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        self.assertIsInstance(result, PDispersion)

    def test_p_dispersion_facility_client_array_from_cost_matrix(self):
        with open(self.dirpath + "pdispersion_fac2cli.pkl", "rb") as f:
            pdispersion_objective = pickle.load(f)

        pdispersion = PDispersion.from_cost_matrix(self.cost_matrix, p_facilities=4)
        pdispersion = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        pdispersion.facility_client_array()

        numpy.testing.assert_array_equal(pdispersion.fac2cli, pdispersion_objective)

    def test_p_dispersion_client_facility_array_from_cost_matrix(self):
        with open(self.dirpath + "pdispersion_cli2fac.pkl", "rb") as f:
            pdispersion_objective = pickle.load(f)

        pdispersion = PDispersion.from_cost_matrix(self.cost_matrix, p_facilities=4)
        pdispersion = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        pdispersion.facility_client_array()
        pdispersion.client_facility_array()

        numpy.testing.assert_array_equal(pdispersion.cli2fac, pdispersion_objective)

    def test_p_dispersion_from_geodataframe(self):
        pdispersion = PDispersion.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            p_facilities=4,
        )
        result = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        self.assertIsInstance(result, PDispersion)

    def test_p_dispersion_facility_client_array_from_geodataframe(self):
        with open(self.dirpath + "pdispersion_geodataframe_fac2cli.pkl", "rb") as f:
            pdispersion_objective = pickle.load(f)

        pdispersion = PDispersion.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            p_facilities=4,
        )
        pdispersion = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        pdispersion.facility_client_array()

        numpy.testing.assert_array_equal(pdispersion.fac2cli, pdispersion_objective)

    def test_p_dispersion_client_facility_array_from_geodataframe(self):
        with open(self.dirpath + "pdispersion_geodataframe_cli2fac.pkl", "rb") as f:
            pdispersion_objective = pickle.load(f)

        pdispersion = PDispersion.from_geodataframe(
            self.clients_snapped,
            self.facilities_snapped,
            "geometry",
            "geometry",
            p_facilities=4,
        )
        pdispersion = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        pdispersion.facility_client_array()
        pdispersion.client_facility_array()

        numpy.testing.assert_array_equal(pdispersion.cli2fac, pdispersion_objective)


class TestRealWorldLocate(unittest.TestCase):
    def setUp(self) -> None:
        self.dirpath = os.path.join(os.path.dirname(__file__), "./data/")
        network_distance = pandas.read_csv(
            self.dirpath
            + "SF_network_distance_candidateStore_16_censusTract_205_new.csv"
        )

        ntw_dist_piv = network_distance.pivot_table(
            values="distance", index="DestinationName", columns="name"
        )

        self.cost_matrix = ntw_dist_piv.to_numpy()

        demand_points = pandas.read_csv(
            self.dirpath + "SF_demand_205_centroid_uniform_weight.csv"
        )
        facility_points = pandas.read_csv(self.dirpath + "SF_store_site_16_longlat.csv")

        self.facility_points_gdf = (
            geopandas.GeoDataFrame(
                facility_points,
                geometry=geopandas.points_from_xy(
                    facility_points.long, facility_points.lat
                ),
            )
            .sort_values(by=["NAME"])
            .reset_index()
        )

        self.demand_points_gdf = (
            geopandas.GeoDataFrame(
                demand_points,
                geometry=geopandas.points_from_xy(
                    demand_points.long, demand_points.lat
                ),
            )
            .sort_values(by=["NAME"])
            .reset_index()
        )

        self.service_dist = 5000.0
        self.p_facility = 4
        self.ai = self.demand_points_gdf["POP2000"].to_numpy()



    def test_optimality_p_dispersion_from_cost_matrix(self):
        pdispersion = PDispersion.from_cost_matrix(
            self.cost_matrix, p_facilities=self.p_facility
        )
        pdispersion = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        self.assertEqual(pdispersion.problem.status, pulp.LpStatusOptimal)

    def test_infeasibility_p_dispersion_from_cost_matrix(self):
        pdispersion = PDispersion.from_cost_matrix(self.cost_matrix, p_facilities=0)
        with self.assertRaises(RuntimeError):
            pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))

    def test_optimality_p_dispersion_from_geodataframe(self):
        pdispersion = PDispersion.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            p_facilities=self.p_facility,
        )
        pdispersion = pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))
        self.assertEqual(pdispersion.problem.status, pulp.LpStatusOptimal)

    def test_infeasibility_p_dispersion_from_geodataframe(self):
        pdispersion = PDispersion.from_geodataframe(
            self.demand_points_gdf,
            self.facility_points_gdf,
            "geometry",
            "geometry",
            p_facilities=0,
        )
        with self.assertRaises(RuntimeError):
            pdispersion.solve(pulp.PULP_CBC_CMD(msg=False))

class TestErrorsWarnings(unittest.TestCase):
    def setUp(self) -> None:

        pol1 = Polygon([(0, 0), (1, 0), (1, 1)])
        pol2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        pol3 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        polygon_dict = {"geometry": [pol1, pol2, pol3]}

        point = Point(10, 10)
        point_dict = {"weight": 4, "geometry": [point]}

        self.gdf_fac = geopandas.GeoDataFrame(polygon_dict, crs="EPSG:4326")
        self.gdf_dem = geopandas.GeoDataFrame(point_dict, crs="EPSG:4326")

        self.gdf_dem_crs = self.gdf_dem.to_crs("EPSG:3857")

        self.gdf_dem_buffered = self.gdf_dem.copy()
        self.gdf_dem_buffered["geometry"] = self.gdf_dem.buffer(2)

 
    def test_error_p_dispersion_different_crs(self):
        with self.assertRaises(ValueError):
            dummy_class = PDispersion.from_geodataframe(
                self.gdf_dem_crs, self.gdf_fac, "geometry", "geometry", 2
            )

 
    def test_warning_p_dispersion_facility_geodataframe(self):
        with self.assertWarns(Warning):
            dummy_class = PDispersion.from_geodataframe(
                self.gdf_dem, self.gdf_fac, "geometry", "geometry", 2
            )

    def test_warning_p_dispersion_demand_geodataframe(self):
        with self.assertWarns(Warning):
            dummy_class = PDispersion.from_geodataframe(
                self.gdf_dem_buffered, self.gdf_fac, "geometry", "geometry", 2
            )

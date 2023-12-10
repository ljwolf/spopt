import numpy
import geopandas
import pandas
import pulp
from shapely.geometry import Point

from spopt.locate.p_median import KNearestPMedian
from spopt.locate.base import SpecificationError
import os
import pickle
import platform
import pytest
import warnings


class TestKNearestPMedian:
    def setup_method(self) -> None:
        # Create the test data
        k = numpy.array([1, 1])
        self.demand_data = {
            "ID": [1, 2],
            "geometry": [Point(0.5, 1), Point(1.5, 1)],
            "demand": [1, 1],
        }
        self.facility_data = {
            "ID": [101, 102, 103],
            "geometry": [Point(1, 1), Point(0, 2), Point(2, 0)],
            "capacity": [1, 1, 1],
        }
        self.gdf_demand = geopandas.GeoDataFrame(self.demand_data, crs="EPSG:4326")
        self.gdf_fac = geopandas.GeoDataFrame(self.facility_data, crs="EPSG:4326")
        self.k_nearest_pmedian = KNearestPMedian.from_geodataframe(
            self.gdf_demand,
            self.gdf_fac,
            "geometry",
            "geometry",
            "demand",
            p_facilities=2,
            facility_capacity_col="capacity",
            k_array=k,
        )
        self.solver = pulp.PULP_CBC_CMD(msg=False)

    def test_knearest_p_median_from_geodataframe(self):
        result = self.k_nearest_pmedian.solve(self.solver)
        assert isinstance(result, KNearestPMedian)

    def test_knearest_p_median_from_geodataframe_no_results(self):
        result = self.k_nearest_pmedian.solve(self.solver, results=False)
        assert isinstance(result, KNearestPMedian)

        with pytest.raises(AttributeError):
            result.cli2fac
        with pytest.raises(AttributeError):
            result.fac2cli
        with pytest.raises(AttributeError):
            result.mean_dist

    def test_solve(self):
        self.k_nearest_pmedian.solve(self.solver)
        assert self.k_nearest_pmedian.problem.status == pulp.LpStatusOptimal

        fac2cli_known = [[1], [0], []]
        cli2fac_known = [[1], [0]]
        mean_dist_known = 0.8090169943749475
        assert self.k_nearest_pmedian.fac2cli == fac2cli_known
        assert self.k_nearest_pmedian.cli2fac == cli2fac_known
        assert self.k_nearest_pmedian.mean_dist == mean_dist_known

    def test_error_overflow_k(self):
        k = numpy.array([10, 10])
        with pytest.raises(ValueError, match="The value of k should be"):
            KNearestPMedian.from_geodataframe(
                self.gdf_demand,
                self.gdf_fac,
                "geometry",
                "geometry",
                "demand",
                p_facilities=2,
                facility_capacity_col="capacity",
                k_array=k,
            )

    def test_error_k_array_non_numpy_array(self):
        k = [1, 1]
        with pytest.raises(TypeError, match="k_array should be a numpy array."):
            KNearestPMedian.from_geodataframe(
                self.gdf_demand,
                self.gdf_fac,
                "geometry",
                "geometry",
                "demand",
                p_facilities=2,
                facility_capacity_col="capacity",
                k_array=k,
            )

    def test_error_k_array_invalid_value(self):
        k = numpy.array([1, 4])
        with pytest.raises(ValueError, match="The value of k should be no more"):
            KNearestPMedian.from_geodataframe(
                self.gdf_demand,
                self.gdf_fac,
                "geometry",
                "geometry",
                "demand",
                p_facilities=2,
                facility_capacity_col="capacity",
                k_array=k,
            )

    def test_error_no_crs_demand(self):
        _gdf_demand = geopandas.GeoDataFrame(self.demand_data)
        k = numpy.array([1, 1])
        with pytest.raises(ValueError, match="GeoDataFrame gdf_demand "):
            KNearestPMedian.from_geodataframe(
                _gdf_demand,
                self.gdf_fac,
                "geometry",
                "geometry",
                "demand",
                p_facilities=2,
                facility_capacity_col="capacity",
                k_array=k,
            )

    def test_error_no_crs_facility(self):
        _gdf_fac = geopandas.GeoDataFrame(self.facility_data)
        k = numpy.array([1, 1])
        with pytest.raises(ValueError, match="GeoDataFrame gdf_facility "):
            KNearestPMedian.from_geodataframe(
                self.gdf_demand,
                _gdf_fac,
                "geometry",
                "geometry",
                "demand",
                p_facilities=2,
                facility_capacity_col="capacity",
                k_array=k,
            )

    def test_error_geodataframe_crs_mismatch(self):
        _gdf_fac = self.gdf_fac.copy().to_crs("EPSG:3857")
        k = numpy.array([1, 1])
        with pytest.raises(ValueError, match="Geodataframes crs are different"):
            KNearestPMedian.from_geodataframe(
                self.gdf_demand,
                _gdf_fac,
                "geometry",
                "geometry",
                "demand",
                p_facilities=2,
                facility_capacity_col="capacity",
                k_array=k,
            )

    def test_error_high_capacity(self):
        _gdf_demand = self.gdf_demand.copy()
        _gdf_demand["demand"] = [10, 10]
        k = numpy.array([1, 1])
        with pytest.raises(
            SpecificationError,
            match="Problem is infeasible. The highest possible capacity",
        ):
            KNearestPMedian.from_geodataframe(
                _gdf_demand,
                self.gdf_fac,
                "geometry",
                "geometry",
                "demand",
                p_facilities=1,
                facility_capacity_col="capacity",
                k_array=k,
            ).solve(self.solver)

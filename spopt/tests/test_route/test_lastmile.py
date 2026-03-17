import geopandas as gpd
import pandas, numpy, pyvrp, sys
from pathlib import Path
import spopt
from spopt.route import engine, heuristic, utils
from spopt.route.heuristic import LastMile
from pyvrp.stop import MaxIterations
import platform
import pytest
from importlib.metadata import version as pkg_version


def test_lastmile_route():
    trucks = pandas.DataFrame(
        [['big', 'lng',      2000,    280, .004,  .50, 5],
        ['big', 'electric', 2000,    480, .002,  .50, 5],
        ['med', 'lng',      800, 280*.66, .0001, .63, 10],
        ['med', 'electric', 800, 480*.66, .004,  .50, 10],
        ['smo', 'lng',      400, 280*0.4, .002,  .50, 20],
        ['smo', 'electric', 400, 480*0.4, .0001, .63, 20],
        ],
        columns = [
                'namesize', 'namefuel', 'capacity', 
                'fixed_cost', 'cost_per_meter', 'cost_per_minute', 'n_truck'
                ]
    )
    DATA_DIR = Path(__file__).resolve().parent / "data"
    gdf = gpd.read_file(DATA_DIR / "dublin-pubs.geojson")
    clients = gdf.iloc[1:,:].reset_index(drop=True)
    clients = clients.set_index(clients.osmid.astype(str))
    depot = gdf.iloc[0,:]

    print('initializing model')
    m = LastMile(
        depot_location=(depot.longitude.item(), depot.latitude.item()),
        depot_open=pandas.Timestamp("2030-01-02 07:00:00"),
        depot_close=pandas.Timestamp("2030-01-02 20:00:00"),
        depot_name=depot['name'],
    )

    print("adding clients")
    m.add_clients(
        locations = clients.geometry, 
        delivery = clients.demand,
        pickup = clients.supply,
        time_windows=None,
        service_times=(numpy.log(clients.demand)**2).astype(int)
    )

    print("adding trucks")
    m.add_trucks_from_frame(
        trucks, 
    )
    
    print("python:", sys.version)
    print("platform:", platform.platform())
    print("pyvrp:", pyvrp.show_versions())
    
    m.solve(stop=MaxIterations(100), seed=1234, display=False)

    assert m.result_.is_feasible()

    pyvrp_version = pkg_version("pyvrp")
    print("pyvrp version string:", pyvrp_version)

    if pyvrp_version == "0.13.2":
        # Baseline for Python 3.14 / PyVRP 0.13.2
        expected_solution = 53_541_709
    else:
        # Baseline for older PyVRP versions
        expected_solution = 174_502_935

    assert m.result_.cost() == pytest.approx(expected_solution, rel=0.01)

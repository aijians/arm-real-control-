import math
from pathlib import Path
from unittest.mock import patch

from control.config import load_config
from control.planner import Planner


ROOT = Path(__file__).resolve().parents[1]


def test_reverse_blue_job_uses_fixed_joint_waypoints_and_reports_clearance():
    config, _ = load_config(ROOT)
    for motor, offset in zip(config["hardware"]["motors"], (-2.32, 2.8)):
        motor["offset_deg"] = offset  # Offsets from this recorded task.
    planner = Planner(ROOT, config)
    start = list(map(math.radians, (158.14, -0.52)))
    sliders = {"蓝": -3.821042942823965, "橙": 10.706531224527424}
    goal = planner.precontact("蓝", "-x", sliders, start)
    phase = planner.free(start, goal, sliders, "-x")
    assert phase["route"] == "joint_left_entry"
    assert phase["time_parameterization"] == "joint_trapezoid"
    assert phase["points"][-1]["q_rad"] == goal
    first, second = phase["joint_waypoints_rad"][1:3]
    assert first[1] == start[1]
    assert first[0] == second[0]
    assert abs(math.degrees(second[1])-34.95) < 1e-9
    assert 0 < phase["minimum_clearance_mm"] < 1.


def test_route_selection_compares_joint_clearances_before_cartesian_fallback():
    planner = Planner(ROOT, load_config(ROOT)[0])
    port = list(map(math.radians, (175., 4.)))
    goal = list(map(math.radians, (165., 18.)))
    sliders = {"蓝": -47., "橙": 11.}
    margins = {"joint_linear": .1, "joint_passage": .2}
    with patch.object(planner.scene, "clearance_mm", return_value=.3), patch.object(
            planner, "_free_candidate", side_effect=lambda name, *args, **kw: {
                "route": name, "minimum_clearance_mm": margins[name]}) as plan:
        phase = planner.free(port, goal, sliders)
    assert plan.call_count == 2
    assert phase["route"] == "joint_passage"
    assert phase["minimum_clearance_mm"] == .2

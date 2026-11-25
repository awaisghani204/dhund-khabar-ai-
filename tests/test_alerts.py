"""Tests for src.alerts."""

import numpy as np

from src.alerts import aqi_to_tier_index, get_aqi_tier

# Independent expectation tables, hardcoded on purpose.
_UPPERS = [50.0, 100.0, 150.0, 200.0, 300.0]
_NAMES = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous (Emergency)",
]
_COLORS = ["#00E400", "#FFFF00", "#FF7E00", "#FF0000", "#8F3F97", "#7E0023"]


def _expected_index(aqi: float) -> int:
    """Tier with lower < aqi <= upper; Good includes 0; above 500 is Hazardous."""
    for i, upper in enumerate(_UPPERS):
        if aqi <= upper:
            return i
    return 5


def test_property_10_epa_tier_mapping_by_decimal_boundaries():
    """Feature: dhund-khabar-ai, Property 10: EPA tier mapping by decimal boundaries

    **Validates: Requirements 12.1, 12.2, 12.9, 18.8**
    """
    rng = np.random.default_rng(42)
    boundaries = [0, 50, 100, 150, 200, 300, 500]
    values = list(rng.uniform(0, 700, 300))
    for b in boundaries:
        values += [b, b + 0.01, b + 0.0001]
        if b > 0:
            values += [b - 0.01, b - 0.0001]
    values += [200.5, 50.0001, 300.0001, 500.0001, 538.0, 600.0, 700.0]
    assert len(values) >= 100

    for v in values:
        idx = _expected_index(float(v))
        name, color = get_aqi_tier(float(v))
        assert name == _NAMES[idx], v
        assert color == _COLORS[idx], v

    arr = np.array(values, dtype=float)
    vec = aqi_to_tier_index(arr)
    assert vec.shape == arr.shape
    assert [int(x) for x in vec] == [_expected_index(float(v)) for v in values]
    for v, i in zip(values, vec):
        assert _NAMES[int(i)] == get_aqi_tier(float(v))[0]

    # Monotonic non-decreasing in AQI.
    order = np.argsort(arr, kind="stable")
    assert np.all(np.diff(vec[order]) >= 0)

    # Above 500 is Hazardous (Emergency).
    for v in (500.0001, 501, 538, 600, 700, 10000):
        assert get_aqi_tier(v) == ("Hazardous (Emergency)", "#7E0023")

    # Fixed list from the design.
    fixed = [0, 50, 51, 100, 101, 150, 151, 200, 201, 300, 301, 500, 600]
    expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5]
    assert [int(x) for x in aqi_to_tier_index(np.array(fixed))] == expected
    assert [_NAMES.index(get_aqi_tier(v)[0]) for v in fixed] == expected


_WARNINGS = [
    "Air quality is satisfactory. Ideal for all outdoor sports and physical activities.",
    "Acceptable air quality. Unusually sensitive individuals should monitor outdoor exertion.",
    "Children, elderly, and asthmatic patients should wear masks outdoors and avoid strenuous exercise.",
    "Everyone may begin to experience health effects. N95 masks strongly advised. Public outdoor events should be minimized.",
    "Health alert: serious risk of respiratory symptoms across the population. Outdoor school sports banned; high-risk individuals stay indoors.",
    "Emergency conditions: entire population affected. Primary schools and non-essential outdoor work should close; heavy air purifier use required.",
]
_ADVISORY_KEYS = {
    "tier", "color", "warnings", "school_status",
    "respiratory_guidance", "industrial_advisory",
}
_CURFEW = "Brick Kiln & Heavy Vehicle Curfew Required"


def test_property_11_advisory_fields_follow_thresholds():
    """Feature: dhund-khabar-ai, Property 11: Advisory fields follow the thresholds

    **Validates: Requirements 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 11.7**
    """
    from src.alerts import get_aqi_advisory
    from src.model import SmogPredictorModel

    rng = np.random.default_rng(42)
    values = [float(v) for v in rng.uniform(0, 700, 200)]
    for b in [0, 50, 100, 150, 200, 300, 500]:
        values += [float(b), b + 0.01]
        if b > 0:
            values += [b - 0.01]
    values += [200.5, 300.0001, 600.0, 700.0]
    assert len(values) >= 100

    model = SmogPredictorModel()
    guidance_by_tier: dict[str, str] = {}
    for v in values:
        idx = _expected_index(v)
        adv = get_aqi_advisory(v)
        assert set(adv.keys()) == _ADVISORY_KEYS, v
        assert adv["tier"] == _NAMES[idx], v
        assert adv["color"] == _COLORS[idx], v
        assert adv["warnings"] == _WARNINGS[idx], v

        if v <= 100:
            school = "Normal"
        elif v <= 300:
            school = "Outdoor Restrictions"
        else:
            school = "Closure Recommended"
        assert adv["school_status"] == school, v

        if v > 200:
            assert adv["industrial_advisory"] == _CURFEW, v
            assert _NAMES[idx] in ("Very Unhealthy", "Hazardous (Emergency)")
        else:
            assert _CURFEW not in adv["industrial_advisory"], v
            assert adv["industrial_advisory"].strip(), v

        guidance = adv["respiratory_guidance"]
        assert isinstance(guidance, str) and guidance.strip(), v
        assert guidance_by_tier.setdefault(_NAMES[idx], guidance) == guidance

        assert model.get_risk_advisory(v) == adv, v

    # Guidance is tier-specific: each of the six tiers has distinct text.
    assert len(guidance_by_tier) == 6
    assert len(set(guidance_by_tier.values())) == 6


def test_property_12_invalid_aqi_inputs_are_rejected():
    """Feature: dhund-khabar-ai, Property 12: Invalid AQI inputs are rejected

    **Validates: Requirements 12.10, 12.11, 18.9**
    """
    import math

    import pytest

    from src.alerts import get_aqi_advisory, validate_aqi
    from src.model import SmogPredictorModel

    rng = np.random.default_rng(42)
    negatives = [float(-(10.0 ** rng.uniform(-9, 12))) for _ in range(150)]
    negatives += [float(-v) for v in rng.uniform(0, 600, 50) if v > 0]
    negatives += [-1e-9, -0.0001, -1.0, -1e300, -float("inf"), -sys_max()]
    assert len(negatives) >= 100

    model = SmogPredictorModel()
    scalar_fns = [validate_aqi, get_aqi_tier, get_aqi_advisory,
                  aqi_to_tier_index, model.get_risk_advisory]

    def rejects(fn, value, msg):
        with pytest.raises(ValueError, match=msg):
            fn(value)

    for v in negatives:
        assert v < 0 or math.isinf(v)
        for fn in scalar_fns:
            rejects(fn, v, "AQI cannot be negative")
        # Arrays: one bad element among valid ones.
        valid = [float(x) for x in rng.uniform(0, 600, 4)]
        arr = np.array(valid[:2] + [v] + valid[2:])
        rejects(aqi_to_tier_index, arr, "AQI cannot be negative")
        rejects(aqi_to_tier_index, arr.tolist(), "AQI cannot be negative")

    # NaN, scalar, numpy scalar, and inside arrays.
    for nan in (float("nan"), np.nan, np.float64("nan"), np.float32("nan")):
        for fn in scalar_fns:
            rejects(fn, nan, "AQI is NaN")
    for pos in range(5):
        arr = np.array([10.0, 60.0, 120.0, 250.0, 400.0])
        arr[pos] = np.nan
        rejects(aqi_to_tier_index, arr, "AQI is NaN")
    rejects(aqi_to_tier_index, np.array([np.nan, -5.0]), "AQI is NaN")

    # +inf is rejected per the source.
    for fn in scalar_fns:
        rejects(fn, float("inf"), "AQI is infinite")
    rejects(aqi_to_tier_index, np.array([10.0, np.inf]), "AQI is infinite")

    # Zero and negative zero are valid and Good.
    for z in (0, 0.0, -0.0, np.float64(-0.0)):
        assert get_aqi_tier(z) == ("Good", "#00E400")
        assert get_aqi_advisory(z)["tier"] == "Good"
        assert int(aqi_to_tier_index(z)) == 0
        assert validate_aqi(z) == 0.0
    assert list(aqi_to_tier_index(np.array([0.0, -0.0]))) == [0, 0]

    # Non-numeric inputs raise TypeError and never return a tier.
    for bad in (None, "50", "abc", "", True, False, np.bool_(True), 1 + 2j, [1.0]):
        for fn in (validate_aqi, get_aqi_tier, get_aqi_advisory,
                   model.get_risk_advisory):
            with pytest.raises(TypeError):
                fn(bad)
    for bad in (None, "50", True, [True, False], [None, 1.0], ["a", "b"], 1 + 2j):
        with pytest.raises(TypeError):
            aqi_to_tier_index(bad)


def sys_max() -> float:
    """Largest finite float, as a negative-extreme test value."""
    return float(np.finfo(float).max)


# ---------------------------------------------------------------------------
# Task 13.21: example tests with literal expected values (Req 12, 18.8)
# ---------------------------------------------------------------------------
import pytest

from src.alerts import get_aqi_advisory

_GOOD = ("Good", "#00E400")
_MOD = ("Moderate", "#FFFF00")
_USG = ("Unhealthy for Sensitive Groups", "#FF7E00")
_UNH = ("Unhealthy", "#FF0000")
_VUN = ("Very Unhealthy", "#8F3F97")
_HAZ = ("Hazardous (Emergency)", "#7E0023")


@pytest.mark.parametrize(
    "aqi, tier, color",
    [
        (0, *_GOOD),
        (50, *_GOOD),
        (51, *_MOD),
        (100, *_MOD),
        (101, *_USG),
        (150, *_USG),
        (151, *_UNH),
        (200, *_UNH),
        (201, *_VUN),
        (300, *_VUN),
        (301, *_HAZ),
        (500, *_HAZ),
        (600, *_HAZ),
        (538, *_HAZ),
    ],
)
def test_example_tier_and_color(aqi, tier, color):
    assert get_aqi_tier(aqi) == (tier, color)
    adv = get_aqi_advisory(aqi)
    assert adv["tier"] == tier
    assert adv["color"] == color


@pytest.mark.parametrize(
    "aqi, text",
    [
        (25, "Air quality is satisfactory. Ideal for all outdoor sports and physical activities."),
        (75, "Acceptable air quality. Unusually sensitive individuals should monitor outdoor exertion."),
        (125, "Children, elderly, and asthmatic patients should wear masks outdoors and avoid strenuous exercise."),
        (175, "Everyone may begin to experience health effects. N95 masks strongly advised. Public outdoor events should be minimized."),
        (250, "Health alert: serious risk of respiratory symptoms across the population. Outdoor school sports banned; high-risk individuals stay indoors."),
        (400, "Emergency conditions: entire population affected. Primary schools and non-essential outdoor work should close; heavy air purifier use required."),
    ],
)
def test_example_warning_texts_verbatim(aqi, text):
    assert get_aqi_advisory(aqi)["warnings"] == text


def test_example_curfew_follows_tier_boundary():
    at_200 = get_aqi_advisory(200)
    assert at_200["tier"] == "Unhealthy"
    assert "Brick Kiln & Heavy Vehicle Curfew Required" not in at_200["industrial_advisory"]
    assert at_200["industrial_advisory"].strip()

    at_200_5 = get_aqi_advisory(200.5)
    assert at_200_5["tier"] == "Very Unhealthy"
    assert at_200_5["industrial_advisory"] == "Brick Kiln & Heavy Vehicle Curfew Required"


@pytest.mark.parametrize(
    "aqi, status",
    [
        (100, "Normal"),
        (101, "Outdoor Restrictions"),
        (300, "Outdoor Restrictions"),
        (301, "Closure Recommended"),
    ],
)
def test_example_school_status(aqi, status):
    assert get_aqi_advisory(aqi)["school_status"] == status


def test_example_dataset_maximum_is_hazardous():
    adv = get_aqi_advisory(538)
    assert adv["tier"] == "Hazardous (Emergency)"
    assert adv["color"] == "#7E0023"


@pytest.mark.parametrize("bad", [-1, float("nan")])
def test_example_invalid_aqi_raises_value_error(bad):
    with pytest.raises(ValueError):
        get_aqi_advisory(bad)

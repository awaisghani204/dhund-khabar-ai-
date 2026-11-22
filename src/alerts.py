"""US EPA AQI tiers and the fixed advisory texts for Dhund-Khabar AI.

Every function here is a plain function, as Requirement 3.2 asks. The module
imports numpy only, so the model, the charts, the training script, and the
dashboard can all use it without pulling in pandas or scikit-learn.

Tier rule: a value belongs to the tier whose range satisfies
``lower < aqi <= upper``, with Good starting at 0 inclusive. So 50 is Good,
50.4 is Moderate, 200 is Unhealthy, and 200.5 is Very Unhealthy. Values above
500 (the top of the EPA scale) are Hazardous (Emergency).
"""

import numpy as np

# One dict per tier, ordered from best to worst, so the position of a tier in
# this list is its tier index. The warning texts are copied verbatim from
# Requirement 12.3 and must not be edited.
TIERS: list[dict] = [
    {
        "name": "Good",
        "lower": 0.0,
        "upper": 50.0,
        "color": "#00E400",
        "warning": (
            "Air quality is satisfactory. Ideal for all outdoor sports and "
            "physical activities."
        ),
        "respiratory_guidance": "No respiratory protection needed.",
    },
    {
        "name": "Moderate",
        "lower": 50.0,
        "upper": 100.0,
        "color": "#FFFF00",
        "warning": (
            "Acceptable air quality. Unusually sensitive individuals should "
            "monitor outdoor exertion."
        ),
        "respiratory_guidance": (
            "Unusually sensitive individuals should monitor breathing during "
            "outdoor exertion."
        ),
    },
    {
        "name": "Unhealthy for Sensitive Groups",
        "lower": 100.0,
        "upper": 150.0,
        "color": "#FF7E00",
        "warning": (
            "Children, elderly, and asthmatic patients should wear masks "
            "outdoors and avoid strenuous exercise."
        ),
        "respiratory_guidance": (
            "Children, elderly, and asthmatic patients should wear masks outdoors."
        ),
    },
    {
        "name": "Unhealthy",
        "lower": 150.0,
        "upper": 200.0,
        "color": "#FF0000",
        "warning": (
            "Everyone may begin to experience health effects. N95 masks strongly "
            "advised. Public outdoor events should be minimized."
        ),
        "respiratory_guidance": "Everyone should wear an N95 mask outdoors.",
    },
    {
        "name": "Very Unhealthy",
        "lower": 200.0,
        "upper": 300.0,
        "color": "#8F3F97",
        "warning": (
            "Health alert: serious risk of respiratory symptoms across the "
            "population. Outdoor school sports banned; high-risk individuals "
            "stay indoors."
        ),
        "respiratory_guidance": (
            "Wear an N95 mask outdoors; high-risk individuals should stay indoors."
        ),
    },
    {
        "name": "Hazardous (Emergency)",
        "lower": 300.0,
        # The EPA scale stops at 500, but the data reaches 538. An open upper
        # bound keeps those readings in the worst tier instead of dropping them.
        "upper": float("inf"),
        "color": "#7E0023",
        "warning": (
            "Emergency conditions: entire population affected. Primary schools "
            "and non-essential outdoor work should close; heavy air purifier use "
            "required."
        ),
        "respiratory_guidance": "Stay indoors and run air purifiers continuously.",
    },
]

# Built from TIERS so the searchsorted boundaries and the tier table can never
# disagree. The open-ended Hazardous tier needs no bound: searchsorted returns
# len(bounds) = 5 for anything above 300.
TIER_UPPER_BOUNDS: np.ndarray = np.array(
    [tier["upper"] for tier in TIERS[:-1]], dtype=float
)

SCHOOL_NORMAL: str = "Normal"
SCHOOL_RESTRICTED: str = "Outdoor Restrictions"
SCHOOL_CLOSURE: str = "Closure Recommended"
CURFEW_TEXT: str = "Brick Kiln & Heavy Vehicle Curfew Required"
NO_CURFEW_TEXT: str = "No Curfew Required: normal industrial and traffic operations"
# The curfew starts exactly where Very Unhealthy starts, so no AQI value can be
# Very Unhealthy or Hazardous (Emergency) without a curfew.
CURFEW_THRESHOLD: float = TIERS[4]["lower"]


def _check_values(values: np.ndarray) -> None:
    """Raise a ValueError if any AQI value is NaN, negative, or +infinity.

    Args:
        values: Float array of AQI values (any shape, including 0-d).

    Returns:
        None. The function only raises.

    Raises:
        ValueError: "AQI is NaN", "AQI cannot be negative", or "AQI is infinite".
    """
    # NaN is checked first because NaN < 0 is False and would otherwise slip
    # through the negative check. -inf counts as negative, which keeps the
    # message for every negative input the same.
    if np.isnan(values).any():
        raise ValueError("AQI is NaN")
    if (values < 0).any():
        raise ValueError("AQI cannot be negative")
    # A +inf AQI can only come from a bug upstream (for example a division by
    # zero), so it is rejected rather than shown as a real Hazardous reading.
    if np.isposinf(values).any():
        raise ValueError("AQI is infinite")


def validate_aqi(aqi: float) -> float:
    """Check that a single AQI value is a real, finite, non-negative number.

    Args:
        aqi: One AQI value. Python int or float and NumPy integer or floating
            scalars are accepted. Booleans are rejected.

    Returns:
        The value converted to a Python float.

    Raises:
        TypeError: If aqi is not a real number (for example a string, None,
            a bool, a complex number, or an array).
        ValueError: "AQI is NaN" for NaN, "AQI cannot be negative" for values
            below 0, and "AQI is infinite" for +inf.
    """
    # bool is a subclass of int, and None would silently become NaN in NumPy,
    # so both are rejected by type before any numeric check runs.
    if isinstance(aqi, (bool, np.bool_)) or not isinstance(
        aqi, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"AQI must be a real number, got {type(aqi).__name__}")
    value = float(aqi)
    _check_values(np.asarray(value))
    return value


def aqi_to_tier_index(aqi_values: np.ndarray | float) -> np.ndarray:
    """Map AQI values to tier indices 0 (Good) to 5 (Hazardous (Emergency)).

    Uses ``np.searchsorted(TIER_UPPER_BOUNDS, aqi, side="left")``. With
    side="left", a value equal to an upper bound returns that bound's position,
    so it stays in the lower tier, and any value above the bound moves to the
    next tier. This gives the rule lower < aqi <= upper with Good starting at 0.

    Args:
        aqi_values: A single AQI value or an array-like of AQI values.

    Returns:
        An integer array with the same shape as the input (0-d for a scalar).

    Raises:
        TypeError: If the values are not real numbers (strings, None, bools,
            complex numbers, or mixed object arrays).
        ValueError: If any value is NaN, negative, or +inf, with the same
            messages as validate_aqi.
    """
    raw = np.asarray(aqi_values)
    # Only signed int, unsigned int, and float dtypes are real numbers. Bool
    # and object arrays are rejected, because NumPy would turn True into 1 and
    # None into NaN without complaint.
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"AQI values must be real numbers, got dtype {raw.dtype}")
    values = raw.astype(float)
    _check_values(values)
    return np.searchsorted(TIER_UPPER_BOUNDS, values, side="left").astype(int)


def get_aqi_tier(aqi: float) -> tuple[str, str]:
    """Return the EPA tier name and colour hex for one AQI value.

    Args:
        aqi: One AQI value. Values above 500 map to Hazardous (Emergency).

    Returns:
        A (tier name, colour hex) tuple, for example ("Good", "#00E400").

    Raises:
        TypeError: If aqi is not a real number.
        ValueError: If aqi is NaN, negative, or +inf.
    """
    tier = TIERS[int(aqi_to_tier_index(validate_aqi(aqi)))]
    return tier["name"], tier["color"]


def get_school_status(aqi: float) -> str:
    """Return the school status for one AQI value.

    The thresholds match the tier boundaries: Normal covers Good and Moderate,
    Outdoor Restrictions covers Unhealthy for Sensitive Groups through Very
    Unhealthy, and Closure Recommended is exactly Hazardous (Emergency).

    Args:
        aqi: One AQI value.

    Returns:
        "Normal" if aqi <= 100, "Outdoor Restrictions" if 100 < aqi <= 300,
        and "Closure Recommended" if aqi > 300.

    Raises:
        TypeError: If aqi is not a real number.
        ValueError: If aqi is NaN, negative, or +inf.
    """
    value = validate_aqi(aqi)
    if value <= 100:
        return SCHOOL_NORMAL
    if value <= 300:
        return SCHOOL_RESTRICTED
    return SCHOOL_CLOSURE


def get_industrial_advisory(aqi: float) -> str:
    """Return the industrial advisory for one AQI value.

    The curfew applies when aqi > 200, which is exactly when the tier is Very
    Unhealthy or Hazardous (Emergency). So 200 gets no curfew and 200.5 does.

    Args:
        aqi: One AQI value.

    Returns:
        "Brick Kiln & Heavy Vehicle Curfew Required" if aqi > 200, otherwise
        "No Curfew Required: normal industrial and traffic operations".

    Raises:
        TypeError: If aqi is not a real number.
        ValueError: If aqi is NaN, negative, or +inf.
    """
    value = validate_aqi(aqi)
    return CURFEW_TEXT if value > CURFEW_THRESHOLD else NO_CURFEW_TEXT


def get_aqi_advisory(aqi: float) -> dict[str, str]:
    """Build the full advisory for one AQI value.

    Args:
        aqi: One AQI value. Values above 500 map to Hazardous (Emergency).

    Returns:
        A dict with exactly the keys tier, color, warnings, school_status,
        respiratory_guidance, and industrial_advisory. warnings holds the
        verbatim warning text of the tier.

    Raises:
        TypeError: If aqi is not a real number.
        ValueError: If aqi is NaN, negative, or +inf.
    """
    value = validate_aqi(aqi)
    tier = TIERS[int(aqi_to_tier_index(value))]
    return {
        "tier": tier["name"],
        "color": tier["color"],
        "warnings": tier["warning"],
        "school_status": get_school_status(value),
        "respiratory_guidance": tier["respiratory_guidance"],
        "industrial_advisory": get_industrial_advisory(value),
    }


from datetime import datetime
from pathlib import Path
import math

import numpy as np
from herbie import Herbie


class WeatherProvider:
    """
    Retrieves and caches NOAA GFS weather fields.

    Public interface:
        provider.get_weather(latitude, longitude, altitude_m)

    Wind components are in m/s.
    Temperature is in K.
    Pressure is in Pa.
    Density is in kg/m^3.

    One provider holds one forecast hour. Interpolation in time between
    forecast hours and selection of the GFS cycle are done by the
    module-level load_weather() and get_weather() below.
    """

    def __init__(
        self,
        initialization=None,
        forecast_hour=0,
        pressure_levels=None,
        model="gfs",
        product="pgrb2.0p25",
        region_center=None,
        region_half_width_deg=None,
    ):
        self.initialization = initialization or datetime(2026, 10, 12, 12)
        self.forecast_hour = forecast_hour
        self.pressure_levels = pressure_levels or [
            1000, 925, 850, 700, 600, 500,
            400, 300, 250, 200, 150, 100
        ]
        self.model = model
        self.product = product

        # Optional (latitude, longitude) centre and (latitude, longitude)
        # half widths in degrees of the region kept in memory. None keeps
        # the whole globe.
        self.region_center = region_center
        self.region_half_width_deg = region_half_width_deg

        # Loaded fields remain in memory for repeated lookups:
        # name -> array (pressure level, latitude, longitude).
        self.fields = {}
        self.levels_hpa = None
        self.latitudes = None
        self.longitudes = None
        self.loaded = False

    def load_forecast(self):
        """
        Download the selected GFS forecast fields into memory.
        Call once before starting the simulation, or when refreshing
        the forecast.
        """
        print(
            f"Loading GFS forecast {self.initialization:%Y-%m-%d %H}Z "
            f"+{self.forecast_hour}h..."
        )

        H = Herbie(
            self.initialization,
            model=self.model,
            product=self.product,
            fxx=self.forecast_hour,
        )

        variables = {
            "u": "UGRD",
            "v": "VGRD",
            "temperature": "TMP",
            "geopotential_height": "HGT",
            "relative_humidity": "RH",
        }

        # Highest pressure (lowest altitude) first.
        levels = sorted(self.pressure_levels, reverse=True)

        for name, grib_variable in variables.items():
            layers = []

            for level in levels:
                query = f":{grib_variable}:{level} mb"
                ds = H.xarray(query)

                # Select the actual meteorological variable.
                variable_name = {
                    "UGRD": "u",
                    "VGRD": "v",
                    "TMP": "t",
                    "HGT": "gh",
                    "RH": "r",                  # NEW
                }[grib_variable]

                if variable_name not in ds.data_vars:
                    variable_name = next(iter(ds.data_vars))

                field = ds[variable_name]
                if self.latitudes is None:
                    self._set_grid(
                        field["latitude"].values,
                        field["longitude"].values,
                    )

                # Keep only the region around the flight to limit memory.
                layers.append(
                    np.asarray(field.values, dtype=np.float32)[
                        np.ix_(self._rows, self._columns)
                    ]
                )

            self.fields[name] = np.stack(layers)

        self.levels_hpa = np.asarray(levels, dtype=float)
        self.loaded = True
        print("GFS forecast loaded successfully.")

    def _ensure_loaded(self):
        if not self.loaded:
            self.load_forecast()

    def _set_grid(self, latitudes, longitudes):
        """
        Choose the grid rows and columns to keep. Latitudes are stored
        ascending. Longitudes are stored as one increasing run that may
        pass 360 (for example 246..326.5), so neighbouring grid points
        stay neighbours across the 0/360 wrap.
        """
        latitudes = np.asarray(latitudes, dtype=float)
        longitudes = np.asarray(longitudes, dtype=float)

        row_order = np.argsort(latitudes)
        sorted_latitudes = latitudes[row_order]
        lat_step = sorted_latitudes[1] - sorted_latitudes[0]
        lon_step = longitudes[1] - longitudes[0]
        if not (
            np.allclose(np.diff(sorted_latitudes), lat_step)
            and np.allclose(np.diff(longitudes), lon_step)
        ):
            raise ValueError("GFS grid is not regularly spaced.")

        if self.region_center is None:
            rows = row_order
            first_column = 0
            # One extra column repeats the first to close the wrap.
            column_count = len(longitudes) + 1
        else:
            center_lat, center_lon = self.region_center
            half_lat, half_lon = self.region_half_width_deg
            keep = np.abs(sorted_latitudes - center_lat) <= half_lat + lat_step
            rows = row_order[keep]
            first_column = int(math.floor(
                ((center_lon - half_lon) % 360.0 - longitudes[0]) / lon_step
            ))
            column_count = min(
                int(math.ceil(2.0 * half_lon / lon_step)) + 3,
                len(longitudes) + 1,
            )

        column_numbers = first_column + np.arange(column_count)
        self._rows = rows
        self._columns = column_numbers % len(longitudes)
        self.latitudes = latitudes[rows]
        self.longitudes = longitudes[0] + column_numbers * lon_step

    def _grid_position(self, latitude, longitude):
        """
        Grid cell containing a point: (row, fraction of the way to the
        next row, column, fraction of the way to the next column).
        """
        lat_step = self.latitudes[1] - self.latitudes[0]
        lon_step = self.longitudes[1] - self.longitudes[0]

        # Same longitude expressed in the stored (possibly > 360) range.
        stored_longitude = (
            self.longitudes[0] + (longitude - self.longitudes[0]) % 360.0
        )

        if not (
            self.latitudes[0] <= latitude <= self.latitudes[-1]
            and stored_longitude <= self.longitudes[-1]
        ):
            def east(lon):
                return (lon + 180.0) % 360.0 - 180.0

            raise ValueError(
                f"Position ({latitude:.3f}, {east(longitude):.3f}) is "
                f"outside the loaded weather region (latitude "
                f"{self.latitudes[0]:.2f} to {self.latitudes[-1]:.2f}, "
                f"longitude {east(self.longitudes[0]):.2f} to "
                f"{east(self.longitudes[-1]):.2f}). Load a larger region "
                f"with load_weather(region_half_width_deg=...)."
            )

        row = min(
            int((latitude - self.latitudes[0]) // lat_step),
            len(self.latitudes) - 2,
        )
        column = min(
            int((stored_longitude - self.longitudes[0]) // lon_step),
            len(self.longitudes) - 2,
        )
        row_fraction = (latitude - self.latitudes[row]) / lat_step
        column_fraction = (
            (stored_longitude - self.longitudes[column]) / lon_step
        )
        return row, row_fraction, column, column_fraction

    def _column(self, latitude, longitude):
        """
        Vertical profile of every field at a point, interpolated
        bilinearly in latitude and longitude from the four surrounding
        grid points.
        """
        row, row_fraction, column, column_fraction = self._grid_position(
            latitude, longitude
        )
        weights = np.array([
            [(1 - row_fraction) * (1 - column_fraction),
             (1 - row_fraction) * column_fraction],
            [row_fraction * (1 - column_fraction),
             row_fraction * column_fraction],
        ])
        return {
            name: (
                field[:, row:row + 2, column:column + 2].astype(float)
                * weights
            ).sum(axis=(1, 2))
            for name, field in self.fields.items()
        }

    def height_range(self, latitude, longitude):
        """
        Lowest and highest geopotential height (m) with complete data
        at a location; get_weather() accepts altitudes in this range.
        """
        self._ensure_loaded()
        profiles = self._column(latitude, longitude)
        valid = np.all(
            [np.isfinite(profile) for profile in profiles.values()],
            axis=0,
        )
        heights = profiles["geopotential_height"][valid]
        if len(heights) < 2:
            raise ValueError(
                "Insufficient valid pressure-level data for interpolation."
            )
        return float(heights.min()), float(heights.max())

    def get_weather(self, latitude, longitude, altitude_m):
        """
        Get approximate weather conditions at a geographic location
        and altitude.

        Each pressure level is first interpolated bilinearly from the
        four surrounding grid points.
        Vertical interpolation is performed between pressure levels
        using geopotential heights from GFS.

        Returns a dictionary of atmospheric conditions.
        """
        self._ensure_loaded()

        # Convert longitude to the 0-360 convention used by GFS.
        longitude = longitude % 360

        # Vertical profiles interpolated to the point (bilinear).
        profiles = self._column(latitude, longitude)

        pressure_hpa = self.levels_hpa

        heights = np.asarray(
            profiles["geopotential_height"],
            dtype=float,
        )
        u_profile = np.asarray(profiles["u"], dtype=float)
        v_profile = np.asarray(profiles["v"], dtype=float)
        temp_profile = np.asarray(
            profiles["temperature"],
            dtype=float,
        )
        temp_profile = np.asarray(
            profiles["temperature"],
            dtype=float,
        )
        rh_profile = np.asarray(                # NEW
            profiles["relative_humidity"],
            dtype=float,
        )
        # GFS HGT on pressure levels is geopotential height in meters.
        # Sort profiles by increasing height for interpolation.
        valid = (
            np.isfinite(heights)
            & np.isfinite(u_profile)
            & np.isfinite(v_profile)
            & np.isfinite(temp_profile)
            & np.isfinite(rh_profile)           # NEW
        )

        heights = heights[valid]
        u_profile = u_profile[valid]
        v_profile = v_profile[valid]
        temp_profile = temp_profile[valid]
        rh_profile = rh_profile[valid]          # NEW
        pressure_profile = pressure_hpa[valid]

        if len(heights) < 2:
            raise ValueError(
                "Insufficient valid pressure-level data for interpolation."
            )

        order = np.argsort(heights)
        heights = heights[order]
        u_profile = u_profile[order]
        v_profile = v_profile[order]
        temp_profile = temp_profile[order]
        rh_profile = rh_profile[order]          # NEW
        pressure_profile = pressure_profile[order]

        # Interpolate wind and temperature against geopotential height.
        # np.interp clamps values outside the available height range,
        # so we explicitly reject out-of-range altitudes.
        if not heights[0] <= altitude_m <= heights[-1]:
            raise ValueError(
                f"Altitude {altitude_m} m is outside the available "
                f"pressure-level height range "
                f"({heights[0]:.0f}-{heights[-1]:.0f} m)."
            )

        u = float(np.interp(altitude_m, heights, u_profile))
        v = float(np.interp(altitude_m, heights, v_profile))
        temperature_k = float(
            np.interp(altitude_m, heights, temp_profile)
        )

        # Pressure interpolation in log space is more physically
        # appropriate than linear interpolation of pressure.
        log_pressure = np.interp(
            altitude_m,
            heights,
            np.log(pressure_profile * 100.0),
        )
        pressure_pa = float(np.exp(log_pressure))

        relative_humidity = float(
            np.clip(np.interp(altitude_m, heights, rh_profile), 0.0, 100.0)
        )

        # Virtual temperature (temperature adjusted for humidity).
        # Saturation vapor pressure via the Bolton (1980) formula.
        temp_c = temperature_k - 273.15
        e_sat = 611.2 * math.exp(17.67 * temp_c / (temp_c + 243.5))  # Pa
        e = (relative_humidity / 100.0) * e_sat                      # Pa
        mixing_ratio = 0.622 * e / (pressure_pa - e)                 # kg/kg
        virtual_temperature_k = temperature_k * (
            (1.0 + mixing_ratio / 0.622) / (1.0 + mixing_ratio)
        )

        # Ideal-gas approximation for dry air.
        R_d = 287.05  # J/(kg K)
        density = pressure_pa / (R_d * temperature_k)

        wind_speed = math.hypot(u, v)

        # Meteorological direction the wind is blowing FROM.
        wind_direction = (
            math.degrees(math.atan2(-u, -v)) + 360
        ) % 360

        return {
            "latitude": latitude,
            "longitude": longitude,
            "altitude_m": altitude_m,
            "forecast_initialization_utc": (
                self.initialization.isoformat() + "Z"
            ),
            "forecast_hour": self.forecast_hour,
            "wind_u_m_s": u,
            "wind_v_m_s": v,
            "wind_speed_m_s": wind_speed,
            "wind_direction_from_deg": wind_direction,
            "temperature_k": temperature_k,
            "temperature_c": temperature_k - 273.15,
            "relative_humidity_percent": relative_humidity,              # NEW
            "virtual_temperature_k": virtual_temperature_k,              # NEW
            "virtual_temperature_c": virtual_temperature_k - 273.15,     # NEW
            "pressure_pa": pressure_pa,
            "density_kg_m3": density,
        }

    def refresh_forecast(
        self,
        initialization=None,
        forecast_hour=None,
    ):
        """
        Replace the cached forecast with a new initialization and/or
        forecast hour. This performs network requests.
        """
        if initialization is not None:
            self.initialization = initialization

        if forecast_hour is not None:
            self.forecast_hour = forecast_hour

        self.fields.clear()
        self.loaded = False
        self.load_forecast()


# ---------------------------------------------------------------------
# Simulation interface (used by world.ipynb)
# ---------------------------------------------------------------------
from datetime import timedelta, timezone

# Pressure levels extended into the stratosphere (5 hPa is roughly
# 36 km) so the forecast covers the full balloon flight.
EXTENDED_PRESSURE_LEVELS = [
    1000, 925, 850, 700, 600, 500,
    400, 300, 250, 200, 150, 100,
    70, 50, 30, 20, 10, 7, 5
]

# Region kept in memory around the launch site: degrees of latitude and
# of longitude either side. Leaving it raises an error.
DEFAULT_REGION_HALF_WIDTH_DEG = (15.0, 40.0)

_run = None          # settings of the selected GFS run (see load_weather)
_providers = {}      # forecast hour -> loaded WeatherProvider


def _as_utc_naive(moment):
    """UTC datetime without tzinfo; naive datetimes are taken as UTC."""
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def latest_gfs_cycle(delay_hours=6):
    """
    Most recent 00/06/12/18 UTC GFS cycle that should already be
    published. GFS output appears roughly 4-5 hours after the cycle
    time, so the default delay leaves some margin.
    """
    now = (
        datetime.now(timezone.utc).replace(tzinfo=None)
        - timedelta(hours=delay_hours)
    )
    return now.replace(
        hour=now.hour - now.hour % 6,
        minute=0,
        second=0,
        microsecond=0,
    )


def select_gfs_run(launch_time):
    """Latest published GFS run at or before launch_time (UTC)."""
    launch_time = _as_utc_naive(launch_time)
    launch_cycle = launch_time.replace(
        hour=launch_time.hour - launch_time.hour % 6,
        minute=0,
        second=0,
        microsecond=0,
    )
    return min(latest_gfs_cycle(), launch_cycle)


# GFS 0.25 degree output is hourly to 120 h, then every 3 h to 384 h.
MAX_FORECAST_HOUR = 384


def _bracketing_forecast_hours(lead_hours):
    """
    Forecast hours either side of a lead time (the same hour twice when
    the lead time falls exactly on one).
    """
    if not 0.0 <= lead_hours <= MAX_FORECAST_HOUR:
        raise ValueError(
            f"Requested time is {lead_hours:.2f} h after the GFS run "
            f"{_run['initialization']:%Y-%m-%d %H}Z, outside the "
            f"0-{MAX_FORECAST_HOUR} h that run covers."
        )
    if lead_hours <= 120:
        return math.floor(lead_hours), math.ceil(lead_hours)
    lower = 120 + 3 * math.floor((lead_hours - 120) / 3)
    upper = lower if lower == lead_hours else lower + 3
    return lower, upper


def _provider_for_hour(forecast_hour):
    """Load one forecast hour of the selected run, once, and cache it."""
    if forecast_hour not in _providers:
        provider = WeatherProvider(
            initialization=_run["initialization"],
            forecast_hour=forecast_hour,
            pressure_levels=_run["pressure_levels"],
            region_center=_run["region_center"],
            region_half_width_deg=_run["region_half_width_deg"],
        )
        provider.load_forecast()
        _providers[forecast_hour] = provider
    return _providers[forecast_hour]


def load_weather(
    initialization=None,
    forecast_hour=None,
    pressure_levels=None,
    launch_time=None,
    latitude=None,
    longitude=None,
    region_half_width_deg=DEFAULT_REGION_HALF_WIDTH_DEG,
):
    """
    Select the GFS run used by get_weather() and download the forecast
    hour for the launch. Later forecast hours are downloaded by
    get_weather() when the flight reaches them, and cached.

    launch_time: UTC datetime of the release (naive datetimes are taken
        as UTC). Defaults to now. The latest published GFS run at or
        before it is used unless initialization is given.
    forecast_hour: forecast hour get_weather() uses when it is called
        without valid_time. Defaults to the hour of launch_time.
    latitude, longitude: centre of the region kept in memory, normally
        the launch site. None keeps the whole globe.

    If the extended stratospheric levels cannot be loaded, the default
    WeatherProvider levels are used instead.
    """
    global _run, _providers

    if launch_time is None:
        launch_time = datetime.now(timezone.utc)
    launch_time = _as_utc_naive(launch_time)

    if initialization is None:
        initialization = select_gfs_run(launch_time)
    initialization = _as_utc_naive(initialization)

    if forecast_hour is None:
        default_valid_time = launch_time
    else:
        default_valid_time = initialization + timedelta(hours=forecast_hour)
    lead_hours = (default_valid_time - initialization).total_seconds() / 3600

    region_center = None
    if latitude is not None and longitude is not None:
        region_center = (latitude, longitude)

    _providers = {}
    _run = {
        "initialization": initialization,
        "launch_time": launch_time,
        "default_valid_time": default_valid_time,
        "pressure_levels": pressure_levels or EXTENDED_PRESSURE_LEVELS,
        "region_center": region_center,
        "region_half_width_deg": (
            region_half_width_deg if region_center is not None else None
        ),
    }

    try:
        lower, upper = _bracketing_forecast_hours(lead_hours)
    except ValueError:
        _run = None
        raise

    try:
        provider = _provider_for_hour(lower)
    except Exception as error:
        if pressure_levels is not None:
            _run = None
            raise
        print(
            f"Extended pressure levels unavailable ({error}); "
            f"retrying with default levels."
        )
        _run["pressure_levels"] = None
        try:
            provider = _provider_for_hour(lower)
        except Exception:
            _run = None
            raise

    # The second hour bracketing the launch, for interpolation in time.
    try:
        _provider_for_hour(upper)
    except Exception:
        _run = None
        raise

    return provider


def _weather_for_hour(forecast_hour, latitude, longitude, altitude_m):
    """
    Weather from one forecast hour of the selected run, with
    extrapolation outside the pressure-level height range and moist-air
    density.
    """
    provider = _provider_for_hour(forecast_hour)

    lowest, highest = provider.height_range(latitude, longitude)
    query_altitude = float(min(max(altitude_m, lowest), highest))
    weather = provider.get_weather(latitude, longitude, query_altitude)

    R_d = 287.05      # J/(kg K)
    g0 = 9.80665      # m/s^2
    dz = altitude_m - query_altitude

    if dz != 0.0:
        reference_temperature = weather["temperature_k"]
        moisture_ratio = (
            weather["virtual_temperature_k"] / reference_temperature
        )

        if dz < 0:
            temperature_k = reference_temperature - 0.0065 * dz
        else:
            temperature_k = reference_temperature

        mean_virtual_temperature = (
            0.5 * (reference_temperature + temperature_k) * moisture_ratio
        )
        weather["pressure_pa"] *= math.exp(
            -g0 * dz / (R_d * mean_virtual_temperature)
        )
        weather["temperature_k"] = temperature_k
        weather["temperature_c"] = temperature_k - 273.15
        weather["virtual_temperature_k"] = temperature_k * moisture_ratio
        weather["virtual_temperature_c"] = (
            weather["virtual_temperature_k"] - 273.15
        )

    # Moist-air density: humid air is lighter than dry air.
    weather["density_kg_m3"] = weather["pressure_pa"] / (
        R_d * weather["virtual_temperature_k"]
    )
    weather["latitude"] = latitude
    weather["longitude"] = longitude
    weather["altitude_m"] = altitude_m
    weather["extrapolated"] = dz != 0.0

    return weather


# Quantities interpolated linearly in time between forecast hours; the
# rest are derived from them afterwards.
_TIME_INTERPOLATED_KEYS = (
    "wind_u_m_s",
    "wind_v_m_s",
    "temperature_k",
    "virtual_temperature_k",
    "relative_humidity_percent",
    "pressure_pa",
)


def get_weather(latitude, longitude, altitude_m, valid_time=None):
    """
    Weather at a location and altitude for the balloon simulation.

    valid_time: UTC datetime the weather should be valid at (naive
        datetimes are taken as UTC). Defaults to the launch time given
        to load_weather(). The two GFS forecast hours either side of it
        are interpolated linearly in time; hours not loaded yet are
        downloaded and cached.

    Wraps WeatherProvider.get_weather() and adds:
      - automatic loading of the latest forecast on first call
      - extrapolation outside the pressure-level height range
        (standard lapse rate below, isothermal above, hydrostatic
        pressure, wind held at the nearest level)
      - air density from virtual temperature (includes humidity)
      - longitude returned in the same convention it was given

    forecast_hour is the earlier of the two forecast hours used and
    forecast_hour_upper the later one.
    """
    if _run is None:
        load_weather(latitude=latitude, longitude=longitude)

    if valid_time is None:
        valid_time = _run["default_valid_time"]
    valid_time = _as_utc_naive(valid_time)
    lead_hours = (
        (valid_time - _run["initialization"]).total_seconds() / 3600
    )
    lower, upper = _bracketing_forecast_hours(lead_hours)

    weather = _weather_for_hour(lower, latitude, longitude, altitude_m)

    if upper != lower:
        later = _weather_for_hour(upper, latitude, longitude, altitude_m)
        weight = (lead_hours - lower) / (upper - lower)
        for key in _TIME_INTERPOLATED_KEYS:
            weather[key] += weight * (later[key] - weather[key])

        u = weather["wind_u_m_s"]
        v = weather["wind_v_m_s"]
        weather["wind_speed_m_s"] = math.hypot(u, v)
        weather["wind_direction_from_deg"] = (
            math.degrees(math.atan2(-u, -v)) + 360
        ) % 360
        weather["temperature_c"] = weather["temperature_k"] - 273.15
        weather["virtual_temperature_c"] = (
            weather["virtual_temperature_k"] - 273.15
        )
        R_d = 287.05  # J/(kg K)
        weather["density_kg_m3"] = weather["pressure_pa"] / (
            R_d * weather["virtual_temperature_k"]
        )
        weather["extrapolated"] = (
            weather["extrapolated"] or later["extrapolated"]
        )

    weather["forecast_hour_upper"] = upper
    weather["lead_time_hours"] = lead_hours
    weather["valid_time_utc"] = valid_time.isoformat() + "Z"

    return weather


if __name__ == "__main__":
    # Basic standalone demonstration.
    # Replace these coordinates and altitude with your launch scenario.
    weather = get_weather(
        latitude=37.89,
        longitude=-122.30,
        altitude_m=5000,
    )

    print("\n--- Balloon Weather ---")
    for key, value in weather.items():
        print(f"{key}: {value}")
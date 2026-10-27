
from datetime import datetime
from pathlib import Path
import math

import numpy as np
import xarray as xr
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

    This initial implementation uses pressure-level data and
    approximate altitude interpolation. Forecast time interpolation
    and automatic selection of the latest available GFS cycle can
    be added as the next development step.
    """

    def __init__(
        self,
        initialization=None,
        forecast_hour=0,
        pressure_levels=None,
        model="gfs",
        product="pgrb2.0p25",
    ):
        self.initialization = initialization or datetime(2026, 10, 12, 12)
        self.forecast_hour = forecast_hour
        self.pressure_levels = pressure_levels or [
            1000, 925, 850, 700, 600, 500,
            400, 300, 250, 200, 150, 100
        ]
        self.model = model
        self.product = product

        # Loaded datasets remain in memory for repeated lookups.
        self.fields = {}
        self.loaded = False

    def load_forecast(self):
        """
        Download the selected GFS forecast fields into memory.
        Call once before starting the simulation, or when refreshing
        the forecast.
        """
        print("Loading GFS forecast...")

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

        for name, grib_variable in variables.items():
            datasets = []

            for level in self.pressure_levels:
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

                field = ds[variable_name].expand_dims(
                    pressure_level=[level]
                )
                datasets.append(field)

            self.fields[name] = xr.concat(
                datasets,
                dim="pressure_level",
            ).sortby("pressure_level", ascending=False)

        self.loaded = True
        print("GFS forecast loaded successfully.")

    def _ensure_loaded(self):
        if not self.loaded:
            self.load_forecast()

    def get_weather(self, latitude, longitude, altitude_m):
        """
        Get approximate weather conditions at a geographic location
        and altitude.

        The nearest horizontal grid point is selected first.
        Vertical interpolation is performed between pressure levels
        using geopotential heights from GFS.

        Returns a dictionary of atmospheric conditions.
        """
        self._ensure_loaded()

        # Convert longitude to the 0-360 convention used by GFS.
        longitude = longitude % 360

        # Extract vertical profiles at the nearest grid point.
        profiles = {}

        for name, ds in self.fields.items():
            profiles[name] = ds.sel(
                latitude=latitude,
                longitude=longitude,
                method="nearest",
            ).values.squeeze()

        pressure_hpa = np.asarray(
            self.fields["temperature"]["pressure_level"].values,
            dtype=float,
        )

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

_provider = None


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


def load_weather(
    initialization=None,
    forecast_hour=None,
    pressure_levels=None,
):
    """
    Download the forecast used by get_weather(). Defaults to the latest
    GFS cycle and the forecast hour closest to the current time. If the
    extended stratospheric levels cannot be loaded, the default
    WeatherProvider levels are used instead.
    """
    global _provider

    if initialization is None:
        initialization = latest_gfs_cycle()

    if forecast_hour is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        forecast_hour = max(
            0,
            round((now - initialization).total_seconds() / 3600),
        )

    provider = WeatherProvider(
        initialization=initialization,
        forecast_hour=forecast_hour,
        pressure_levels=pressure_levels or EXTENDED_PRESSURE_LEVELS,
    )

    try:
        provider.load_forecast()
    except Exception as error:
        if pressure_levels is not None:
            raise
        print(
            f"Extended pressure levels unavailable ({error}); "
            f"retrying with default levels."
        )
        provider = WeatherProvider(
            initialization=initialization,
            forecast_hour=forecast_hour,
        )
        provider.load_forecast()

    _provider = provider
    return provider


def get_weather(latitude, longitude, altitude_m):
    """
    Weather at a location and altitude for the balloon simulation.

    Wraps WeatherProvider.get_weather() and adds:
      - automatic loading of the latest forecast on first call
      - extrapolation outside the pressure-level height range
        (standard lapse rate below, isothermal above, hydrostatic
        pressure, wind held at the nearest level)
      - air density from virtual temperature (includes humidity)
      - longitude returned in the same convention it was given
    """
    if _provider is None:
        load_weather()

    heights = np.asarray(
        _provider.fields["geopotential_height"].sel(
            latitude=latitude,
            longitude=longitude % 360,
            method="nearest",
        ).values,
        dtype=float,
    ).squeeze()
    heights = heights[np.isfinite(heights)]

    query_altitude = float(
        min(max(altitude_m, heights.min()), heights.max())
    )
    weather = _provider.get_weather(latitude, longitude, query_altitude)

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
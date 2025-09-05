import matplotlib.pyplot as plt
import numpy as np
import warnings
import xarray as xr
from IPython import embed

# Settings
window_intervals = [
    (10.3, 10.8),
    (11.0, 12.0),
    # (13.3, 13.6),
    # (1.0, 15.0)  # All data
]



number_test_spectra = 3

# Constants
files = {
    "clear":  "./Dataset_cloud_clear_nosol/clear_sky_IASI_radiances.nc",
    "cloudy": "./Dataset_cloud_clear_nosol/cloudy_sky_IASI_radiances.nc",
    "test":   "./Dataset_cloud_clear_nosol/test_set.nc",
}

dim_mapping = {"n_match": "n_spectra", "n_ch_I": "n_channels"}
var_mapping = {
    "IASI_wavenumber": "wavenumber",
    "IASI_radiance": "rad",
}

colors = {
    "clear": "tab:red",
    "cloudy": "tab:blue",
    "test": "k",
}

training_labels = ["clear", "cloudy"]

# Helpers
def to_wavelength(wavenumber):
    """Transform wavenumber (cm-1) to wavelength (micron)."""
    return np.divide(1, wavenumber) * 1e4

def get_window_mask(wavelengths, intervals):
    """
    Returns a mask where the wavelengths are inside any of the intervals.
    """
    mask = xr.zeros_like(wavelengths, dtype=bool)
    masks = mask.expand_dims(dim={"dim_intervals": len(intervals)}).copy()
    for i, interval in enumerate(intervals):
        masks.loc[{"dim_intervals": i}] = (wavelengths > interval[0]) & (wavelengths < interval[1])
    mask = masks.any(dim="dim_intervals")
    return mask

def rad2brightb(wav, rad):
    """From radiance to BT"""
    wav = np.array(wav)  # wavenumber (cm-1)
    rad = np.array(rad)  # Radiance   (mW/m2 sr cm-1)
    h = 6.6260755e-34  # Planck constant (Joule second)
    c = 2.9979246e8    # Speed of light in vacuum (meters per second)
    k = 1.380658e-23   # Boltzmann constant (Joules per Kelvin)
    c1 = (2.0 * h * c**2) * 1.0e11
    c2 = (h * c * 100.0) / k  # 100 is to account for cm-1
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        BT = c2 * wav / (np.log(1 + (c1 * wav**3) / rad))
    return BT

def get_index_of_wavelength(wavelengths, wavelength):
    """Find the index of the wavelength (or closest) in an array of wavelengths."""
    if isinstance(wavelengths, xr.DataArray):
        _w = wavelengths.values
    else:
        _w = wavelengths
    return np.argmin(np.abs(_w - wavelength))

# Main loop
def main():
    # Load datasets
    print("Loading datasets ...")
    datasets = {}
    for label, file in files.items():
        print(f"Reading file {file}")
        with xr.open_dataset(file) as ds:
            ds = ds.rename(dim_mapping)
            ds = ds.rename_vars(var_mapping)
            tb = rad2brightb(ds.wavenumber, ds.rad)
            ds["TB"] = xr.DataArray(tb, dims=ds.rad.dims)
            ds["wavelength"] = to_wavelength(ds.wavenumber)
            ds["windows"] = get_window_mask(ds.wavelength, window_intervals)
            datasets[label] = ds

    # Compute statistics
    print("Compute statistics ...")
    means = {}
    stds = {}
    for label, ds in datasets.items():
        window_mask = ds.windows
        means[label] = xr.where(window_mask, ds.TB.mean(dim="n_spectra"), np.nan)
        stds[label] = xr.where(window_mask, ds.TB.std(dim="n_spectra"), np.nan)

    print("Plot ...")

    # Plot lat-lon
    if False:
        for label in training_labels:
            ds = datasets[label]
            plt.plot(ds["fov_lon"], ds["fov_lat"], ".", label=label)
        plt.legend()
        plt.xlabel("lon (deg)")
        plt.ylabel("lat (deg)")
        plt.show()


    tb_thresh = 278.0
    wavelength_1 = 10.75
    wavelength_2 = 11.85
    # Plot mean Tb
    if True:
        fig, axes = plt.subplots(nrows=1, sharex=True)
        for label in training_labels:
            ds = datasets[label]
            axes.plot(ds.wavelength, means[label], color=colors[label], label=f"mean {label}")
            axes.fill_between(
                ds.wavelength,
                means[label] - stds[label],
                means[label] + stds[label],
                color=colors[label],
                alpha=0.33,
            )
        

        axes.axhline(tb_thresh, ls="--", c="k")
        axes.axvline(wavelength_1, ls=":", c="k")
        axes.axvline(wavelength_2, ls=":", c="k")
        ds = datasets["test"]
        for i_plot in range(number_test_spectra):
            label = "test spectra" if i_plot == 0 else None
            tb = xr.where(ds.windows, ds.TB.isel(n_spectra=i_plot), np.nan)
            axes.plot(ds.wavelength, tb, c="k", label=label)
        axes.set_xlabel("wavelength ($\mu$m)")
        axes.set_ylabel("TB (K)")
        axes.legend()
        plt.show()

    # 2D plot
    if False:
        fig, axes = plt.subplots(ncols=2)
        for i_ax, label in enumerate(training_labels):
            ds = datasets[label]
            ds.rad.plot(ax=axes[i_ax])
            axes[i_ax].set_title(label)
        plt.show()

    # Scatter plot
    if True:
        for label, ds in datasets.items():
            i_1 = get_index_of_wavelength(ds.wavelength, wavelength_1)
            i_2 = get_index_of_wavelength(ds.wavelength, wavelength_2)
            tb_1 = ds.TB.isel(n_channels=i_1)
            tb_2 = ds.TB.isel(n_channels=i_2)
            plt.plot(
                tb_1, tb_2 - tb_1, "o", color=colors[label], label=label, alpha=0.5
            )
        plt.axvline(tb_thresh, c="k", ls="--")
        plt.axhline(0, c="grey", ls="-", lw=0.5)
        plt.xlabel(f"TB at {wavelength_1} $\mu$m (K)")
        plt.ylabel(f"TB at {wavelength_2} $\mu$m - TB at {wavelength_1} $\mu$m (K)")
        plt.legend()
        plt.show()

    # Classify
    print("Classify ...")
    ds = datasets["test"]
    clas_wavelength = wavelength_1
    i_1 = get_index_of_wavelength(ds.wavelength, clas_wavelength)
    tb_1 = ds.TB.isel(n_channels=i_1)
    is_cloudy = tb_1 < tb_thresh  # ToDo: Where is our control?

    # Plot classification
    if True:
        bins = np.arange(tb_thresh - 60, tb_thresh + 26, 1)
        fig, axes = plt.subplots(nrows=2, sharex=True)
        for i, label in enumerate(training_labels):
            ds = datasets[label]
            i_1 = get_index_of_wavelength(ds.wavelength, wavelength_1)
            axes[0].hist(
                ds.TB.isel(n_channels=i_1),
                bins=bins,
                facecolor=colors[label],
                alpha=0.5,
            )
        axes[1].hist(
            tb_1[is_cloudy], bins=bins, facecolor=colors["cloudy"], alpha=0.5, hatch="/"
        )
        axes[1].hist(
            tb_1[~is_cloudy],
            bins=bins,
            facecolor=colors["clear"],
            alpha=0.5,
            hatch="\\",
        )
        axes[0].set_ylabel("Training data")
        axes[1].set_ylabel("Test data")
        axes[1].set_xlabel("Tb (K)")
        plt.show()

# Execute
if __name__ == "__main__":
    main()

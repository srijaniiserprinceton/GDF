import cdflib


cdflib_major_version = int(cdflib.__version__.split('.')[0])

if cdflib_major_version < 1:
    from cdflib import cdf_to_xarray
    def _cdf_to_xarray(*args, **kwargs):
        return cdf_to_xarray(*args, **kwargs)
else:
    from cdflib.xarray import cdf_to_xarray
    def _cdf_to_xarray(*args, **kwargs):
        return cdf_to_xarray(*args, **kwargs)

try:
    from cdflib.epochs_astropy import CDFAstropy
    def _convert_to_astropy(epochs):
        return CDFAstropy.convert_to_astropy(epochs)
except (ImportError, AttributeError):
    from cdflib.epochs_astropy import CDFAstropy
    def _convert_to_astropy(epochs):
        return CDFAstropy.convert_to_astropy(epochs)

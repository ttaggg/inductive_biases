"""Alias for SIREN-FM, for backcompat reasons."""

from ib.models.inrs.sirenfm import Modulator
from ib.models.inrs.sirenfm import SirenFmLayer
from ib.models.inrs.sirenfm import SirenFm


class Modulator(Modulator):
    pass


class VainerLayer(SirenFmLayer):
    pass


class Vainer(SirenFm):
    pass

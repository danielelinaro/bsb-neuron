import numpy
from bsb import LocationTargetting, config, warn

from ..device import NeuronDevice


@config.node
class RandomCurrentClamp(NeuronDevice, classmap_entry="random_current_clamp"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "soma"})
    amplitude_mean = config.attr(type=float, required=True)
    amplitude_std = config.attr(type=float, required=True)
    before = config.attr(type=float, default=None)
    duration = config.attr(type=float, default=None)
    seed = config.attr(type=int, default=None)

    def implement(self, adapter, simulation, simdata):
        if self.seed is not None:
            from numpy.random import MT19937, RandomState, SeedSequence

            self.rs = RandomState(MT19937(SeedSequence(self.seed)))
        else:
            self.rs = numpy.random
        for model, pop in self.targetting.get_targets(
            adapter, simulation, simdata
        ).items():
            for target in pop:
                clamped = False
                for location in self.locations.get_locations(target):
                    if clamped:
                        warn(f"Multiple current clamps placed on {target}")
                    self._add_clamp(
                        simdata,
                        location,
                        name=self.name,
                        cell_type=target.cell_model.name,
                        cell_id=target.id,
                    )
                    clamped = True

    def _add_clamp(self, simdata, location, **annotations):
        sx = location.arc(0.5)
        amp = self.rs.normal(loc=self.amplitude_mean, scale=self.amplitude_std)
        print("Amplitude: {:g} pA".format(amp * 1e3))
        clamp = location.section.iclamp(
            x=sx, delay=self.before, duration=self.duration, amplitude=amp
        )
        simdata.result.record(clamp._ref_i, **annotations, units="nA")

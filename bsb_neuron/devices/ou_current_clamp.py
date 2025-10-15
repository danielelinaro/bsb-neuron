import numpy as np
from bsb import LocationTargetting, config, warn
from neuron import h
from numpy.random import MT19937, RandomState, SeedSequence

from ..device import NeuronDevice


def OU(dt, mean, stddev, tau, N, random_state=None):
    """
    OU returns a realization of the Ornstein-Uhlenbeck process given its
    parameters.

    Parameters
    ----------
    dt : float
        Time step.
    mean : float
        Mean of the process.
    stddev : float
        Standard deviation of the process.
    tau : float
        Time constant of the autocorrelation function of the process.
    N : integer
        Number of samples.
    random_state : RandomState object, optional
        The object used to draw the random numbers. The default is None.

    Returns
    -------
    ou : array of length N
        A realization of the Ornstein-Uhlenbeck process with the above parameters.

    """
    const = 2 * stddev**2 / tau
    mu = np.exp(-dt / tau)
    coeff = np.sqrt(const * tau / 2 * (1 - mu**2))
    if random_state is not None:
        rnd = random_state.normal(size=N)
    else:
        rnd = np.random.normal(size=N)
    ou = np.zeros(N)
    ou[0] = mean
    for i in range(1, N):
        ou[i] = mean + mu * (ou[i - 1] - mean) + coeff * rnd[i]
    return ou


@config.node
class OUCurrentClamp(NeuronDevice, classmap_entry="ou_current_clamp"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "soma"})
    mean_amplitude = config.attr(type=float, required=True)
    stddev_amplitude = config.attr(type=float, required=True)
    acorr_time_const = config.attr(type=float, required=True)
    before = config.attr(type=float, default=None)
    duration = config.attr(type=float, default=None)
    seed = config.attr(type=int, default=None)

    def implement(self, adapter, simulation, simdata):
        if self.seed is None:
            import time

            self.seed = int(time.time())
        self.master_rs = RandomState(MT19937(SeedSequence(self.seed)))
        self.random_states = []

        self.dt = simulation.resolution
        t = np.r_[
            0 : simulation.duration + simulation.resolution / 2 : simulation.resolution
        ]
        self.n_samples = t.size
        self.mask = (t < self.before) & (t > self.before + self.duration)

        self.t_vec = h.Vector(t)
        self.ou_vecs = []

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
        clamp = location.section.iclamp(x=sx, delay=0, duration=1e6, amplitude=0)
        seed = self.master_rs.randint(100000)
        rs = RandomState(MT19937(SeedSequence(seed)))
        ou = OU(
            self.dt,
            self.mean_amplitude,
            self.stddev_amplitude,
            self.acorr_time_const,
            self.n_samples,
            random_state=rs,
        )
        ou[self.mask] = 0
        vec = h.Vector(ou)
        vec.play(clamp._ref_amp, self.t_vec, 1)
        self.ou_vecs.append(vec)
        simdata.result.record(clamp._ref_i, **annotations, units="nA")

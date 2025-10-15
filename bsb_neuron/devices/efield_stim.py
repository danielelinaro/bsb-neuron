import math

import numpy as np
from bsb import LocationTargetting, config

from bsb_neuron.device import NeuronDevice


def compute_Ve_over_cell(
    cell, Efield=None, Emag=None, theta=None, phi=None, Ve_soma=0, full_output=False
):
    from neuron import nrn

    if hasattr(cell, "sections"):
        # potentially a cell from the repo of the Brain and Behavioral
        # Sciences Laboratory at the University of Pavia
        cell_secs = cell.sections
    elif isinstance(cell.soma[0], nrn.Section) and hasattr(cell, "all"):
        # potentially a cell from the repo of the Blue Brain Project
        cell_secs = [sec for sec in cell.all]
    else:
        raise Exception("Do not know how to deal with this type of cell")

    if Efield is not None:
        const_E = False
    elif Emag is not None and theta is not None and phi is not None:
        const_E = True
        sin_th, cos_th = np.sin(theta), np.cos(theta)
        sin_ph, cos_ph = np.sin(phi), np.cos(phi)
    else:
        raise Exception("Either Efield or Emag, theta and phi must be passed")

    from scipy.interpolate import interp1d

    # potential difference between the parent and the present points
    _compute_Ve = lambda x_c, x_p, Ve_p, Efun: Ve_p - 0.5 * (Efun(x_c) + Efun(x_p)) @ (
        (x_c - x_p) * 1e-6
    )
    # the relative locations of all segments in a section
    _segment_loc = lambda nseg: np.linspace(0, 1, 2 * nseg + 1)[1::2]

    points_data = {
        sec.name(): np.array(
            [
                [
                    sec.x3d(i),  # x coord
                    sec.y3d(i),  # y coord
                    sec.z3d(i),  # z coord
                    sec.arc3d(i) / sec.L,  # relative position in section
                    # electrical potential (will be filled later)
                    Ve_soma,
                ]
                for i in range(sec.n3d())
            ]
        )
        for sec in cell_secs
    }
    segments = {sec.name(): np.zeros(sec.nseg) + Ve_soma for sec in cell_secs}

    for sec in cell_secs:
        key_c = sec.name()
        if const_E:
            for j in range(sec.n3d()):
                x, y, z = points_data[key_c][j][:3] * 1e-6  # [m]
                points_data[key_c][j, -1] = -Emag * (
                    x * sin_th * cos_ph + y * sin_th * sin_ph + z * cos_th
                )
        else:
            if sec.parentseg() is not None:
                X_c = points_data[key_c][0, :3]
                key_p = sec.parentseg().sec.name()
                if np.sum((X_c - points_data[key_p][-1, :3]) ** 2) < 1e-6:
                    Ve = points_data[key_p][-1, -1]
                else:
                    X_p = points_data[key_p][-2, :3]
                    assert np.sum((X_c - X_p) ** 2) > 0
                    Ve = _compute_Ve(X_c, X_p, points_data[key_p][-2, -1], Efield)
                points_data[key_c][0, -1] = Ve
            for j in range(1, sec.n3d()):
                X_c = points_data[key_c][j, :3]
                X_p = points_data[key_c][j - 1, :3]
                points_data[key_c][j, -1] = _compute_Ve(
                    X_c, X_p, points_data[key_c][j - 1, -1], Efield
                )

        if sec.n3d() == 1:
            segments[key_c][0] = points_data[key_c][0, -1]
        else:
            bins = np.linspace(0, 1, sec.nseg + 1)
            pos = points_data[key_c][:, 3]
            idx = np.digitize(pos, bins, right=False) - 1
            idx[idx == sec.nseg] = sec.nseg - 1
            for j, seg in enumerate(sec):
                (jdx,) = np.where(idx == j)
                if jdx.size == 0:
                    if j == 0:
                        Ve = segments[key_p][-1]
                    else:
                        Ve = segments[key_c][j - 1]
                else:
                    if seg.x < points_data[key_c][jdx[0], 3]:
                        jdx = np.concatenate(([jdx[0] - 1], jdx))
                    if seg.x > points_data[key_c][jdx[-1], 3]:
                        jdx = np.concatenate((jdx, [jdx[-1] + 1]))
                    Ve = interp1d(points_data[key_c][jdx, 3], points_data[key_c][jdx, 4])(
                        seg.x
                    )
                segments[key_c][j] = Ve
    V_extra = [segments[sec.name()] for sec in cell_secs]
    if full_output:
        return V_extra, cell_secs, points_data
    return V_extra


@config.node
class ElectricFieldStimulator(NeuronDevice, classmap_entry="electrical_field_stimulator"):
    locations = config.attr(type=LocationTargetting, default={"strategy": "everywhere"})
    magnitude = config.attr(type=float, required=True)
    theta = config.attr(type=float, required=True)
    phi = config.attr(type=float, required=True)
    delay = config.attr(type=float, default=0)
    stim_freq = config.attr(type=float, default=-1)
    cycle_freq = config.attr(type=float, required=True)
    n_cycles = config.attr(type=int, default=1)
    waveform_fun = config.attr(type=str, default="math.sin")

    def implement(self, adapter, simulation, simdata):
        from neuron import h

        dur, dt = simulation.duration, simulation.resolution
        # use vectorize here so that any function can be mapped onto a NumPy array
        waveform_fun = np.vectorize(eval(self.waveform_fun))
        time = np.r_[0:dur:dt]
        e_stim = np.zeros_like(time)
        n_stim_samples = int(np.ceil(1000 / (self.cycle_freq * dt) * self.n_cycles))
        stim_time = np.arange(n_stim_samples) * dt / 1000
        stim = waveform_fun(2 * np.pi * self.cycle_freq * stim_time)
        offset = int(self.delay / dt)
        if offset + n_stim_samples < time.size:
            e_stim[offset : offset + n_stim_samples] = stim
        if self.stim_freq > 0:
            T = int(1000 / self.stim_freq / dt)
            offset += T
            while offset + n_stim_samples < time.size:
                e_stim[offset : offset + n_stim_samples] = stim
                offset += T
        self.t_vec = h.Vector(time)
        self.E_vecs = []
        for model, population in self.targetting.get_targets(
            adapter, simulation, simdata
        ).items():
            E_vecs = [[[] for _ in range(len(cell.sections))] for cell in population]
            for i, cell in enumerate(population):
                Ve = compute_Ve_over_cell(
                    cell, Emag=self.magnitude, theta=self.theta, phi=self.phi
                )
                for j, sec in enumerate(cell.sections):
                    sec.insert("extracellular")
                    for k, seg in enumerate(sec):
                        vec = h.Vector(e_stim * Ve[j][k] * 1e3)
                        vec.play(seg._ref_e_extracellular, self.t_vec, 1)
                        E_vecs[i][j].append(vec)
            self.E_vecs.append(E_vecs)

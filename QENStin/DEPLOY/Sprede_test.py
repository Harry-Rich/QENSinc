import pytest
import numpy as np
import Sprede as sp
import MDAnalysis as mda



@pytest.fixture(scope = 'session')
def test_sp():
    time_step = 1.0
    step_skip = 1
    test_u = mda.Universe('input_data/example_LAMMPS.data','input_data/example_LAMMPS.dcd')
    test_u_sp = sp.MDAnalysisParser(test_u,specie=['1','2'],isotopes = ['41K','35Cl'],time_step=time_step,step_skip=step_skip)
    return test_u_sp

@pytest.fixture(scope = 'session')
def incoh(test_sp):
    q_points = test_sp.calculate_q_points(q_max = 5)
    test_incoh = test_sp.calculate_Finc_qt(q_points)
    return test_incoh


def test_dt_assignement(test_sp):
    dt_correct = np.arange(1,200,2)
    dt_correct[-1] = 200
    assert dt_correct.all() == test_sp.delta_t.all()


def test_scattering_length(test_sp):
    assert np.sum(test_sp.scattering_lengths) == (52*6.1 + (256-52)*1.5)

def test_incoh_shape(incoh,test_sp):
    q_points = test_sp.calculate_q_points(q_max = 5)
    assert incoh.shape == (len(q_points),len(test_sp.delta_t))

def test_incoh_0th_q_point(incoh):
    assert (incoh[0,:].round(3) == 9.351).all()
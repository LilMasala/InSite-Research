"""Reproduce a synthetic numerical check against simglucose; no fitted records."""
from pathlib import Path
import os
import sys
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'insite-portfolio-mpl'))
os.environ.setdefault('TWIN_COMPILE', '0')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'twin')]
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from t1d_twin import ode
from t1d_twin.vpatients import t1d_patient_class
T1DPatient = t1d_patient_class()
from simglucose.patient.t1dpatient import Action

names = ['adult#001', 'adult#006']  # Published virtual subjects, not people.
bases = [ode.base_patient(name) for name in names]
basal = np.array([base.basal_u_per_hr for base in bases])/60
reference = []
for index, name in enumerate(names):
    patient = T1DPatient.withName(name)
    trace = []
    for minute in range(360):
        carbs = 6.0 if 60 <= minute < 70 else 0.0
        insulin = basal[index] + (3.0 if 60 <= minute < 62 else 0.0)
        patient.step(Action(CHO=carbs, insulin=insulin))
        trace.append(patient.observation.Gsub)
    reference.append(trace)
reference = np.asarray(reference)[:, 1::2]
p, vg, x0 = ode.stack_params(bases)
carbs = torch.zeros(2, 180, dtype=torch.float64)
starts = torch.zeros(2, 180, dtype=torch.bool)
insulin = torch.tensor(basal)[:,None].repeat(1,180)
carbs[:,30:35], starts[:,30], insulin[:,30] = 6.0, True, insulin[:,30]+3.0
one = torch.ones(2,180,dtype=torch.float64)
with torch.no_grad():
    actual = ode.simulate(x0,p,vg,carbs,starts,insulin,one,one,one).numpy()
error = actual-reference
assert np.abs(error).max() < 2.0
result = {'syntheticOnly':True,'scope':'numerical solver agreement under matched virtual inputs',
          'virtualSubjects':2,'durationHours':6,'stepMinutes':2,
          'maximumAbsoluteDifferenceMgDl':float(np.abs(error).max()),
          'rmseDifferenceMgDl':float(np.sqrt(np.mean(error**2))),
          'acceptanceMaximumDifferenceMgDl':2.0,
          'interpretation':'Numerical implementation check. Real-person forecast and intervention validity require separate evaluation.'}
(ROOT/'examples').mkdir(exist_ok=True)
(ROOT/'assets').mkdir(exist_ok=True)
(ROOT/'examples/twin-numerical-check.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True,layout='constrained')
fig.patch.set_facecolor('#f6f9fd')
for i,ax in enumerate(axes):
    ax.set_facecolor('#f6f9fd')
    hours=np.arange(1,181)*2/60
    ax.plot(hours,reference[i],color='#d48a9d',linewidth=3,label='simglucose reference')
    ax.plot(hours,actual[i],color='#326e97',linewidth=1.7,linestyle='--',label='Differentiable twin core')
    ax.axvspan(1,70/60,color='#a8cce3',alpha=.25)
    ax.set_title(f'Virtual subject {i+1}',loc='left',fontweight='bold')
    ax.set_xlabel('Hours'); ax.grid(alpha=.14)
axes[0].set_ylabel('Glucose (mg/dL)'); axes[1].legend(frameon=False,fontsize=8)
fig.suptitle('Numerical agreement under matched synthetic inputs',fontsize=13,fontweight='bold')
fig.savefig(ROOT/'assets/twin-numerical-check.png',dpi=180)
plt.close(fig)
print(json.dumps(result))

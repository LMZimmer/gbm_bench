import numpy as np
from typing import Dict, List, Union, Tuple


def coverage(recurrence, plan):
     if np.sum(tumorRecurrence) <=  0.00001:
         return 1

     intersection = np.logical_and(tumorRecurrence, treatmentPlan)
     coverage = np.sum(intersection) / np.sum(tumorRecurrence)
     return coverage

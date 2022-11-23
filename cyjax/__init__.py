# Copyright 2022 Mathis Gerdes
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import util
from . import random
from . import polynomial
from . import donaldson
from . import ml

from .projective import index_hom_to_affine
from .projective import index_affine_to_hom
from .projective import change_chart
from .projective import hom_to_affine
from .projective import affine_to_hom
from .projective import fs_metric
from .projective import fs_potential

from .differential import induced_metric
from .differential import complex_hessian
from .differential import jacobian_embed

from .polynomial import Poly
from .polynomial import HomPoly

from .variety import VarietySingle
from .variety import Dwork
from .variety import Fermat

__version__ = '1.0.0'

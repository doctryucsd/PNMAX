# PATH Configuration for Simulator Modules

This snippet configures the import path so simulator modules can resolve the local `external`
dependencies from the repo root.

```python
from pnmax.paths import repo_root

PROJECT_ROOT = repo_root()  # honors the PNMAX_ROOT env override
EXTERNAL_PATH = PROJECT_ROOT / "external"

if str(EXTERNAL_PATH) not in sys.path:
    sys.path.append(str(EXTERNAL_PATH))

from unindp.tools import LEVEL, OPTYPE, SimConfig
```

# RNA secondary structure predictor
This predictor uses MLP (from two python deep learning libraries - mxnet and lasagne), genetic algorithm and MFT net to make predictions about RNA secondary structure.

## Dependencies
- **Python 3.5** (not tested on previous versions)
- **mxnet** and **Lasagne** (if you want to use Naïve Predictor)
- **numpy**
- **matplotlib**

## How to use it
Clone this repository, import one of the predictors and start to play! The simplest example of usage may look like this:
```python
from MFTPredictor import MFTPredictor
import rna

p = MFTPredictor()
result = p.predict(rna.Molecule("GGCCUGAGGAGACUCAGAAGCC"))
result.show()
```
For more examples check [examples.ipynb](examples.ipynb).

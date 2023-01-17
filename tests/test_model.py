import torch
from src.models.model import MyAwesomeModel
from tests import _PATH_DATA
import pytest

#N_train = train.data.shape
#train.data.shape
def test_model():
   assert 1==1

def test_error_on_wrong_shape():
   with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
      model = MyAwesomeModel()
      model(torch.randn(1,2,3))
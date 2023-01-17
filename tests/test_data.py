import torch
from src.data.make_dataset import CorruptMnist
from tests import _PATH_DATA
import os.path
import pytest

#N_train = train.data.shape
#train.data.shape
@pytest.mark.skipif(not os.path.exists(os.path.dirname(__file__)+"/../data/processed"), reason="Data files not found")
def test_data():
   dataset = CorruptMnist(train=True, in_folder=_PATH_DATA+"/raw", out_folder=_PATH_DATA+"/processed")
   dtest = CorruptMnist(train=False, in_folder=_PATH_DATA+"/raw", out_folder=_PATH_DATA+"/processed")
   #print(len(dataset))
   #print(dataset.data.shape)
   #print(len(torch.unique(dataset.targets)))
   assert len(dataset) == 25000, "Dataset did not have the correct number of samples"
   assert len(dtest) == 5000, "Testset did not have the correct number of samples"
   assert list(dataset.data.shape[1:]) == [1,28,28], "shape of input images is not correct"
   assert len(torch.unique(dataset.targets))  == 10, "the targets numbers are not matching the dataset"
   #assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
   #assert that all labels are represented

import torch
from src.data.make_dataset import CorruptMnist
from tests import _PATH_DATA

#N_train = train.data.shape
#train.data.shape
def test_data():

   dataset = CorruptMnist(train=True, in_folder="data/raw", out_folder="data/processed")
   dtest = CorruptMnist(train=False, in_folder="data/raw", out_folder="data/processed")
   print(len(dataset))
   assert len(dataset) == 25000 #for training and N_test for test
   assert len(dtest) == 5000
   #assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
   #assert that all labels are represented

def test_answer():
    assert func(3) == 5
import pandas as pd
import datetime
from sklearn.neural_network import MLPClassifier
import random
import torch
from net import gtnet
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
model = torch.load('./model/model_first_half.pt')
model.eval()
adj1 = model.gc(model.idx).cpu().detach().numpy()

model = torch.load('./model/model_last_half.pt')
model.eval()
adj2 = model.gc(model.idx).cpu().detach().numpy()
print(adj1.shape)
print(np.linalg.norm(adj1-adj2))
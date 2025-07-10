from BKT_Model import BKTModel
import pandas as pd

df = pd.read_csv("../Data/23-24-problem_logs.csv").head(50000)
bkt = BKTModel(n_iter=100)
bkt.fit(df)
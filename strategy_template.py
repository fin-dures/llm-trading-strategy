import pandas as pd


class Strategy:

    def generate_signals(self, df):


        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        return entries, exits
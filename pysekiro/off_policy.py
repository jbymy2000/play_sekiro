import os
import time

import pandas as pd

from pysekiro.Agent import Sekiro_Agent
from pysekiro.key_tools.get_keys import key_check

# ---*---

class Play_Sekiro_Offline:
    def __init__(
        self,
        lr,
        batch_size,
        load_memory_path,
        save_weights_path,
        load_weights_path=None
    ):
        self.sekiro_agent = Sekiro_Agent(
            lr         = lr,    # learning
            batch_size = batch_size,    # Number of samples drawn
            load_weights_path = load_weights_path,    # Specify the path to save the model weight. The default is None, do not save.
            save_weights_path = save_weights_path     # Specify the path to load the model weight. The default is None, do not load.
        )

        self.load_memory_path = load_memory_path     # Specify the path to load the memory/experience. The default is None, do not load.

        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.load_memory_path):
            last_time = time.time()
            self.sekiro_agent.replayer.memory = pd.read_json(self.load_memory_path)    # load the memory/experience from json file.
            print(f'Load {self.load_memory_path}. Took {round(time.time()-last_time, 3):>5} seconds.')

            self.sekiro_agent.replayer.count = self.sekiro_agent.replayer.memory.action.count()
        else:
            print('No memory to load.')

    def run(self):

        paused = True
        print("Ready!")

        while True:
            keys = key_check()
            if paused:
                if 'T' in keys:
                    paused = False
                    print('\nStarting!')
            else:    # After pressing'T', it will enter here immediately in the next round 
                self.sekiro_agent.learn(verbose=1)

                print(f'\r step:{self.sekiro_agent.step:>6}', end='')

                if 'P' in keys:
                	break

        self.sekiro_agent.save_evaluate_network()
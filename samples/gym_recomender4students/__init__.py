from gym.envs.registration import register

register(id='Recomender4Students-v0',
         entry_point='gym_recomender4students.envs:Recomender4StudentsEnv',
         )

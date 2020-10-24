from gym.envs.registration import register

register(id='Recommender4Students-v0',
         entry_point='gym_recommender4students.envs:Recommender4StudentsEnv',
         )

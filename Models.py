import random
import numpy as np

class Agent():
    def __init__(self, id, id_node, s, r ):
        self.id = id
        self.id_node = id_node
        self.s = s
        self.r = r

        self.t_death = 0
    
    def get_neighbors(self, model):

        for agent in model.agents_list:
            neighbors = len(model.node[self.id_node])
            return neighbors
    
    def calculate_crowding_stress(self, model):

        neighbors = self.get_neighbors(model)
        P_i = neighbors

        A_i = model.c * (P_i / (self.r if self.r > 0 else 1)) * (1 + self.s)
        return A_i
    
    def update_stress(self, model):
        A_i = self.calculate_crowding_stress(model)
        self.s = (self.s + A_i) * (1.0 - model.d)
        self.s = max(0.0, self.s)


    def reproduce(self, model):
        if np.random.rand() < (model.n / (1 + (self.s * model.k))):
            new_agent = Agent(model.id_counter, self.id_node, self.s, self.r)
            model.agents_list.append(new_agent)
            model.birth += 1
            model.id_counter += 1

    def die(self, model):
        if np.random.rand() < model.m:
            self.t_death = model.t
            model.death_list.append(self)
            model.death += 1
            return True
        return False   

class IndividualModel:
    def __init__(self, T, c, m, n, d, k, R, P_0, s_0, z):

        self.T = T
        self.c = c
        self.m = m
        self.n = n
        self.d = d
        self.k = k
        self.R = R
        self.P_0 = P_0
        self.s_0 = s_0
        self.z = z

        # states
        self.agents_list = []
        self.death_list = []
        self.node = {i: [] for i in range(1, self.z + 1)}
        self.S = 0
        self.P = 0
        self.id_counter = 0
        self.t = 0
        self.birth = 0
        self.death = 0

        # output
        self.S_ts = []
        self.P_ts = []
        self.birth_ts = []
        self.death_ts = []

    def setup(self):
        for i in range(self.P_0):
            r = ( self.R / self.z )
            node_id = random.randint(1,self.z)
            self.agents_list.append(Agent(i, node_id, self.s_0, r))
        self.id_counter = self.P_0
        self.compute_globals()
        self.S_ts = [self.S]
        self.P_ts = [self.P]
        self.birth_ts = [self.birth]
        self.death_ts = [self.death]

    def step(self):
        self.t += 1
        self.birth = 0
        self.death = 0
        self.update_nodes()
        
        agents_to_remove = []
        for a in self.agents_list:
            a.update_stress(self)
            a.reproduce(self)
            if a.die(self):
                agents_to_remove.append(a)
        for agent in agents_to_remove:
            self.agents_list.remove(agent)
            
        self.compute_globals()
    
    def update_nodes(self):
        for key in self.node.keys():
            self.node[key] = []
        for agent in self.agents_list:
            self.node[agent.id_node].append(agent)

    def compute_globals(self):
        self.P = len(self.agents_list)

        if self.P == 0:
            self.S = 0.0 
        else:
            self.S = sum(a.s for a in self.agents_list) / self.P

        self.S_ts.append(self.S)
        self.P_ts.append(self.P)
        self.birth_ts.append(self.birth)
        self.death_ts.append(self.death)

    def run(self):
        for _ in range(self.T):
            self.step()


def aggregate_model(T, c, m, n, d, k, R, P_0, S_0):

    P, S = P_0, S_0
    S_ts, P_ts = [S_0], [P_0]

    for t in range(T):
        A = c * ( P / R ) * (1 + S)
        S = ( S + A ) * ( 1 - d )
        P = P * ( 1 + (( n / ( 1 + ( S * k )) - m )))
        P = max(P, 0)
        S_ts.append(S)
        P_ts.append(P)

    return S_ts, P_ts 
"""
Main file to validate model assumptions.
"""

# import
import sys
import pickle
import numpy as np
from CMDP import CMDP
import scipy.stats as st
from itertools import product


def define_CMDP(name, params):
    """
    Define an MDP
    """
    # which one to define
    # ----------------- random -------------------------
    if "random" in name:
        n_states = int(params[0])
        n_actions = int(params[1])
        # states
        states = list(range(n_states))
        # actions
        actions = list(range(n_actions))
        # transition matrix
        trans_mat = {}
        for a in actions:
            trans_mat[a] = np.random.rand(len(states), len(states))
            # normalize
            trans_mat[a] = trans_mat[a] / trans_mat[a].sum(
                axis=1, keepdims=True
            )
        # reward matrix
        reward_mat = np.random.randint(
            low=0, high=100, size=(len(states), len(actions))
        )

        # transition function
        def trans_func(new_state, old_state, action):
            """transition function"""
            return trans_mat[action][int(old_state), int(new_state)]

        # reward function
        def reward_func(state, action):
            """reward function"""
            return reward_mat[int(state), int(action)]

        # initial distribution
        initial_distr = np.random.rand(len(states))
        # normalize
        initial_distr = initial_distr / initial_distr.sum(
            axis=0, keepdims=True
        )
        # set the length of D equal to |S|
        # 100, 100: 40000 -- 60000
        # 500, 500: 40000 -- 60000
        D = st.uniform.rvs(loc=65, scale=25, size=len(states))
        d = {}
        for i in range(len(D)):
            for s in states:
                for a in actions:
                    d[i, s, a] = st.uniform.rvs(loc=0, scale=1, size=1)[0]
        # discound factor
        discount_factor = 0.99
    # ----------------- queue --------------------------
    elif "queue" in name:
        """
        deFarias2003a:
        - state: number of jobs;
        - action: which rate of service to choose;
        - transition, reward: see paper
        """
        # parameters
        n_state = int(params[0])
        n_action = int(params[1] + 1)
        p = params[2]
        q = list(np.linspace(0, 1 - p, n_action))
        # states
        states = list(range(n_state))
        # actions
        actions = list(range(n_action))
        # transition matrix
        trans_mat = {}
        for a in actions:
            trans_mat[a] = np.zeros(shape=(len(states), len(states)))
            for i in range(trans_mat[a].shape[0]):
                # 0
                if i == 0:
                    trans_mat[a][i, 0] = 1 - p
                    trans_mat[a][i, 1] = p
                    continue
                # last
                elif i == trans_mat[a].shape[0] - 1:
                    trans_mat[a][i, i - 1] = q[a]
                    trans_mat[a][i, i] = 1 - q[a]
                    continue
                # middle
                else:
                    # trans_mat[a][i, i - 1] = q[a]
                    # trans_mat[a][i, i] = p
                    # trans_mat[a][i, i + 1] = 1 - (q[a] + p)
                    trans_mat[a][i, i - 1] = (1 - p) * q[a]
                    trans_mat[a][i, i] = (1 - p) * (1 - q[a]) + p * q[a]
                    trans_mat[a][i, i + 1] = p * (1 - q[a])
                    continue
        # reward matrix
        reward_mat = np.zeros(shape=(len(states), len(actions)))
        for i in range(reward_mat.shape[0]):
            for j in range(reward_mat.shape[1]):
                reward_mat[i, j] = -1 * states[i] - (
                    60 * q[j] * q[j] * q[j]
                )

        # transition function
        def trans_func(new_state, old_state, action):
            """transition function"""
            return trans_mat[action][int(old_state), int(new_state)]

        # reward function
        def reward_func(state, action):
            """reward function"""
            return reward_mat[int(state), int(action)]

        # initial distribution
        initial_distr = np.random.rand(len(states))
        # normalize
        initial_distr = initial_distr / initial_distr.sum(
            axis=0, keepdims=True
        )
        # set the length of D equal to |S|
        # 20, 20: 500 -- 600
        # 100, 100: 40000 -- 60000
        # 500, 500: 40000 -- 60000
        D = st.uniform.rvs(loc=50, scale=10, size=len(states))
        d = {}
        for i in range(len(D)):
            for s in states:
                for a in actions:
                    d[i, s, a] = st.uniform.rvs(loc=0, scale=1, size=1)[0]
        # discound factor
        discount_factor = 0.99
    # ----------------- bandit -------------------------
    elif "bandit" in name:
        """
        Bertsimas2016:
        - n_arm bandit machine, each has n_states states,
            total n_arm^n_states states;
        - transition drawn uniformly;
        - no action, no transition;
        """
        # parameters
        n_arm = int(params[0])
        n_states = int(params[1])
        # pr of each bandit at all state, increasing
        pr = {}
        for i in range(n_arm):
            pr[i] = np.sort(st.uniform.rvs(loc=0, scale=100, size=n_states))
            pr[i] = pr[i] / np.sum(pr[i])
        # states
        state_list = list(product(range(n_states), repeat=n_arm))
        states = list(range(len(state_list)))
        # actions
        actions = list(range(n_arm))
        # transition matrix
        trans_mat = {}
        for a in actions:
            # arm transit
            trans_arm = np.random.rand(n_states, n_states)
            # normalize
            trans_arm = trans_arm / trans_arm.sum(
                axis=1, keepdims=True
            )
            # matrix
            trans_mat[a] = np.zeros(shape=(len(states), len(states)))
            for i in range(trans_mat[a].shape[0]):
                for j in range(trans_mat[a].shape[1]):
                    if all([
                        state_list[i][:a] == state_list[j][:a],
                        state_list[i][a + 1:] == state_list[j][a + 1:]
                    ]):
                        trans_mat[a][i, j] = trans_arm[
                            state_list[i][a], state_list[j][a]
                        ]
        # reward matrix
        reward_mat = np.zeros(shape=(len(states), len(actions)))
        for i in range(reward_mat.shape[0]):
            for a in range(reward_mat.shape[1]):
                reward_mat[i, a] = (10 / n_states) * state_list[i][a]

        # transition function
        def trans_func(new_state, old_state, action):
            """transition function"""
            return trans_mat[action][int(old_state), int(new_state)]

        # reward function
        def reward_func(state, action):
            """reward function"""
            return reward_mat[int(state), int(action)]

        # initial distribution
        initial_distr = np.random.rand(len(states))
        # normalize
        initial_distr = initial_distr / initial_distr.sum(
            axis=0, keepdims=True
        )
        # set the length of D equal to |S|
        # 100, 100: 40000 -- 60000
        # 500, 500: 40000 -- 60000
        D = st.uniform.rvs(loc=65, scale=25, size=len(states))
        d = {}
        for i in range(len(D)):
            for s in states:
                for a in actions:
                    d[i, s, a] = st.uniform.rvs(loc=0, scale=1, size=1)[0]
        # discound factor
        discount_factor = 0.99
    # ----------------- inventory ----------------------
    elif "inventory" in name:
        """
        Lee2017, Puterman2014:
        - state: inventory of items;
        - action: how much to order;
        - transition, reward: based on stochastic demand;
        """
        # parameters
        n_inv = int(params[0]) + 1
        discount_factor = params[1]
        b = st.uniform.rvs(loc=10, scale=15, size=1)[0]
        K = st.uniform.rvs(loc=3, scale=5, size=1)[0]
        c = st.uniform.rvs(loc=5, scale=7, size=1)[0]
        h = st.uniform.rvs(loc=0.1, scale=0.2, size=1)[0]
        d = st.norm.rvs(loc=0.5 * n_inv, scale=0.1 * n_inv, size=1)[0]
        p = st.poisson.pmf(range(n_inv), mu=d)
        p = p / np.sum(p)
        q = []
        for i in range(len(p)):
            q.append(np.sum(p[i:]))
        # states
        states = list(range(n_inv))
        # actions
        actions = list(range(n_inv))
        # transition matrix
        trans_mat = {}
        for a in actions:
            # matrix
            trans_mat[a] = np.zeros(shape=(len(states), len(states)))
            for i in range(trans_mat[a].shape[0]):
                # overflow
                if states[i] + a >= n_inv:
                    trans_mat[a][i, len(states) - 1] = 1
                    continue
                # else, look at each possible future state
                for j in range(trans_mat[a].shape[1]):
                    # demand
                    demand = states[i] + a - states[j]
                    # demnad < 0
                    if demand < 0:
                        trans_mat[a][i, j] = 0
                    elif demand >= 0 and states[j] > 0:
                        trans_mat[a][i, j] = p[demand]
                    elif demand >= 0 and states[j] == 0:
                        trans_mat[a][i, j] = q[states[i] + a]
        # reward matrix
        reward_mat = np.zeros(shape=(len(states), len(actions)))
        for i in range(reward_mat.shape[0]):
            for a in range(reward_mat.shape[1]):
                # overflow
                if states[i] + actions[a] >= n_inv:
                    reward_mat[i, a] = -100000
                    continue
                # normal reward
                Oa = 0 if actions[a] == 0 else K + c * actions[a]
                reward_mat[i, a] = np.sum([
                    b * j * p[j]
                    for j in range(states[i] + actions[a] - 1)
                ]) - Oa - h * (states[i] + actions[a])

        # transition function
        def trans_func(new_state, old_state, action):
            """transition function"""
            return trans_mat[action][int(old_state), int(new_state)]

        # reward function
        def reward_func(state, action):
            """reward function"""
            return reward_mat[int(state), int(action)]

        # initial distribution
        initial_distr = np.random.rand(len(states) - 1)
        # normalize
        initial_distr = initial_distr / initial_distr.sum(
            axis=0, keepdims=True
        )
        initial_distr = [0] + list(initial_distr)
        # set the length of D equal to |S|
        # 100, 100: 40000 -- 60000
        # 500, 500: 40000 -- 60000
        D = st.uniform.rvs(loc=65, scale=15, size=len(states))
        d = {}
        for i in range(len(D)):
            for s in states:
                for a in actions:
                    d[i, s, a] = st.uniform.rvs(loc=0, scale=1, size=1)[0]
        # discound factor
        discount_factor = 0.99
    # ----------------- replace ------------------------
    elif "replace" in name:
        """
        Modified Puterman2014:
        - machine with states, 0: best state, higher the worse;
        - actions: which degree of maintenance;
        - transition: larger action has a higher pr to restore
            the state to 0;
        - reward: larger action has higher cost.
        """
        # parameters
        n_states = int(params[0])
        n_actions = int(params[1])
        # states
        states = list(range(n_states))
        # actions, 0: no maintenance
        actions = list(range(n_actions + 1))
        # -------- transition matrix ----------
        # deterorate pr
        p_det = np.zeros(shape=(len(states), len(states)))
        for i in range(p_det.shape[0]):
            mu = st.uniform.rvs(loc=0, scale=1, size=1)[0]
            pr = st.norm.pdf(
                states[i:], loc=states[i] - mu, scale=(i + 1)
            )
            pr = pr / np.sum(pr)
            for j in range(p_det.shape[1]):
                if j >= i:
                    p_det[i, j] = pr[j - i]
        # restore pr
        p_res = {}
        pr = np.sort(st.uniform.rvs(loc=0.05, scale=0.85, size=n_actions))
        # st.uniform.rvs(loc=1, scale=n_actions, size=n_actions)
        # pr = pr / np.sum(pr)
        for a in range(1, n_actions + 1):
            p_res[a] = pr[a - 1]
        # print(p_res)
        # exit()
        # transition
        trans_mat = {}
        for a in actions:
            trans_mat[a] = np.zeros(shape=(len(states), len(states)))
            # no maintenance
            if a == 0:
                for i in range(trans_mat[a].shape[0]):
                    for j in range(trans_mat[a].shape[1]):
                        trans_mat[a][i, j] = p_det[i, j]
            # maintenance
            else:
                for i in range(trans_mat[a].shape[0]):
                    for j in range(trans_mat[a].shape[1]):
                        # success
                        if j < i:
                            trans_mat[a][i, j] = p_res[a] * p_det[0, j]
                        # no idea
                        else:
                            trans_mat[a][i, j] = np.sum([
                                # success
                                p_res[a] * p_det[0, j],
                                # fail
                                (1 - p_res[a]) * p_det[i, j]
                            ])
        # ---------- reward matrix -----------
        C_rew = 0.5 * len(states)
        C_rep = 0.1 * len(states) * np.array(actions)
        C_opr = np.sort(st.uniform.rvs(
            loc=0, scale=0.75 * len(states), size=len(states)
        ))
        reward_mat = np.zeros(shape=(len(states), len(actions)))
        for i in range(reward_mat.shape[0]):
            for j in range(reward_mat.shape[1]):
                reward_mat[i, j] = C_rew - C_rep[j] - C_opr[i]

        # transition function
        def trans_func(new_state, old_state, action):
            """transition function"""
            return trans_mat[action][int(old_state), int(new_state)]

        # reward function
        def reward_func(state, action):
            """reward function"""
            return reward_mat[int(state), int(action)]

        # initial distribution
        initial_distr = [1 / n_states] * n_states
        # set the length of D equal to |S|
        # 100, 100: 40000 -- 60000
        # 500, 500: 40000 -- 60000
        D = st.uniform.rvs(loc=45, scale=15, size=len(states))
        d = {}
        for i in range(len(D)):
            for s in states:
                for a in actions:
                    d[i, s, a] = st.uniform.rvs(loc=0, scale=1, size=1)[0]
        # discound factor
        discount_factor = 0.99
    # ----------------- transmission -------------------
    elif "transmission" in name:
        """
        Modified Krishnamurthy2016:
        - state: how many packages left and current channel;
        - action: transmit a package with which efforts;
        - transition: larger action has a higher chance of
            successful transmission;
        - reward: holding cost of packages, larger action
            has a higher transmission cost.
        """
        # parameters
        n_channel = int(params[0])
        n_package = int(params[1])
        n_action = int(params[2])
        # states
        state_list = []
        for j in range(n_channel):
            for k in range(n_package + 1):
                state_list.append((j, k))
        states = list(range(len(state_list)))
        # actions
        actions = list(range(n_action + 1))
        # success rate
        mu = np.sort(
            st.uniform.rvs(loc=0, scale=1, size=n_action)
        )
        p = {}
        for a in range(1, n_action + 1):
            p[a] = st.norm.pdf(
                range(n_channel), loc=n_channel - 1, scale=1 / mu[a - 1]
            )
            p[a] = p[a] / np.sum(p[a])
        # transmission cost
        c_t = -1 * np.sort(
            np.abs(st.uniform.rvs(loc=5, scale=15, size=n_action))
        )
        # hold cost
        c_h = -1 * np.abs(st.uniform.rvs(loc=0, scale=5, size=1)[0])
        # transition matrix
        trans_mat = {}
        # channel transmission
        channel_trans = np.random.rand(n_channel, n_channel)
        # normalize
        channel_trans = channel_trans / channel_trans.sum(
            axis=1, keepdims=True
        )
        for a in actions:
            trans_mat[a] = np.zeros(shape=(len(states), len(states)))
            # not transimitting
            if a == 0:
                for i in range(trans_mat[a].shape[0]):
                    # channel
                    for j in range(trans_mat[a].shape[1]):
                        if all([
                            # remaining item
                            state_list[i][1] == state_list[j][1]
                        ]):
                            trans_mat[a][i, j] = channel_trans[
                                state_list[i][0], state_list[j][0]
                            ]
            # transmitting
            else:
                for i in range(trans_mat[a].shape[0]):
                    for j in range(trans_mat[a].shape[1]):
                        # no package left
                        if all([
                            # remaining item
                            state_list[i][1] == 0,
                            state_list[i][1] == state_list[j][1]
                        ]):
                            trans_mat[a][i, j] = channel_trans[
                                state_list[i][0], state_list[j][0]
                            ]
                        # fail
                        elif all([
                            # remaining item
                            state_list[i][1] == state_list[j][1]
                        ]):
                            trans_mat[a][i, j] = channel_trans[
                                state_list[i][0], state_list[j][0]
                            ] * (1 - p[a][state_list[i][0]])
                        # success
                        elif all([
                            # remaining item
                            state_list[i][1] - 1 == state_list[j][1]
                        ]):
                            trans_mat[a][i, j] = channel_trans[
                                state_list[i][0], state_list[j][0]
                            ] * p[a][state_list[i][0]]
        # reward matrix
        reward_mat = np.zeros(shape=(len(states), len(actions)))
        for i in range(reward_mat.shape[0]):
            for j in range(reward_mat.shape[1]):
                if actions[j] == 0:
                    reward_mat[i, j] = state_list[i][1] * c_h
                else:
                    reward_mat[i, j] = np.sum([
                        # package cost
                        state_list[i][1] * c_h,
                        # transmit cost
                        c_t[j - 1]
                    ])

        # transition function
        def trans_func(new_state, old_state, action):
            """transition function"""
            return trans_mat[action][int(old_state), int(new_state)]

        # reward function
        def reward_func(state, action):
            """reward function"""
            return reward_mat[int(state), int(action)]

        # initial distribution
        initial_distr = np.random.rand(len(states))
        # normalize
        initial_distr = initial_distr / initial_distr.sum(
            axis=0, keepdims=True
        )
        # set the length of D equal to |S|
        # 100, 100: 40000 -- 60000
        # 500, 500: 40000 -- 60000
        D = st.uniform.rvs(loc=65, scale=15, size=len(states))
        d = {}
        for i in range(len(D)):
            for s in states:
                for a in actions:
                    d[i, s, a] = st.uniform.rvs(loc=0, scale=1, size=1)[0]
        # discound factor
        discount_factor = 0.99
    # ================ model ===================
    model = CMDP(
        name=name,
        states=states,
        actions=actions,
        trans_func=trans_func,
        reward_func=reward_func,
        initial_distr=initial_distr,
        discount_factor=discount_factor,
        d=d, D=D
    )
    return model


def benchmarking(instance, params):
    """
    Benchmarking of PDPI
    """
    # name
    instance_name = instance
    for s in params:
        instance_name = instance_name + "-{}".format(s)
    # statistics
    statistics = {}
    for i in range(10):
        print(i)
        name = instance_name + "-{}".format(i)
        # MDP model
        CMDP_model = define_CMDP(name, params)
        # LP benchamark
        statistics[i, 'LP'] = CMDP_model.LP(sol_dir="results")
        # LP dual benchamark
        statistics[i, 'LPD'] = CMDP_model.LP_dual(sol_dir="results")
        # primal-dual policy iteration
        statistics[i, 'PDPI'] = CMDP_model.PDPI(sol_dir="results")
    # save
    pickle.dump(statistics, open(
        'results/statistics/{}.pickle'.format(instance_name), 'wb'
    ))
    return


def main(instance, **kwargs):
    """main"""
    np.random.seed(1)
    # benchmarking
    benchmarking(instance, params=tuple(
        float(arg) for arg in kwargs['params']
    ))
    return


if __name__ == "__main__":
    main(instance=sys.argv[1], params=sys.argv[2:])

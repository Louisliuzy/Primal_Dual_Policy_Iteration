"""
Main file to validate model assumptions.
"""

# import
import sys
import pickle
import numpy as np
from CMDP import CMDP
import scipy.stats as st


def define_CMDP(name, params, load="None"):
    """
    Define an MDP
    """
    # which one to define
    # ----------------- replace ------------------------
    if "replace" in name:
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
        # C_rew = 0.5 * len(states)
        # C_rep = 0.1 * len(states) * np.array(actions)
        # C_opr = np.sort(st.uniform.rvs(
        #     loc=0, scale=0.75 * len(states), size=len(states)
        # ))
        C_rew = 10
        C_rep = 2 * np.array(actions)
        C_opr = np.sort(st.uniform.rvs(
            loc=0, scale=1.75, size=len(states)
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
        # set the length of D equal to |S|, loc=55, scale=15
        D = st.uniform.rvs(loc=55, scale=15, size=len(states))
        d = {}
        for i in range(len(D)):
            baseline = np.array(
                [st.uniform.rvs(loc=0.1, scale=0.3, size=1)[0]] * len(actions)
            )
            increment = np.flip(np.sort(
                st.uniform.rvs(loc=0.01, scale=0.15, size=len(actions))
            ))
            for s in states:
                for a in actions:
                    if s == 0:
                        d[i, s, a] = baseline[a]
                    else:
                        d[i, s, a] = d[i, s - 1, a] + increment[a]
        # discound factor
        discount_factor = 0.99
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
        CMDP_model = define_CMDP(name, params, load="None")
        # LPD
        # statistics[i, 'LDP'] = CMDP_model.LP_dual(sol_dir="results")
        # PDPI
        statistics[i, 'PDPI'] = CMDP_model.PDPI(sol_dir="results")
        # PDPI Monotone
        statistics[i, 'PDPI-MOP'] = CMDP_model.PDPI_monotone(
          sol_dir="results"
        )
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

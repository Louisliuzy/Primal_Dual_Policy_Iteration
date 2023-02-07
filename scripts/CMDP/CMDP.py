"""
MDP problem definition as a class
with value iteration algorithm
"""

# import
import time
import numpy as np
import gurobipy as grb


class CMDP:
    """MDP problem class with constraints"""

    def __init__(
        self, name, states, actions, trans_func,
        reward_func, initial_distr, discount_factor,
        d, D
    ):
        """
        `name`: str, name of the MDP;
        `states`: list, states;
        `actions`: list, actions;
        `trans_func`: function, the transition function,
            input: (new_state, old_state, action), output: pr;
        `reward_func`: function, the reward function,
            input: (state, action), output: number;
        `initial_distr`: list, initial distribution of states;
        `discount_factor`: numeric, discount factor, < 1;
        `d`: dict, (i, |S|, |A|), coefficient of dual variable,
            i is the number of constraints;
        `D`: list, ith element is the rhs of the ith constraint.
        """
        super().__init__()
        self.name = name
        self.states = states
        self.actions = actions
        self.trans_func = trans_func
        self.reward_func = reward_func
        self.alpha = initial_distr
        self.gamma = discount_factor
        self.d, self.D = d, D
        # PDPI variables
        self.var_theta, self.var_lambda, self.var_y = {}, {}, {}
        self.MP = float("nan")
        # monotone
        self.policy = {s: 0 for s in self.states}

    # linear programming, dual
    def LP_dual(self, sol_dir='None'):
        """
        Solving using linear programming, dual formulation
        """
        # time
        run_time = time.time()
        solve_time = 0
        # encoding state and action to dict
        state_dict = {
            i: self.states[i]
            for i in range(len(self.states))
        }
        action_dict = {
            i: self.actions[i]
            for i in range(len(self.actions))
        }
        # Gurobi model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        model.setParam("IntFeasTol", 1e-9)
        # the model pay the highest attention to numeric coherency.
        model.setParam("NumericFocus", 3)
        model.setParam("DualReductions", 0)
        # ----------------------- Variables -------------------------
        # policy, pr
        var_x = {}
        for s in state_dict.keys():
            for a in action_dict.keys():
                var_x[s, a] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name="x_{}_{}".format(s, a)
                )
        model.update()
        # objective
        objective = grb.quicksum([
            self.reward_func(self.states[s], self.actions[a]) * var_x[s, a]
            for s in state_dict.keys()
            for a in action_dict.keys()
        ])
        model.setObjective(objective, grb.GRB.MAXIMIZE)
        # ---------------------- Constraints ------------------------
        # the MDP constraint
        for s in state_dict.keys():
            model.addLConstr(
                lhs=grb.quicksum([
                    grb.quicksum([
                        var_x[s, a]
                        for a in action_dict.keys()
                    ]),
                    -1 * grb.quicksum([
                        self.gamma * self.trans_func(
                            self.states[s], self.states[s_old], self.actions[a]
                        ) * var_x[s_old, a]
                        for s_old in state_dict.keys()
                        for a in action_dict.keys()
                    ])
                ]),
                sense=grb.GRB.EQUAL,
                rhs=self.alpha[s],
                name="constr_{}".format(s)
            )
        model.update()
        # other constraints
        for i in range(len(self.D)):
            model.addLConstr(
                lhs=grb.quicksum([
                    self.d[i, s, a] * var_x[s, a]
                    for s in state_dict.keys()
                    for a in action_dict.keys()
                ]),
                sense=grb.GRB.LESS_EQUAL,
                rhs=self.D[i],
                name="dD_{}".format(s)
            )
        model.update()
        # ------------------------ Solving --------------------------
        temp_time = time.time()
        model.optimize()
        solve_time = solve_time + (time.time() - temp_time)
        # check status
        if model.status != grb.GRB.OPTIMAL:
            raise ValueError(
                "Model not optimal! Status code: {}".format(model.status)
            )
        # Finding the best policy
        policy, value = {}, {}
        for s in state_dict.keys():
            # the chosen action
            action_ind = np.argmax([
                var_x[s, a].X for a in action_dict.keys()
            ])
            # policy
            policy[self.states[s]] = self.actions[action_ind]
            # value
            value[self.states[s]] = model.getConstrByName(
                "constr_{}".format(s)
            ).Pi
        run_time = time.time() - run_time
        # gap
        gap = 0 if model.IsMIP == 0 else model.MIPGap
        # ------------------------- Output --------------------------
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}-LPD.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write(
                "Total algorithm run time: {} seconds;\n".format(run_time)
            )
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(model.ObjVal))
            file.write("Gap: {};\n".format(gap))
            for state in self.states:
                file.write("{}: {}\n".format(state, value[state]))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for state in self.states:
                file.write("{}: {}\n".format(state, policy[state]))
            file.write("==============================\n")
            file.close()
        return {
            'obj': model.ObjVal, 'runtime': run_time, 'policy': policy
        }

    # linear programming, primal
    def LP(self, sol_dir='None'):
        """
        Solving using linear programming, primal formulation
        """
        # time
        run_time = time.time()
        solve_time = 0
        # encoding state and action to dict
        state_dict = {
            i: self.states[i]
            for i in range(len(self.states))
        }
        action_dict = {
            i: self.actions[i]
            for i in range(len(self.actions))
        }
        # Gurobi model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        model.setParam("IntFeasTol", 1e-9)
        # the model pay the highest attention to numeric coherency.
        model.setParam("NumericFocus", 3)
        model.setParam("DualReductions", 0)
        # ----------------------- Variables -------------------------
        # value, v
        var_v = {}
        for s in state_dict.keys():
            var_v[s] = model.addVar(
                lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="v_{}".format(s)
            )
        model.update()
        # lambda, additional constraints
        var_lambda = {}
        for i in range(len(self.D)):
            var_lambda[i] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="lambda_{}".format(i)
            )
        model.update()
        # objective
        objective = grb.quicksum([
            grb.quicksum([
                self.alpha[s] * var_v[s]
                for s in state_dict.keys()
            ]),
            grb.quicksum([
                self.D[i] * var_lambda[i]
                for i in range(len(self.D))
            ])
        ])
        model.setObjective(objective, grb.GRB.MINIMIZE)
        # ---------------------- Constraints ------------------------
        # the only constraint
        for s in state_dict.keys():
            for a in action_dict.keys():
                model.addLConstr(
                    lhs=grb.quicksum([
                        var_v[s],
                        -1 * grb.quicksum([
                            self.gamma * self.trans_func(
                                self.states[s_new], self.states[s],
                                self.actions[a]
                            ) * var_v[s_new]
                            for s_new in state_dict.keys()
                        ]),
                        grb.quicksum([
                            self.d[i, s, a] * var_lambda[i]
                            for i in range(len(self.D))
                        ])
                    ]),
                    sense=grb.GRB.GREATER_EQUAL,
                    rhs=self.reward_func(self.states[s], self.actions[a]),
                    name="constr_{}_{}".format(s, a)
                )
        model.update()
        # ------------------------ Solving --------------------------
        # model.write("model/{}-LP.lp".format(self.name))
        temp_time = time.time()
        model.optimize()
        solve_time = solve_time + (time.time() - temp_time)
        # check status
        if model.status != grb.GRB.OPTIMAL:
            raise ValueError(
                "Model not optimal! Status code: {}".format(model.status)
            )
        # Finding the best policy
        policy, value = {}, {}
        for s in state_dict.keys():
            # the chosen action
            action_ind = np.argmax([
                model.getConstrByName(
                    "constr_{}_{}".format(s, a)
                ).Pi
                for a in action_dict.keys()
            ])
            # policy
            policy[self.states[s]] = self.actions[action_ind]
            # value
            value[self.states[s]] = var_v[s].X
        run_time = time.time() - run_time
        # gap
        gap = 0 if model.IsMIP == 0 else model.MIPGap
        # ------------------------- Output --------------------------
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}-LP.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write(
                "Total algorithm run time: {} seconds;\n".format(run_time)
            )
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(model.ObjVal))
            file.write("Gap: {};\n".format(gap))
            for state in self.states:
                file.write("{}: {}\n".format(state, value[state]))
            file.write("==============================\n")
            for i in range(len(self.D)):
                file.write("lambda {}: {}\n".format(i, var_lambda[i].X))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for state in self.states:
                file.write("{}: {}\n".format(state, policy[state]))
            file.write("==============================\n")
            file.close()
        return {
            'obj': model.ObjVal, 'runtime': run_time, 'policy': policy
        }

    # primal-dual policy iteration, current version
    def PDPI(self, epsilon=0.001, sol_dir='None'):
        """
        Policy iteration for CMDP
        """
        run_time = time.time()
        # reset policy
        self.policy = {s: 0 for s in self.states}
        # MP
        self.MP = self.__define_MP()
        # DP
        self.DP = {}
        for s in self.states:
            self.DP[s] = self.__define_DP(s)
        # start iteration
        iteration, optimal = 0, False
        while not optimal:
            # solve MP
            self.MP.optimize()
            # variable values
            theta_val, lambda_val = {
                s: self.var_theta[s].X
                for s in self.states
            }, {
                i: self.var_lambda[i].X
                for i in range(len(self.D))
            }
            # control parameters
            optimal = True
            # loop states
            for s in self.states:
                # modify DP and solve
                self.__modify_DP(s, theta_val, lambda_val)
                self.DP[s].optimize()
                # policy
                self.policy[s] = self.actions[
                    np.argmax([self.var_y[s, a].X for a in self.actions])
                ]
                # condition
                if np.abs(
                    self.var_theta[s].X - self.DP[s].ObjVal
                ) > epsilon:
                    optimal = False
                    # add optimality cut
                    self.MP.addLConstr(
                        lhs=self.var_theta[s],
                        sense=grb.GRB.GREATER_EQUAL,
                        rhs=grb.quicksum([
                            self.reward_func(s, self.policy[s]),
                            grb.quicksum([
                                self.gamma * self.trans_func(
                                    s_n, s, self.policy[s]
                                ) * self.var_theta[s_n]
                                for s_n in self.states
                            ]),
                            -grb.quicksum([
                                self.d[i, s, self.policy[s]]
                                * self.var_lambda[i]
                                for i in range(len(self.D))
                            ])
                        ])
                    )
            iteration += 1
        # time
        run_time = time.time() - run_time
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}-PDPI.txt".format(sol_dir, self.name), mode="w+")
            file.write("==============================\n")
            file.write(
                "Total algorithm runtime: {} seconds;\n".format(run_time)
            )
            file.write("Iterations: {};\n".format(iteration))
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(self.MP.ObjVal))
            for s in self.states:
                file.write("{}: {};\n".format(s, self.var_theta[s].X))
            file.write("==============================\n")
            for i in range(len(self.D)):
                file.write("lambda {}: {}\n".format(i, self.var_lambda[i].X))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for s in self.states:
                file.write("{}: {}\n".format(s, self.policy[s]))
            file.write("==============================\n")
            file.close()
        return {
            'obj': self.MP.ObjVal, 'runtime': run_time, 'policy': self.policy,
            'K': iteration
        }

    # primal-dual policy iteration, current version
    def PDPI_monotone(self, epsilon=0.001, sol_dir='None'):
        """
        Policy iteration for CMDP
        """
        run_time = time.time()
        # reset policy
        self.policy = {s: 0 for s in self.states}
        # MP
        self.MP = self.__define_MP()
        # DP
        self.DP = {}
        # start iteration
        iteration, optimal = 0, False
        while not optimal:
            # solve MP
            self.MP.optimize()
            # control parameters
            optimal = True
            # loop states
            for s in self.states:
                # modify DP and solve
                self.DP[s] = self.__define_DP_monotone(s)
                self.DP[s].optimize()
                # policy
                if s == self.states[0]:
                    candidate_actions = self.actions
                else:
                    candidate_actions = self.actions[
                        self.actions.index(self.policy[s - 1]):
                    ]
                self.policy[s] = candidate_actions[np.argmax([
                    self.var_y[s, a].X
                    for a in candidate_actions
                ])]
                # condition
                if np.abs(
                    self.var_theta[s].X - self.DP[s].ObjVal
                ) > epsilon:
                    optimal = False
                    # add optimality cut
                    self.MP.addLConstr(
                        lhs=self.var_theta[s],
                        sense=grb.GRB.GREATER_EQUAL,
                        rhs=grb.quicksum([
                            self.reward_func(s, self.policy[s]),
                            grb.quicksum([
                                self.gamma * self.trans_func(
                                    s_n, s, self.policy[s]
                                ) * self.var_theta[s_n]
                                for s_n in self.states
                            ]),
                            -grb.quicksum([
                                self.d[i, s, self.policy[s]]
                                * self.var_lambda[i]
                                for i in range(len(self.D))
                            ])
                        ])
                    )
            iteration += 1
        # time
        run_time = time.time() - run_time
        # print solution to file
        if sol_dir == "None":
            pass
        else:
            file = open("{}/{}-PDPI-MOP.txt".format(
                sol_dir, self.name
            ), mode="w+")
            file.write("==============================\n")
            file.write(
                "Total algorithm runtime: {} seconds;\n".format(run_time)
            )
            file.write("Iterations: {};\n".format(iteration))
            file.write("==============================\n")
            file.write("Optimal Value:\n")
            file.write("Total value: {};\n".format(self.MP.ObjVal))
            for s in self.states:
                file.write("{}: {}\n".format(s, self.var_theta[s].X))
            file.write("==============================\n")
            for i in range(len(self.D)):
                file.write("lambda {}: {}\n".format(i, self.var_lambda[i].X))
            file.write("==============================\n")
            file.write("Optimal policy:\n")
            for s in self.states:
                file.write("{}: {}\n".format(s, self.policy[s]))
            file.write("==============================\n")
            file.close()
        return {
            'obj': self.MP.ObjVal, 'runtime': run_time, 'policy': self.policy,
            'K': iteration
        }

    def __define_MP(self):
        """
        Define the MP.
        """
        # Gurobi model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        # model.setParam("IntFeasTol", 1e-3)
        model.setParam("NumericFocus", 3)
        model.setParam("DualReductions", 0)
        # ----------------------- Variables -------------------------
        # theta
        for s in self.states:
            self.var_theta[s] = model.addVar(
                lb=-1e8, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS, name="theta_{}".format(s)
            )
        model.update()
        # theta
        for i in range(len(self.D)):
            self.var_lambda[i] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS, name="lambda_{}".format(i)
            )
        model.update()
        # objective
        objective = grb.quicksum([
            grb.quicksum([
                self.alpha[s] * self.var_theta[s]
                for s in self.states
            ]),
            grb.quicksum([
                self.var_lambda[i] * self.D[i]
                for i in range(len(self.D))
            ])
        ])
        model.setObjective(objective, grb.GRB.MINIMIZE)
        # ---------------------- Constraints ------------------------
        model.update()
        return model

    def __define_DP(self, s):
        """
        define SP
        """
        # Gurobi model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        # model.setParam("IntFeasTol", 1e-3)
        model.setParam("NumericFocus", 3)
        model.setParam("DualReductions", 0)
        # ----------------------- Variables -------------------------
        # policy, pr
        for a in self.actions:
            self.var_y[s, a] = model.addVar(
                lb=0, ub=grb.GRB.INFINITY,
                vtype=grb.GRB.CONTINUOUS,
                name="y_{}_{}".format(s, a)
            )
        model.update()
        # objective
        objective = grb.quicksum([
            grb.quicksum([
                self.reward_func(s, a),
                grb.quicksum([
                    self.gamma * self.trans_func(s_n, s, a) * -1e8
                    for s_n in self.states
                ])
            ]) * self.var_y[s, a]
            for a in self.actions
        ])
        model.setObjective(objective, grb.GRB.MAXIMIZE)
        # ---------------------- Constraints ------------------------
        # the only constraint
        model.addLConstr(
            lhs=grb.quicksum([
                self.var_y[s, a] for a in self.actions
            ]),
            sense=grb.GRB.EQUAL,
            rhs=1
        )
        model.update()
        return model

    def __modify_DP(self, s, theta_val, lambda_val):
        """
        modify SP
        """
        # find variables and modify value
        for a in self.actions:
            self.var_y[s, a].setAttr(
                "Obj", np.sum([
                    self.reward_func(s, a),
                    np.sum([
                        self.gamma * self.trans_func(s_n, s, a)
                        * theta_val[s_n]
                        for s_n in self.states
                    ]),
                    -np.sum([
                        self.d[i, s, a] * lambda_val[i]
                        for i in range(len(self.D))
                    ])
                ])
            )
        self.DP[s].update()
        return

    def __define_DP_monotone(self, s):
        """
        define SP
        """
        # Gurobi model
        model = grb.Model()
        model.setParam("OutputFlag", False)
        # model.setParam("IntFeasTol", 1e-3)
        model.setParam("NumericFocus", 3)
        model.setParam("DualReductions", 0)
        # first state
        if s == self.states[0]:
            # variables
            for a in self.actions:
                self.var_y[s, a] = model.addVar(
                    lb=0, ub=grb.GRB.INFINITY,
                    vtype=grb.GRB.CONTINUOUS,
                    name="y_{}_{}".format(s, a)
                )
            model.update()
            # objective
            objective = grb.quicksum([
                np.sum([
                    self.reward_func(s, a),
                    np.sum([
                        self.gamma * self.trans_func(s_n, s, a)
                        * self.var_theta[s_n].X
                        for s_n in self.states
                    ]),
                    -np.sum([
                        self.d[i, s, a] * self.var_lambda[i].X
                        for i in range(len(self.D))
                    ])
                ]) * self.var_y[s, a]
                for a in self.actions
            ])
            model.setObjective(objective, grb.GRB.MAXIMIZE)
            # the only constraint
            model.addLConstr(
                lhs=grb.quicksum([
                    self.var_y[s, a]
                    for a in self.actions
                ]),
                sense=grb.GRB.EQUAL,
                rhs=1
            )
            model.update()
        # following states
        else:
            for a in self.actions:
                if a >= self.policy[s - 1]:
                    self.var_y[s, a] = model.addVar(
                        lb=0, ub=grb.GRB.INFINITY,
                        vtype=grb.GRB.CONTINUOUS,
                        name="y_{}_{}".format(s, a)
                    )
            model.update()
            # objective
            objective = grb.quicksum([
                np.sum([
                    self.reward_func(s, a),
                    np.sum([
                        self.gamma * self.trans_func(s_n, s, a)
                        * self.var_theta[s_n].X
                        for s_n in self.states
                    ]),
                    -np.sum([
                        self.d[i, s, a] * self.var_lambda[i].X
                        for i in range(len(self.D))
                    ])
                ]) * self.var_y[s, a]
                for a in self.actions if a >= self.policy[s - 1]
            ])
            model.setObjective(objective, grb.GRB.MAXIMIZE)
            # the only constraint
            model.addLConstr(
                lhs=grb.quicksum([
                    self.var_y[s, a]
                    for a in self.actions if a >= self.policy[s - 1]
                ]),
                sense=grb.GRB.EQUAL,
                rhs=1
            )
            model.update()
        return model

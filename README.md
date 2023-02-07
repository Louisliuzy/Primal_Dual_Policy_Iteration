# A Primal-dual Policy Iteration Algorithm for Constrained Markov Decision Processes

This archive is distributed under the [MIT License](LICENSE).

The source code and data in this repository are a snapshot of the software and data
that were used in the research reported on in the manuscript under review 
[A Primal-dual Policy Iteration Algorithm for Constrained Markov Decision Processes](https://www.researchgate.net/publication/353295872_A_Primal-dual_Policy_Iteration_Algorithm_for_Constrained_Markov_Decision_Processes) by Z. Liu et. al. The data generated for this study are included with the codes.

## Cite

To cite this repository, please cite the [manuscript](https://www.researchgate.net/publication/353295872_A_Primal-dual_Policy_Iteration_Algorithm_for_Constrained_Markov_Decision_Processes).

Below is the BibTex for citing the manuscript.

```
@article{Liu2023,
  title={A Primal-dual Policy Iteration Algorithm for Constrained Markov Decision Processes},
  author={Liu, Zeyu and Li, Xueping and Khojandi, Anahita},
  journal={Preprint},
  year={2023},
  url={https://www.researchgate.net/publication/353295872_A_Primal-dual_Policy_Iteration_Algorithm_for_Constrained_Markov_Decision_Processes}
}
```

## Description

The goal of this repository is to solve Constrained Markov Decision Processes (CMDP) with a novel policy iteration algorithm that exploits the primal-dual relationships. The proposed primal-dual policy iteration algorithm (PDPI) evaluates CMDP policies through a primal master problem and improves CMDP policies through dual subproblems. Please refer to the manuscript for further details.

The data used in this study include a randomly generated MDP problem and five other problems from the literature:
1. A queueing problem [[1]](#1);
2. A bandit machine problem [[2]](#2);
3. An inventory management problem [[3]](#3);
4. A machine maintenance problem [[4]](#4);
5. A data transmission problem [[5]](#5).


## Python Prerequisite

The following Python libraries are required to run the source codes:
1. `numpy`;
2. `scipy`;
3. `pickle`;
4. `gurobipy`.

## Usage

### CMDP

Run `python main.py instance param1 param2 ...` in the `scripts/CMDP` folder. Instances can be set up using the provided variables `instance` and `param`. All six problems are available, i.e., `instance` should be one of `random`, `queue`, `bandit`, `inventory`, `replace`, or `transmission`. The following describes the parameters for each problem:
1. `random`: `param1 = number of states`, `param2 = number of actions`;
2. `queue`: `param1 = length of queue`, `param2 = number of service modes`, `param3 = arrival probability`;
3. `bandit`: `param1 = number of arms`, `param2 = number of states for each arm`;
4. `inventory`: `param1 = inventory capacity`, `param2 = discount factor`;
5. `replace`: `param1 = number of states of the machine`, `param2 = number of maintenance options`;
6. `transmission`: `param1 = number of channels`, `param2 = number of data packages`, `param3 = number of transmission modes`.

### CMDP with Monotone Optimal Policy

Run `python main.py instance param1 param2 ...` in the `scripts/CMDP-MOP` folder. Instances can be set up using the provided variables `instance` and `params`. Only `replace` is available.


## Support

For support in using this software, submit an
[issue](https://github.com/Louisliuzy/Primal_Dual_Policy_Iteration/issues/new).

## Reference
<a id="1">[1]</a> de Farias DP, Van Roy B (2003) The linear programming approach to approximate dynamic programming. Operations research 51(6):850-865.

<a id="2">[2]</a> Bertsimas D, Misic VV (2016) Decomposable markov decision processes: A fluid optimization approach. Operations Research 64(6):1537-1555.

<a id="3">[3]</a> Lee I, Epelman MA, Romeijn HE, Smith RL (2017) Simplex algorithm for countable-state discounted markov decision processes. Operations Research 65(4):1029-1042.

<a id="4">[4]</a> Puterman ML (2014) Markov decision processes: discrete stochastic dynamic programming (John Wiley & Sons).

<a id="5">[5]</a> Krishnamurthy V (2016) Partially observed Markov decision processes (Cambridge university press).
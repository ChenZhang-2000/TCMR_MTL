from collections import defaultdict

import torch
from torch.optim import Adam

from .min_norm_solvers import MinNormSolver, gradient_normalizers


def return_zero(*args, **kwargs):
    return 0.


class LeafNode:
    def __init__(self, loss_funcs: dict):
        self.loss_funcs = loss_funcs


class TreeNode:
    def __init__(self, children, parameters: [torch.Tensor], norm_type="loss+"):
        self.children = children
        self.parameters = parameters
        self.gradients = {}
        self.alphas = []
        self.norm_type = norm_type

    def zero_grad(self):
        self.gradients = {}
        for child in self.children:
            if isinstance(child, TreeNode):
                child.zero_grad()
            elif isinstance(child, LeafNode):
                pass
            else:
                raise

    def get_gradient(self, name):
        for child in self.children:
            if isinstance(child, TreeNode):
                child.get_gradient(name)
            elif isinstance(child, LeafNode):
                pass
            else:
                raise
        for i, params in enumerate(self.parameters):
            self.gradients[name] = [[], []]
            for p_name, param in params.named_parameters():
                # if p_name == "smpl.betas":
                #     print(p_name, param, param.grad)
                if param.grad is not None and param.requires_grad:
                    self.gradients[name][0].append(p_name)
                    self.gradients[name][1].append(param.grad.detach().clone())

    def run_solver(self, loss_data, norm_type, replace_grad=True):
        self.alphas = []
        # n = len(self.parameters)
        names = self.gradients.keys()
        m = len(names)
        # for i, gradients in enumerate(self.gradients):

        gn = gradient_normalizers({name: self.gradients[name][1] for name in names},
                                  loss_data,
                                  norm_type)

        for t in names:
            for gr_i in range(len(self.gradients[t][1])):
                # print(self.gradients[t][1][gr_i])
                self.gradients[t][1][gr_i] = self.gradients[t][1][gr_i] / gn[t]

        sol, _ = MinNormSolver.find_min_norm_element([[grad for grad in self.gradients[name][1]] for name in names])

        grad_dict = defaultdict(return_zero)
        for j, name in enumerate(names):
            alpha = sol[j]
            self.alphas.append((name, alpha))
            for p_name, grad in zip(*self.gradients[name]):
                # if p_name == "smpl.betas":
                #     print(p_name, type(grad))
                grad_dict[p_name] += alpha * grad

        if replace_grad:
            grad_dict = dict(grad_dict)
            for i, params in enumerate(self.parameters):
                for p_name, param in self.parameters[i].named_parameters():
                    # if p_name == "smpl.betas":
                    #     print(p_name, param, param.grad)
                    # print(p_name, grad_dict[p_name])
                    if p_name in grad_dict.keys():
                        param.grad = grad_dict[p_name]  # * m
            # print(param_dict)
            # param_dict = self.parameters[i].state_dict(keep_vars=True)
            # for p_name, grad in grad_dict.items():
            #     param_dict[p_name] = grad
            #     print(param_dict[p_name])

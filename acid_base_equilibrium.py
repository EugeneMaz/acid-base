from IPython.display import display, Latex
from joblib import Parallel, delayed
from sympy import Eq, symbols, sqrt, solve, nsolve, latex
import matplotlib.pyplot as plt
import numpy as np


class AcidBaseEquilibrium:
    def __init__(self, pK_values, H_OH_model='Truesdell-Jones', include_water=True):
        self.I, self.Kw, self.C_t, self.pH, self.I_add = symbols(
            r'I, K_W, C_{total}, pH, I_{add}', positive=True, real=True
        )
        self.H, self.OH, self.y_H, self.y_OH = symbols(
            r'[H^{+}], [OH^{-}], y_{H^+}, y_{OH^-}', positive=True, real=True
        )
        self.pK_values = pK_values
        self.include_water = include_water
        self.num_pK = len(pK_values)

        if H_OH_model == 'Davies':
            self.y_H_expr = 10 ** (-0.5115 * ((sqrt(self.I) / (1 + sqrt(self.I))) - 0.3 * self.I))
            self.y_OH_expr = 10 ** (-0.5115 * ((sqrt(self.I) / (1 + sqrt(self.I))) - 0.3 * self.I))
        elif H_OH_model == 'Truesdell-Jones':
            self.y_H_expr = 10 ** (-0.5115 * (sqrt(self.I) / (1 + 3.2914 * 0.478 * sqrt(self.I))) + 0.24 * self.I)
            self.y_OH_expr = 10 ** (-0.5115 * (sqrt(self.I) / (1 + 3.2914 * 1.065 * sqrt(self.I))) + 0.21 * self.I)
        else:
            raise ValueError("Unsupported H_OH_model. Supported models are 'Davies' and 'Truesdell-Jones'")

        self.Ks = []
        self.ys = {}
        self.ys_expr = {}
        self.species = {}
        for i, pK in enumerate(pK_values):
            self.Ks.append(symbols(fr'K_{i+1}', positive=True, real=True))
            self.species[i] = symbols(
                fr'[H_{{{self.num_pK-i if (self.num_pK-i) != 1 else ""}}}'
                fr'B^{{{i if i not in [0, 1] else ""}{"-" if i != 0 else ""}}}]',
                positive=True, real=True
            )
            self.ys[i] = symbols(
                fr'y_{{H_{{{self.num_pK-i if (self.num_pK-i) != 1 else ""}}}'
                fr'B^{{{i if i not in [0, 1] else ""}{"-" if i != 0 else ""}}}}}',
                positive=True, real=True
            )
            self.ys_expr[i] = 10 ** (-0.5115 * i ** 2 * ((sqrt(self.I) / (1 + sqrt(self.I))) - 0.3 * self.I))

        self.species[self.num_pK] = symbols(
            fr'[B^{{{self.num_pK if self.num_pK != 1 else ""}-}}]', positive=True, real=True
        )
        self.ys[self.num_pK] = symbols(
            fr'y_{{B^{{{self.num_pK if self.num_pK != 1 else ""}-}}}}', positive=True, real=True
        )
        self.ys_expr[self.num_pK] = 10 ** (
            -0.5115 * self.num_pK ** 2 * ((sqrt(self.I) / (1 + sqrt(self.I))) - 0.3 * self.I)
        )

        display(Latex(r"Expression for the ionic force according to $I=\frac{1}{2} \sum_{i}{c_i z_i^2}$ is:"))
        self.I_expr = 0.5 * (
            sum(self.species[i] * i ** 2 for i in self.species)
            + sum(self.species[i] * i for i in self.species)
        ) + self.I_add
        display(Eq(self.I, self.I_expr))

        display(Latex(r"The Davies activity coefficients for the species according to $\log \gamma_i=-A z_i^2\left(\frac{\sqrt{I}}{1+\sqrt{I}}-0.3 I\right)$ are:"))
        for i in range(self.num_pK + 1):
            display(Eq(self.ys[i], self.ys_expr[i]))

        display(Latex("We will solve the following system of equations:"))
        self.K_exprs = {}
        for i in range(self.num_pK):
            equation = (self.H * self.species[i + 1] / self.species[i]) * (self.y_H * self.ys[i + 1] / self.ys[i])
            self.K_exprs[i] = Eq(self.Ks[i], equation)
            display(self.K_exprs[i])

        if self.include_water:
            self.K_exprs[self.num_pK] = Eq(self.Kw, self.H * self.OH * self.y_H * self.y_OH)
            display(self.K_exprs[self.num_pK])

        self.mass_balance_expr = Eq(sum(self.species[i] for i in self.species), self.C_t)
        display(self.mass_balance_expr)

        self.init_subs = {
            self.H: 10 ** (-self.pH) / self.y_H,
            self.y_OH: self.y_OH_expr,
        }
        self.init_subs.update({self.ys[i]: self.ys_expr[i] for i in self.ys})
        self.init_subs.update({self.Ks[i]: 10 ** (-self.pK_values[i]) for i in range(self.num_pK)})
        self.init_subs.update({self.Kw: 10 ** (-14)})
        self.init_subs.update({self.y_H: self.y_H_expr})

        self.system = [eq.subs(self.init_subs) for eq in self.K_exprs.values()]
        self.system.append(self.mass_balance_expr.subs(self.init_subs))
        self.dummy_system = [eq.subs(self.I, 0) for eq in self.system]
        self.full_system = [eq.subs(self.I, self.I_expr) for eq in self.system]

    def solve_dummy(self, C_tot, pH0, verbose=False):
        """Solve the dummy system without ionic strength for initial estimates."""
        dummy_system_complete = [eq.subs({self.C_t: C_tot, self.pH: pH0}) for eq in self.dummy_system]
        self.dummy_solution = solve(dummy_system_complete, dict=True)[0]
        if verbose:
            display(self.dummy_solution)

    def solve_single(self, C_tot, pH0, I_add=0, verbose=True):
        """Solve the system of equations for a single set of parameters."""
        self.solve_dummy(C_tot, pH0)
        system_complete = [eq.subs(self.I_add, I_add).subs({self.C_t: C_tot, self.pH: pH0}) for eq in self.full_system]
        vars_to_solve = tuple(self.species[i] for i in self.species)
        if self.include_water:
            vars_to_solve += (self.OH,)
        solution = nsolve(
            system_complete,
            vars_to_solve,
            tuple(self.dummy_solution.get(var) for var in vars_to_solve),
            dict=True,
            prec=25,
        )[0]

        ion_str = self.I_expr.subs(self.C_t, C_tot).subs(self.I_add, I_add).subs(solution)
        solution.update({self.H: 10.0 ** (-pH0) / self.y_H_expr.subs(self.I, ion_str)})
        if verbose:
            print(f"Ionic strength is {ion_str:.3e} M")
            for k, v in solution.items():
                display(Latex(fr"The concentration of ${latex(k)}$ is: {v:.3e} M"))
        else:
            return solution

    def solve_range(self, C_tot=[0.01, 0.02, 0.05], pH0=[4, 5, 6], I_add=[0], parallel=False, verbose=False):
        """Solve the system of equations for a range of parameters."""
        self.C_range = np.asarray(C_tot)
        self.pH_range = np.asarray(pH0)
        self.I_add_range = np.asarray(I_add)
        n_jobs = -1 if parallel else 1
        result = Parallel(n_jobs=n_jobs)(
            delayed(self.solve_wrapper)(c, pH, I_add)
            for I_add in self.I_add_range
            for c in self.C_range
            for pH in self.pH_range
        )
        result = np.asarray(result)
        if verbose:
            solution_keys = self.solve_single(self.C_range[0], self.pH_range[0], I_add=0, verbose=False).keys()
            fig, axes = plt.subplots(nrows=self.C_range.size, ncols=1, figsize=(16, 12))
            axes = np.atleast_1d(axes)
            for i, cc in enumerate(self.C_range):
                for sp in range(result.shape[1] - 3):
                    axes[i].plot(result[result[:, 1] == cc, 2], result[result[:, 1] == cc, sp + 3])
                    axes[i].set_title(f"C_tot = {float(cc):.3g} M")
                    axes[i].set_xlabel("pH")
                    axes[i].set_ylabel("Concentration (M)")
                    axes[i].legend([fr"${latex(k)}$" for k in solution_keys])
            plt.show()
        self.result_range = result
        return self.result_range

    def solve_wrapper(self, c, pH, I_add):
        solution = self.solve_single(c, pH, I_add, verbose=False)
        solution_list = [solution[k] for k in solution]
        return [I_add, c, pH, *solution_list]

    def save_range(self, filename, header=""):
        """Save the results of the range calculation to a CSV file."""
        header += (
            "pKa: "
            + ", ".join(str(pKa) for pKa in self.pK_values)
            + "\n"
            + "I_add; C_tot (M); pH; "
            + "; ".join(
                latex(k)
                for k in self.solve_single(self.C_range[0], self.pH_range[0], I_add=0, verbose=False).keys()
            )
        )
        np.savetxt(filename, self.result_range, delimiter=";", header=header, comments="% ")

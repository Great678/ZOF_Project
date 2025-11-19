#!/usr/bin/env python3
"""
ZOF_CLI.py
Zero Of Functions (ZOF) Solver - CLI
Supports: Bisection, Regula Falsi, Secant, Newton-Raphson, Fixed Point, Modified Secant

Usage:
    python ZOF_CLI.py
It will launch an interactive menu asking for function, method and parameters.

Notes:
- You may enter functions like: x**3 - 2*x - 5
- For Fixed Point method, provide g(x) such that x = g(x).
- Tolerance is used for |f(x)| (or |x_new - x_old| depending on method).
"""
import sys
import math
from typing import Callable, Tuple, List, Any
try:
    import sympy as sp
except Exception as e:
    print("Sympy not found. Please install requirements from requirements.txt (sympy).")
    raise

# ----- Utility: parse function strings to callable -----
def make_function(func_str: str) -> Tuple[Callable[[float], float], Any]:
    """
    Returns (f, sympy_expr)
    """
    x = sp.symbols('x')
    expr = sp.sympify(func_str)
    f = sp.lambdify(x, expr, modules=["math"])
    return f, expr

def derivative_of(expr):
    x = sp.symbols('x')
    dexpr = sp.diff(expr, x)
    return sp.lambdify(x, dexpr, modules=["math"]), dexpr

# ----- Methods implementations (return iteration list and final info) -----
def bisection(f, a: float, b: float, tol: float, max_iter: int):
    iters = []
    fa = f(a); fb = f(b)
    if fa * fb > 0:
        raise ValueError("Function has same sign at interval endpoints. Bisection requires sign change.")
    for k in range(1, max_iter+1):
        c = (a + b) / 2.0
        fc = f(c)
        # error estimate: half interval width
        err = abs(b - a) / 2.0
        iters.append((k, a, b, c, fa, fb, fc, err))
        if abs(fc) < tol or err < tol:
            return iters, c, fc, err, k
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return iters, c, fc, err, k

def regula_falsi(f, a: float, b: float, tol: float, max_iter: int):
    iters = []
    fa = f(a); fb = f(b)
    if fa * fb > 0:
        raise ValueError("Function has same sign at interval endpoints. Regula Falsi requires sign change.")
    c = a
    for k in range(1, max_iter+1):
        c_prev = c
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        err = abs(c - c_prev) if k > 1 else abs(b - a)
        iters.append((k, a, b, c, fa, fb, fc, err))
        if abs(fc) < tol or err < tol:
            return iters, c, fc, err, k
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return iters, c, fc, err, k

def secant(f, x0: float, x1: float, tol: float, max_iter: int):
    iters = []
    f0 = f(x0); f1 = f(x1)
    for k in range(1, max_iter+1):
        if (f1 - f0) == 0:
            raise ZeroDivisionError("Denominator became zero in Secant method.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        f2 = f(x2)
        err = abs(x2 - x1)
        iters.append((k, x0, x1, x2, f0, f1, f2, err))
        if abs(f2) < tol or err < tol:
            return iters, x2, f2, err, k
        x0, f0 = x1, f1
        x1, f1 = x2, f2
    return iters, x2, f2, err, k

def newton_raphson(f, fprime_callable, x0: float, tol: float, max_iter: int):
    iters = []
    x = x0
    fx = f(x)
    for k in range(1, max_iter+1):
        fpx = fprime_callable(x)
        if fpx == 0:
            raise ZeroDivisionError("Derivative zero during Newton-Raphson.")
        x_new = x - fx / fpx
        fx_new = f(x_new)
        err = abs(x_new - x)
        iters.append((k, x, x_new, fx, fpx, fx_new, err))
        if abs(fx_new) < tol or err < tol:
            return iters, x_new, fx_new, err, k
        x, fx = x_new, fx_new
    return iters, x_new, fx_new, err, k

def fixed_point_iteration(g, x0: float, tol: float, max_iter: int):
    iters = []
    x = x0
    for k in range(1, max_iter+1):
        x_new = g(x)
        err = abs(x_new - x)
        fx_new = None  # Not known unless user also supplies f
        iters.append((k, x, x_new, err))
        if err < tol:
            return iters, x_new, err, k
        x = x_new
    return iters, x_new, err, k

def modified_secant(f, x0: float, delta: float, tol: float, max_iter: int):
    iters = []
    x = x0
    for k in range(1, max_iter+1):
        denom = f(x + delta * x) - f(x)
        if denom == 0:
            raise ZeroDivisionError("Denominator zero in Modified Secant.")
        x_new = x - (delta * x) * f(x) / denom
        f_new = f(x_new)
        err = abs(x_new - x)
        iters.append((k, x, x_new, f(x), f(x + delta * x), f_new, err))
        if abs(f_new) < tol or err < tol:
            return iters, x_new, f_new, err, k
        x = x_new
    return iters, x_new, f_new, err, k

# ----- CLI helpers -----
def float_input(prompt, default=None):
    while True:
        try:
            s = input(prompt).strip()
            if s == "" and default is not None:
                return default
            return float(s)
        except ValueError:
            print("Enter a valid number.")

def int_input(prompt, default=None):
    while True:
        try:
            s = input(prompt).strip()
            if s == "" and default is not None:
                return default
            return int(s)
        except ValueError:
            print("Enter an integer.")

def main_menu():
    print("\nZOF Solver - CLI")
    print("Choose method:")
    print("1. Bisection")
    print("2. Regula Falsi (False Position)")
    print("3. Secant")
    print("4. Newton-Raphson")
    print("5. Fixed Point Iteration")
    print("6. Modified Secant")
    print("0. Exit")
    choice = input("Enter choice (0-6): ").strip()
    return choice

def print_bisection_table(iters):
    print(f"{'k':>3} {'a':>12} {'b':>12} {'c':>12} {'f(a)':>12} {'f(b)':>12} {'f(c)':>12} {'err':>12}")
    for row in iters:
        k,a,b,c,fa,fb,fc,err = row
        print(f"{k:3d} {a:12.6g} {b:12.6g} {c:12.6g} {fa:12.6g} {fb:12.6g} {fc:12.6g} {err:12.6g}")

def print_regula_table(iters):
    print_bisection_table(iters)

def print_secant_table(iters):
    print(f"{'k':>3} {'x_{k-2}':>12} {'x_{k-1}':>12} {'x_k':>12} {'f(x_{k-2})':>12} {'f(x_{k-1})':>12} {'f(x_k)':>12} {'err':>12}")
    for row in iters:
        k,x0,x1,x2,f0,f1,f2,err = row
        print(f"{k:3d} {x0:12.6g} {x1:12.6g} {x2:12.6g} {f0:12.6g} {f1:12.6g} {f2:12.6g} {err:12.6g}")

def print_newton_table(iters):
    print(f"{'k':>3} {'x':>12} {'x_new':>12} {'f(x)':>12} {'f\'(x)':>12} {'f(x_new)':>12} {'err':>12}")
    for row in iters:
        k,x,x_new,fx,fpx,fx_new,err = row
        print(f"{k:3d} {x:12.6g} {x_new:12.6g} {fx:12.6g} {fpx:12.6g} {fx_new:12.6g} {err:12.6g}")

def print_fixed_table(iters):
    print(f"{'k':>3} {'x':>12} {'g(x)':>12} {'err':>12}")
    for row in iters:
        k,x,x_new,err = row
        print(f"{k:3d} {x:12.6g} {x_new:12.6g} {err:12.6g}")

def print_modified_secant_table(iters):
    print(f"{'k':>3} {'x':>12} {'x_new':>12} {'f(x)':>12} {'f(x+dx)':>12} {'f(x_new)':>12} {'err':>12}")
    for row in iters:
        k,x,x_new,fx,fxdx,fn,err = row
        print(f"{k:3d} {x:12.6g} {x_new:12.6g} {fx:12.6g} {fxdx:12.6g} {fn:12.6g} {err:12.6g}")

def run_cli():
    while True:
        choice = main_menu()
        if choice == '0':
            print("Goodbye.")
            break
        if choice not in [str(i) for i in range(1,7)]:
            print("Invalid choice.")
            continue

        func_str = input("Enter f(x) (e.g. x**3 - 2*x - 5): ").strip()
        try:
            f, expr = make_function(func_str)
        except Exception as e:
            print("Error parsing function:", e)
            continue

        tol = float_input("Tolerance (default 1e-6): ", default=1e-6)
        max_iter = int_input("Max iterations (default 50): ", default=50)

        try:
            if choice == '1':  # Bisection
                a = float_input("Left endpoint a: ")
                b = float_input("Right endpoint b: ")
                iters, root, fval, err, nit = bisection(f,a,b,tol,max_iter)
                print_bisection_table(iters)
                print(f"\nEstimated root: {root}\nf(root): {fval}\nFinal error estimate: {err}\nIterations: {nit}")

            elif choice == '2':  # Regula Falsi
                a = float_input("Left endpoint a: ")
                b = float_input("Right endpoint b: ")
                iters, root, fval, err, nit = regula_falsi(f,a,b,tol,max_iter)
                print_regula_table(iters)
                print(f"\nEstimated root: {root}\nf(root): {fval}\nFinal error estimate: {err}\nIterations: {nit}")

            elif choice == '3':  # Secant
                x0 = float_input("x0: ")
                x1 = float_input("x1: ")
                iters, root, fval, err, nit = secant(f,x0,x1,tol,max_iter)
                print_secant_table(iters)
                print(f"\nEstimated root: {root}\nf(root): {fval}\nFinal error estimate: {err}\nIterations: {nit}")

            elif choice == '4':  # Newton-Raphson
                fprime_callable, dexpr = derivative_of(expr)
                x0 = float_input("Initial guess x0: ")
                iters, root, fval, err, nit = newton_raphson(f, fprime_callable, x0, tol, max_iter)
                print_newton_table(iters)
                print(f"\nEstimated root: {root}\nf(root): {fval}\nFinal error estimate: {err}\nIterations: {nit}")

            elif choice == '5':  # Fixed Point
                g_str = input("Enter g(x) for x = g(x) (required): ").strip()
                try:
                    g, gexpr = make_function(g_str)
                except Exception as e:
                    print("Error parsing g(x):", e)
                    continue
                x0 = float_input("Initial guess x0: ")
                iters, root, err, nit = fixed_point_iteration(g,x0,tol,max_iter)
                print_fixed_table(iters)
                print(f"\nEstimated fixed point: {root}\nFinal error estimate: {err}\nIterations: {nit}")

            elif choice == '6':  # Modified Secant
                x0 = float_input("Initial guess x0: ")
                delta = float_input("Delta (relative perturbation, e.g. 1e-3): ", default=1e-3)
                iters, root, fval, err, nit = modified_secant(f,x0,delta,tol,max_iter)
                print_modified_secant_table(iters)
                print(f"\nEstimated root: {root}\nf(root): {fval}\nFinal error estimate: {err}\nIterations: {nit}")

        except Exception as e:
            print("Method failed with error:", e)

if __name__ == "__main__":
    run_cli()

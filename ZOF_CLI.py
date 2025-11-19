#!/usr/bin/env python3
"""
Zero of Functions (ZOF) Solver - Command Line Interface
Implements six numerical methods for finding roots of nonlinear equations
"""

import math
import sympy as sp
from typing import Callable, List, Dict, Tuple


class ZOFSolver:
    """Class implementing six root-finding numerical methods"""
    
    def __init__(self):
        self.x = sp.Symbol('x')
        self.iteration_data = []
    
    def parse_function(self, func_str: str) -> Callable:
        """Parse string function to callable"""
        try:
            expr = sp.sympify(func_str)
            func = sp.lambdify(self.x, expr, 'math')
            return func, expr
        except Exception as e:
            raise ValueError(f"Error parsing function: {e}")
    
    def bisection_method(self, func_str: str, a: float, b: float, 
                        tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """
        Bisection Method: Repeatedly bisects interval and selects subinterval 
        where sign change occurs
        """
        func, expr = self.parse_function(func_str)
        self.iteration_data = []
        
        fa = func(a)
        fb = func(b)
        
        if fa * fb > 0:
            return {"error": "Function must have opposite signs at endpoints"}
        
        for i in range(1, max_iter + 1):
            c = (a + b) / 2
            fc = func(c)
            error = abs(b - a) / 2
            
            self.iteration_data.append({
                "iteration": i,
                "a": a,
                "b": b,
                "c": c,
                "f(c)": fc,
                "error": error
            })
            
            if abs(fc) < tol or error < tol:
                return {
                    "root": c,
                    "iterations": i,
                    "final_error": error,
                    "iteration_data": self.iteration_data,
                    "method": "Bisection Method"
                }
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        return {
            "root": c,
            "iterations": max_iter,
            "final_error": error,
            "iteration_data": self.iteration_data,
            "method": "Bisection Method",
            "warning": "Maximum iterations reached"
        }
    
    def regula_falsi_method(self, func_str: str, a: float, b: float, 
                           tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """
        Regula Falsi (False Position) Method: Uses weighted average based on 
        function values instead of midpoint
        """
        func, expr = self.parse_function(func_str)
        self.iteration_data = []
        
        fa = func(a)
        fb = func(b)
        
        if fa * fb > 0:
            return {"error": "Function must have opposite signs at endpoints"}
        
        c_old = a
        
        for i in range(1, max_iter + 1):
            c = b - (fb * (b - a)) / (fb - fa)
            fc = func(c)
            
            if i > 1:
                error = abs(c - c_old)
            else:
                error = abs(b - a)
            
            self.iteration_data.append({
                "iteration": i,
                "a": a,
                "b": b,
                "c": c,
                "f(c)": fc,
                "error": error
            })
            
            if abs(fc) < tol or error < tol:
                return {
                    "root": c,
                    "iterations": i,
                    "final_error": error,
                    "iteration_data": self.iteration_data,
                    "method": "Regula Falsi Method"
                }
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            c_old = c
        
        return {
            "root": c,
            "iterations": max_iter,
            "final_error": error,
            "iteration_data": self.iteration_data,
            "method": "Regula Falsi Method",
            "warning": "Maximum iterations reached"
        }
    
    def secant_method(self, func_str: str, x0: float, x1: float, 
                     tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """
        Secant Method: Uses secant line through two points to approximate root
        """
        func, expr = self.parse_function(func_str)
        self.iteration_data = []
        
        f0 = func(x0)
        f1 = func(x1)
        
        for i in range(1, max_iter + 1):
            if abs(f1 - f0) < 1e-12:
                return {"error": "Division by zero: f(x1) - f(x0) too small"}
            
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            f2 = func(x2)
            error = abs(x2 - x1)
            
            self.iteration_data.append({
                "iteration": i,
                "x0": x0,
                "x1": x1,
                "x2": x2,
                "f(x2)": f2,
                "error": error
            })
            
            if abs(f2) < tol or error < tol:
                return {
                    "root": x2,
                    "iterations": i,
                    "final_error": error,
                    "iteration_data": self.iteration_data,
                    "method": "Secant Method"
                }
            
            x0, f0 = x1, f1
            x1, f1 = x2, f2
        
        return {
            "root": x2,
            "iterations": max_iter,
            "final_error": error,
            "iteration_data": self.iteration_data,
            "method": "Secant Method",
            "warning": "Maximum iterations reached"
        }
    
    def newton_raphson_method(self, func_str: str, x0: float, 
                             tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """
        Newton-Raphson Method: Uses tangent line at point to approximate root
        Requires derivative
        """
        func, expr = self.parse_function(func_str)
        self.iteration_data = []
        
        # Calculate derivative
        derivative_expr = sp.diff(expr, self.x)
        derivative = sp.lambdify(self.x, derivative_expr, 'math')
        
        x = x0
        
        for i in range(1, max_iter + 1):
            fx = func(x)
            dfx = derivative(x)
            
            if abs(dfx) < 1e-12:
                return {"error": "Derivative too small, division by zero"}
            
            x_new = x - fx / dfx
            error = abs(x_new - x)
            
            self.iteration_data.append({
                "iteration": i,
                "x": x,
                "f(x)": fx,
                "f'(x)": dfx,
                "x_new": x_new,
                "error": error
            })
            
            if abs(func(x_new)) < tol or error < tol:
                return {
                    "root": x_new,
                    "iterations": i,
                    "final_error": error,
                    "iteration_data": self.iteration_data,
                    "method": "Newton-Raphson Method"
                }
            
            x = x_new
        
        return {
            "root": x,
            "iterations": max_iter,
            "final_error": error,
            "iteration_data": self.iteration_data,
            "method": "Newton-Raphson Method",
            "warning": "Maximum iterations reached"
        }
    
    def fixed_point_iteration(self, func_str: str, x0: float, 
                             tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """
        Fixed Point Iteration: Rearrange f(x) = 0 to x = g(x) and iterate
        User must provide function in form x = g(x)
        """
        func, expr = self.parse_function(func_str)
        self.iteration_data = []
        
        x = x0
        
        for i in range(1, max_iter + 1):
            x_new = func(x)
            error = abs(x_new - x)
            
            self.iteration_data.append({
                "iteration": i,
                "x": x,
                "g(x)": x_new,
                "error": error
            })
            
            if error < tol:
                return {
                    "root": x_new,
                    "iterations": i,
                    "final_error": error,
                    "iteration_data": self.iteration_data,
                    "method": "Fixed Point Iteration"
                }
            
            # Check for divergence
            if abs(x_new) > 1e10:
                return {"error": "Method diverging, |x| > 10^10"}
            
            x = x_new
        
        return {
            "root": x,
            "iterations": max_iter,
            "final_error": error,
            "iteration_data": self.iteration_data,
            "method": "Fixed Point Iteration",
            "warning": "Maximum iterations reached"
        }
    
    def modified_secant_method(self, func_str: str, x0: float, delta: float = 0.01, 
                               tol: float = 1e-6, max_iter: int = 100) -> Dict:
        """
        Modified Secant Method: Uses perturbation to approximate derivative
        """
        func, expr = self.parse_function(func_str)
        self.iteration_data = []
        
        x = x0
        
        for i in range(1, max_iter + 1):
            fx = func(x)
            fx_delta = func(x + delta * x) if x != 0 else func(x + delta)
            
            denominator = fx_delta - fx
            if abs(denominator) < 1e-12:
                return {"error": "Division by zero: f(x + δx) - f(x) too small"}
            
            x_new = x - fx * (delta * x if x != 0 else delta) / denominator
            error = abs(x_new - x)
            
            self.iteration_data.append({
                "iteration": i,
                "x": x,
                "f(x)": fx,
                "f(x+δx)": fx_delta,
                "x_new": x_new,
                "error": error
            })
            
            if abs(func(x_new)) < tol or error < tol:
                return {
                    "root": x_new,
                    "iterations": i,
                    "final_error": error,
                    "iteration_data": self.iteration_data,
                    "method": "Modified Secant Method"
                }
            
            x = x_new
        
        return {
            "root": x,
            "iterations": max_iter,
            "final_error": error,
            "iteration_data": self.iteration_data,
            "method": "Modified Secant Method",
            "warning": "Maximum iterations reached"
        }


def display_results(result: Dict):
    """Display results in a formatted way"""
    print("\n" + "="*70)
    print(f"Method: {result.get('method', 'Unknown')}")
    print("="*70)
    
    if "error" in result:
        print(f"\nError: {result['error']}")
        return
    
    print("\nIteration Details:")
    print("-"*70)
    
    # Display iteration data
    if result.get('iteration_data'):
        first_iter = result['iteration_data'][0]
        headers = list(first_iter.keys())
        
        # Print headers
        header_line = " | ".join(f"{h:>12}" for h in headers)
        print(header_line)
        print("-"*70)
        
        # Print data
        for data in result['iteration_data']:
            values = []
            for key in headers:
                val = data[key]
                if isinstance(val, (int,)):
                    values.append(f"{val:>12}")
                elif isinstance(val, (float,)):
                    values.append(f"{val:>12.6e}")
                else:
                    values.append(f"{str(val):>12}")
            print(" | ".join(values))
    
    print("-"*70)
    print(f"\nFinal Root: {result['root']:.10f}")
    print(f"Final Error: {result['final_error']:.6e}")
    print(f"Iterations: {result['iterations']}")
    
    if "warning" in result:
        print(f"\nWarning: {result['warning']}")
    
    print("="*70 + "\n")


def main():
    """Main CLI function"""
    solver = ZOFSolver()
    
    print("\n" + "="*70)
    print(" "*15 + "ZERO OF FUNCTIONS (ZOF) SOLVER")
    print("="*70)
    
    methods = {
        "1": "Bisection Method",
        "2": "Regula Falsi Method",
        "3": "Secant Method",
        "4": "Newton-Raphson Method",
        "5": "Fixed Point Iteration",
        "6": "Modified Secant Method"
    }
    
    print("\nAvailable Methods:")
    for key, method in methods.items():
        print(f"  {key}. {method}")
    
    choice = input("\nSelect method (1-6): ").strip()
    
    if choice not in methods:
        print("Invalid choice!")
        return
    
    print(f"\nYou selected: {methods[choice]}")
    print("\nNote: Enter functions using Python syntax")
    print("Examples: x**2 - 4, sin(x) - x/2, exp(x) - 3*x")
    
    func_str = input("\nEnter function f(x): ").strip()
    
    try:
        if choice == "1":  # Bisection
            a = float(input("Enter left endpoint (a): "))
            b = float(input("Enter right endpoint (b): "))
            tol = float(input("Enter tolerance (default 1e-6): ") or 1e-6)
            max_iter = int(input("Enter max iterations (default 100): ") or 100)
            result = solver.bisection_method(func_str, a, b, tol, max_iter)
        
        elif choice == "2":  # Regula Falsi
            a = float(input("Enter left endpoint (a): "))
            b = float(input("Enter right endpoint (b): "))
            tol = float(input("Enter tolerance (default 1e-6): ") or 1e-6)
            max_iter = int(input("Enter max iterations (default 100): ") or 100)
            result = solver.regula_falsi_method(func_str, a, b, tol, max_iter)
        
        elif choice == "3":  # Secant
            x0 = float(input("Enter first initial guess (x0): "))
            x1 = float(input("Enter second initial guess (x1): "))
            tol = float(input("Enter tolerance (default 1e-6): ") or 1e-6)
            max_iter = int(input("Enter max iterations (default 100): ") or 100)
            result = solver.secant_method(func_str, x0, x1, tol, max_iter)
        
        elif choice == "4":  # Newton-Raphson
            x0 = float(input("Enter initial guess (x0): "))
            tol = float(input("Enter tolerance (default 1e-6): ") or 1e-6)
            max_iter = int(input("Enter max iterations (default 100): ") or 100)
            result = solver.newton_raphson_method(func_str, x0, tol, max_iter)
        
        elif choice == "5":  # Fixed Point
            print("\nNote: Enter function in form g(x) where x = g(x)")
            print("Example: For x^2 - 4 = 0, use sqrt(4) or (x+4/x)/2")
            func_str = input("Enter function g(x): ").strip()
            x0 = float(input("Enter initial guess (x0): "))
            tol = float(input("Enter tolerance (default 1e-6): ") or 1e-6)
            max_iter = int(input("Enter max iterations (default 100): ") or 100)
            result = solver.fixed_point_iteration(func_str, x0, tol, max_iter)
        
        elif choice == "6":  # Modified Secant
            x0 = float(input("Enter initial guess (x0): "))
            delta = float(input("Enter perturbation δ (default 0.01): ") or 0.01)
            tol = float(input("Enter tolerance (default 1e-6): ") or 1e-6)
            max_iter = int(input("Enter max iterations (default 100): ") or 100)
            result = solver.modified_secant_method(func_str, x0, delta, tol, max_iter)
        
        display_results(result)
        
    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    # Ask if user wants to continue
    again = input("\nWould you like to solve another equation? (y/n): ").strip().lower()
    if again == 'y':
        main()


if __name__ == "__main__":
    main()
